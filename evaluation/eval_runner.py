"""RAGAS 评估运行器：核心评估逻辑"""

import asyncio
import copy
import sys
from dataclasses import dataclass, field

from datasets import Dataset as HFDataset

from evaluation.test_dataset import load_dataset
from rag.rag_service import RagSummarizeService
from utils.logger_handler import logger


def _ensure_ragas_importable():
    """修复 ragas 在新版 langchain-community 下的 import 兼容性问题。

    ragas 依赖旧路径 `langchain_community.chat_models.vertexai.ChatVertexAI`，
    该模块已从新版 langchain-community 移除，需从 `langchain_google_vertexai` 桥接。
    """
    if "langchain_community.chat_models.vertexai" in sys.modules:
        return
    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError:
        raise ImportError(
            "ragas 评估需要安装 langchain-google-vertexai，请执行: "
            "pip install langchain-google-vertexai"
        )

    mod = type(sys)("langchain_community.chat_models.vertexai")
    mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = mod


_ensure_ragas_importable()

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.metrics._faithfulness import NLIStatementPrompt, StatementGeneratorPrompt
from ragas.metrics._answer_relevance import ResponseRelevancePrompt
from ragas.metrics._context_recall import ContextRecallClassificationPrompt
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


AVAILABLE_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_recall": context_recall,
}

_chinese_metrics_cache: dict | None = None


async def _adapt_prompts_to_chinese(llm):
    """将三个指标的 prompt 模板翻译为中文（仅翻译 examples，不翻译 instruction）。

    翻译 instruction 可能破坏 ragas 对输出格式的约束（如 JSON schema），
    因此只翻译 examples 让 LLM 在评估中文内容时参照中文示例。
    """
    adapted = {
        "faithfulness": copy.deepcopy(faithfulness),
        "answer_relevancy": copy.deepcopy(answer_relevancy),
        "context_recall": copy.deepcopy(context_recall),
    }

    # faithfulness: 翻译 NLI + statement generator 的 examples
    adapted["faithfulness"].nli_statements_prompt = (
        await faithfulness.nli_statements_prompt.adapt("chinese", llm)
    )
    adapted["faithfulness"].statement_generator_prompt = (
        await faithfulness.statement_generator_prompt.adapt("chinese", llm)
    )

    # answer_relevancy: 翻译 question_generation 的 examples
    adapted["answer_relevancy"].question_generation = (
        await answer_relevancy.question_generation.adapt("chinese", llm)
    )

    # context_recall: 翻译 context_recall_prompt 的 examples
    adapted["context_recall"].context_recall_prompt = (
        await context_recall.context_recall_prompt.adapt("chinese", llm)
    )

    return adapted


def _build_chinese_metrics(llm):
    """获取中文 prompt 版本的评估指标（首次调用时通过 LLM 翻译，后续走缓存）。"""
    global _chinese_metrics_cache
    if _chinese_metrics_cache is not None:
        return _chinese_metrics_cache

    logger.info("[Eval] 正在将 ragas prompt 翻译为中文（仅首次需要）...")

    loop = asyncio.new_event_loop()
    try:
        _chinese_metrics_cache = loop.run_until_complete(_adapt_prompts_to_chinese(llm))
    finally:
        loop.close()

    logger.info("[Eval] 中文 prompt 翻译完成并已缓存")
    return _chinese_metrics_cache


@dataclass
class EvalResult:
    metric_scores: dict[str, float] = field(default_factory=dict)
    per_question: list[dict] = field(default_factory=list)


def run_evaluation(
    dataset_path: str | None = None,
    metrics: list[str] | None = None,
    language: str = "english",
) -> EvalResult:
    """执行 RAGAS 评估。

    Args:
        dataset_path: 外部 JSON 数据集路径，为 None 时使用内置样本数据。
        metrics: 评估指标列表，为 None 时使用全部三个默认指标。
        language: prompt 模板语言，\"english\" 或 \"chinese\"。
                  中文模式下会通过 LLM 将 prompt 示例翻译为中文，使评估更适配中文内容。

    Returns:
        EvalResult: 包含 metric_scores（各指标均分）和 per_question（每题明细）。
    """
    if metrics is None:
        metrics = ["faithfulness", "answer_relevancy", "context_recall"]

    selected_names = [m for m in metrics if m in AVAILABLE_METRICS]
    if not selected_names:
        return EvalResult()

    test_data = load_dataset(dataset_path)

    rag = RagSummarizeService()
    llm_wrapper = LangchainLLMWrapper(rag.chain.middle[0])

    if language == "chinese":
        cn_metrics = _build_chinese_metrics(llm_wrapper)
        selected = [cn_metrics[m] for m in selected_names]
    else:
        selected = [AVAILABLE_METRICS[m] for m in selected_names]

    records = []
    for item in test_data:
        result = rag.summarize_with_context(item["question"])
        records.append({
            "user_input": item["question"],
            "response": result["answer"],
            "retrieved_contexts": result["contexts"],
            "reference": item["ground_truth"],
        })
        logger.info(
            f"[Eval] Q: {item['question'][:40]}... "
            f"contexts={len(result['contexts'])} answer_len={len(result['answer'])}"
        )

    hf_dataset = HFDataset.from_list(records)
    ragas_result = evaluate(
        dataset=hf_dataset,
        metrics=selected,
        llm=llm_wrapper,
        embeddings=LangchainEmbeddingsWrapper(
            rag.vector_store.vectors._embedding_function
        ),
    )

    df = ragas_result.to_pandas()

    scores = {}
    for m in metrics:
        if m in df.columns:
            scores[m] = float(df[m].mean())

    per_q = []
    for i, item in enumerate(test_data):
        entry = {"question": item["question"]}
        for m in metrics:
            if m in df.columns:
                entry[m] = float(df[m].iloc[i])
        per_q.append(entry)

    return EvalResult(metric_scores=scores, per_question=per_q)
