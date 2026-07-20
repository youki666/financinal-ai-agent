"""RAG 总结服务：查询重写 → 混合检索 → 重排序 → 上下文组装 → LLM 总结"""
import re
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from rag.vector_store import VectorStoreService
from rag.query_rewriter import query_rewriter
from rag.reranker import reranker
from rag.context_assembler import context_assembler
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model
from utils.logger_handler import logger


def _check_context_relevance(query: str, context_docs: list[Document]) -> bool:
    """验证检索到的文档是否真的与查询相关。

    n-gram 重叠检查：查询中的 3-4 字子串至少有一个出现在检索文档中。
    如果完全无交集，说明向量检索命中了语义相近但内容无关的文档。

    极端短的查询（<4 个中文字）放行，因为此时无法生成有效的 n-gram。

    误拦的代价（返回"暂未检索到"）远低于误放的代价（用无关文档编造答案），
    因此宁严勿松。
    """
    # 去标点/数字/英文，只留中文
    clean = re.sub(r"[\d\s，,。\.！!？?：:；;、""''（）()a-zA-Z]+", "", query)
    if len(clean) < 4:
        return True

    all_text = "".join(doc.page_content for doc in context_docs)

    # 股票代码精确匹配
    codes = re.findall(r"(?<!\d)(\d{6})(?!\d)", query)
    for code in codes:
        if code not in all_text:
            logger.warning(f"[RAG] 检索结果不含股票代码 {code}")
            return False

    # 3-4 gram 集合
    grams = set()
    for n in (3, 4):
        for i in range(len(clean) - n + 1):
            grams.add(clean[i:i + n])

    found = [g for g in grams if g in all_text]

    if not found:
        logger.warning(
            f"[RAG] 检索结果与查询无交集，拒绝回答: "
            f"q={query[:30]}..., grams_missing={sorted(grams)[:5]}..."
        )
        return False

    return True


class RagSummarizeService:

    def __init__(self):
        self.vector_store = VectorStoreService()
        self.chain = (
            PromptTemplate.from_template(load_rag_prompts())
            | chat_model
            | StrOutputParser()
        )

    def _retrieve(self, query: str, k: int = 5) -> list[Document]:
        """混合检索 + 重排序"""
        docs = self.vector_store.hybrid_search(query, k=k)
        if not docs:
            return []
        return reranker.rerank(query, docs, top_k=3)

    def retrieve_docs(self, query: str) -> list[Document]:
        """对外暴露的检索接口：多查询融合检索"""
        sub_queries = query_rewriter.rewrite(query)

        all_docs: dict[str, Document] = {}
        for sq in sub_queries:
            docs = self._retrieve(sq, k=3)
            for doc in docs:
                key = doc.page_content[:100]
                if key not in all_docs:
                    all_docs[key] = doc

        return list(all_docs.values())

    def _filter_irrelevant_docs(self, query: str, docs: list[Document]) -> list[Document]:
        """逐篇过滤：只保留与查询有 n-gram 交集的文档。

        全局 `_check_context_relevance` 检查整体是否有交集，
        这个函数进一步逐篇过滤，确保每篇文档都至少包含一个查询 n-gram。
        被过滤掉的文档不会进入上下文，LLM 自然也不会引用它们。
        """
        if not docs:
            return docs

        clean = re.sub(r"[\d\s，,。\.！!？?：:；;、""''（）()a-zA-Z]+", "", query)
        if len(clean) < 4:
            return docs

        grams = set()
        for n in (3, 4):
            for i in range(len(clean) - n + 1):
                grams.add(clean[i:i + n])

        kept = []
        for doc in docs:
            if any(g in doc.page_content for g in grams):
                kept.append(doc)

        if len(kept) < len(docs):
            logger.info(
                f"[RAG] 逐篇过滤: {len(docs)} → {len(kept)} 篇 "
                f"(移除 {len(docs) - len(kept)} 篇无关文档)"
            )

        # 至少保留一篇（极端情况下全局检查通过了但逐篇全挂，留一篇给 LLM）
        return kept if kept else docs[:1]

    def summarize_with_context(self, query: str) -> dict:
        """完整的 RAG 总结流程，同时返回检索到的上下文文档列表（供评估使用）。

        Returns:
            dict: {"answer": str, "contexts": list[str]}
        """
        logger.info(f"[RAG] [Eval] 开始处理: '{query}'")

        context_docs = self.retrieve_docs(query)
        logger.info(f"[RAG] [Eval] 检索到 {len(context_docs)} 篇相关文档")

        if not context_docs or not _check_context_relevance(query, context_docs):
            return {
                "answer": (
                    "当前知识库中暂未检索到与您问题直接相关的资料。"
                    "建议：1) 尝试使用更具体的检索词 2) 上传相关研究报告 PDF 到知识库"
                ),
                "contexts": [],
            }

        context_docs = self._filter_irrelevant_docs(query, context_docs)
        context = context_assembler.assemble(query, context_docs)
        result = self.chain.invoke({"input": query, "context": context})
        return {
            "answer": result,
            "contexts": [doc.page_content for doc in context_docs],
        }

    def summarize(self, query: str) -> str:
        """完整的 RAG 总结流程"""
        logger.info(f"[RAG] 开始处理: '{query}'")

        context_docs = self.retrieve_docs(query)
        logger.info(f"[RAG] 检索到 {len(context_docs)} 篇相关文档")

        if not context_docs or not _check_context_relevance(query, context_docs):
            return (
                "当前知识库中暂未检索到与您问题直接相关的资料。"
                "建议：1) 尝试使用更具体的检索词 2) 上传相关研究报告 PDF 到知识库"
            )

        context_docs = self._filter_irrelevant_docs(query, context_docs)
        context = context_assembler.assemble(query, context_docs)
        result = self.chain.invoke({"input": query, "context": context})
        return result


if __name__ == '__main__':
    rag = RagSummarizeService()

    queries = [
        "紫金矿业股价",
        "铜价走势分析",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"查询: {q}")
        print("=" * 60)
        result = rag.summarize(q)
        print(result)
        print()
