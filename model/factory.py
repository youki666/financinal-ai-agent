import os
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.tools import BaseTool

from utils.config_handler import rag_conf
from utils.logger_handler import logger

load_dotenv(override=True)

# ============================================================
# API Key
# ============================================================
def _get_api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY", "")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("DASHSCOPE_API_KEY", "")
    except Exception:
        return ""


# ============================================================
# Embedding
# ============================================================
_embed_model_cache: DashScopeEmbeddings | None = None


def get_embed_model() -> DashScopeEmbeddings:
    global _embed_model_cache
    if _embed_model_cache is None:
        _embed_model_cache = DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=_get_api_key(),
        )
    return _embed_model_cache


# ============================================================
# 1. LLM 配置数据类
# ============================================================
@dataclass
class LLMConfig:
    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str = ""
    base_url: str = ""
    description: str = ""


# ============================================================
# 2. 模型路由器
# ============================================================
class ModelRouter:
    """根据查询意图自动选择目标模型"""

    def __init__(self, configs: dict[str, LLMConfig]):
        self.configs = configs
        self._default = configs.get("standard", list(configs.values())[0])

    def route(self, query: str) -> LLMConfig:
        q = query.lower()

        report_keywords = ["报告", "研报", "生成一份", "撰写", "综合分析", "深度"]
        if any(kw in q for kw in report_keywords):
            config = self.configs.get("powerful", self._default)
            logger.info(f"[ModelRouter] → {config.description}")
            return config

        analysis_keywords = [
            "分析", "投资", "估值", "风险", "行业", "产业链",
            "政策", "展望", "走势", "财务", "基本面", "技术面",
            "竞争力", "龙头", "对标",
        ]
        if any(kw in q for kw in analysis_keywords):
            config = self.configs.get("standard", self._default)
            logger.info(f"[ModelRouter] → {config.description}")
            return config

        config = self.configs.get("fast", self._default)
        logger.info(f"[ModelRouter] → {config.description}")
        return config


# ============================================================
# 3. 可路由 ChatModel
# ============================================================
def _extract_query(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, HumanMessage):
            continue
        content = msg.content if hasattr(msg, "content") else str(msg)
        if not content:
            continue
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
            return " ".join(parts)
    return ""


def _build_model(config: LLMConfig) -> BaseChatModel:
    api_key = config.api_key or _get_api_key()
    kwargs: dict = {
        "model": config.model_name,
        "temperature": config.temperature,
        "model_provider": config.provider,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return init_chat_model(**kwargs, timeout=60, max_retries=1)


class RoutableChatModel(BaseChatModel):
    """继承 BaseChatModel，每次调用前路由到目标模型，对上层透明"""

    router: ModelRouter
    _bound_tools: list | None = None
    _bound_tools_kwargs: dict = {}

    def __init__(self, router: ModelRouter, **kwargs):
        super().__init__(router=router, **kwargs)
        self._model_cache: dict[str, BaseChatModel] = {}
        self._bound_tools = None
        self._bound_tools_kwargs = {}

    def _get_model(self, config: LLMConfig) -> BaseChatModel:
        suffix = "_bound" if self._bound_tools else ""
        cache_key = f"{config.provider}:{config.model_name}{suffix}"
        if cache_key not in self._model_cache:
            model = _build_model(config)
            if self._bound_tools:
                model = model.bind_tools(self._bound_tools, **self._bound_tools_kwargs)
            self._model_cache[cache_key] = model
        return self._model_cache[cache_key]

    def _route(self, messages: list[BaseMessage]) -> BaseChatModel:
        query = _extract_query(messages)
        config = self.router.route(query)
        return self._get_model(config)

    def bind_tools(
        self, tools: Sequence[BaseTool], *, tool_choice: str | None = None, **kwargs,
    ) -> BaseChatModel:
        self._bound_tools = list(tools)
        self._bound_tools_kwargs = {"tool_choice": tool_choice, **kwargs}
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: Any = None, **kwargs,
    ) -> ChatResult:
        model = self._route(messages)
        result_msg = model.invoke(messages, stop=stop)
        return ChatResult(generations=[
            ChatGeneration(message=result_msg, generation_info={"finish_reason": "stop"})
        ])

    def _stream(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: Any = None, **kwargs,
    ) -> Iterator[Any]:
        model = self._route(messages)
        for chunk in model.stream(messages, stop=stop):
            if type(chunk).__name__ == "AIMessageChunk":
                chunk = ChatGenerationChunk(
                    message=chunk, generation_info={"finish_reason": "stop"},
                )
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "routable-chat-model"

    @property
    def _identifying_params(self) -> dict:
        return {"router_configs": list(self.router.configs.keys())}


# ============================================================
# 4. 模型配置 & 全局实例
# ============================================================
_model_configs: dict[str, LLMConfig] = {
    "fast": LLMConfig(
        provider="openai",
        model_name="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.0,
        description="快速模型 (qwen-turbo) — 简单问答",
    ),
    "standard": LLMConfig(
        provider="openai",
        model_name="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.0,
        description="标准模型 (qwen-plus) — 个股/行业分析",
    ),
    "powerful": LLMConfig(
        provider="openai",
        model_name="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.0,
        description="强力模型 (qwen-max) — 研报生成/复杂分析",
    ),
}

chat_model = RoutableChatModel(router=ModelRouter(configs=_model_configs))
