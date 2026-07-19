"""模型路由器：根据查询内容动态选择合适的 LLM"""

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.tools import BaseTool
from utils.logger_handler import logger


# ============================================================
# 1. LLM 配置数据类
# ============================================================
@dataclass
class LLMConfig:
    """封装一个大模型的完整配置"""
    provider: str                        # "tongyi" | "deepseek"
    model_name: str                      # "qwen-max" | "qwen-plus" | "deepseek-v4-flash"
    temperature: float = 0.0
    api_key: str = ""
    base_url: str = ""
    description: str = ""                # 该模型的用途说明


# ============================================================
# 2. 模型路由器
# ============================================================
class ModelRouter:
    """根据查询内容和上下文，返回最合适的 LLMConfig"""

    def __init__(self, configs: dict[str, LLMConfig]):
        """
        configs: {"fast": LLMConfig(...), "standard": ..., "powerful": ...}
        """
        self.configs = configs
        self._default = configs.get("standard", list(configs.values())[0])

    def route(self, query: str, context: dict | None = None) -> LLMConfig:
        return self._rule_based_route(query)

    def _rule_based_route(self, query: str) -> LLMConfig:
        """关键词 + 意图识别的路由规则"""
        q = query.lower()

        # 规则 1: 研报生成 / 复杂分析 → 最强模型
        report_keywords = ["报告", "研报", "生成一份", "撰写", "综合分析", "深度"]
        if any(kw in q for kw in report_keywords):
            config = self.configs.get("powerful", self._default)
            logger.info(f"[ModelRouter] → {config.description} (理由: 研报/复杂分析)")
            return config

        # 规则 2: 行业分析 / 个股分析 → 标准模型
        analysis_keywords = [
            "分析", "投资", "估值", "风险", "行业", "产业链",
            "政策", "展望", "走势", "财务", "基本面", "技术面",
            "竞争力", "龙头", "对标",
        ]
        if any(kw in q for kw in analysis_keywords):
            config = self.configs.get("standard", self._default)
            logger.info(f"[ModelRouter] → {config.description} (理由: 分析类查询)")
            return config

        # 规则 3: 默认 → 快速模型
        config = self.configs.get("fast", self._default)
        logger.info(f"[ModelRouter] → {config.description} (理由: 默认)")
        return config


# ============================================================
# 3. 可路由的 ChatModel 包装器
# ============================================================
class RoutableChatModel(BaseChatModel):
    """
    实现 BaseChatModel 接口，内部持有多模型实例池，
    每次 invoke/stream 前调用 Router 选择目标模型并委托执行。

    这样对上层 Agent/RAG 完全透明 —— 它们只需调用这一个模型对象。
    """

    router: ModelRouter
    _bound_tools: list | None = None
    _bound_tools_kwargs: dict = {}

    def __init__(self, router: ModelRouter, **kwargs):
        super().__init__(router=router, **kwargs)
        self._model_cache: dict[str, BaseChatModel] = {}
        self._bound_tools: list | None = None
        self._bound_tools_kwargs: dict = {}

    def _get_model(self, config: LLMConfig) -> BaseChatModel:
        """按模型名 + 工具绑定状态缓存实例，避免每次 bind_tools 重复创建"""
        # bound 和 unbound 分开缓存，避免每次请求都重新 bind_tools
        suffix = "_bound" if self._bound_tools else ""
        cache_key = f"{config.provider}:{config.model_name}{suffix}"

        if cache_key not in self._model_cache:
            model = self._build_model(config)
            if self._bound_tools:
                model = model.bind_tools(self._bound_tools, **self._bound_tools_kwargs)
            self._model_cache[cache_key] = model

        return self._model_cache[cache_key]

    def _build_model(self, config: LLMConfig) -> BaseChatModel:
        kwargs: dict = {
            "model": config.model_name,
            "temperature": config.temperature,
            "model_provider": config.provider,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return init_chat_model(**kwargs)

    def bind_tools(
            self,
            tools: Sequence[BaseTool],
            *,
            tool_choice: str | None = None,
            **kwargs,
    ) -> BaseChatModel:
        """重写：保存工具信息并返回自身，在 _generate/_stream 时动态绑定到路由后的模型"""
        self._bound_tools = list(tools)
        self._bound_tools_kwargs = {"tool_choice": tool_choice, **kwargs}
        return self  # 返回自身，保持路由能力

    def _extract_query(self, messages: list[BaseMessage]) -> str:
        """从消息列表中提取用户最后一条文本"""
        for msg in reversed(messages):
            content = msg.content if hasattr(msg, "content") else str(msg)
            if not content:
                continue
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
                return " ".join(parts)
        return ""

    # ---- 核心：路由逻辑 ----
    # _get_model 返回的 RunnableBinding（bind_tools 产物）只通过 invoke 做工具注入。
    # _generate 改调用 model.invoke() 而非 model._generate() 来保证工具正常工作。
    def _route(self, messages: list[BaseMessage]) -> BaseChatModel:
        query = self._extract_query(messages)
        config = self.router.route(query)
        return self._get_model(config)

    def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs,
    ) -> ChatResult:
        model = self._route(messages)
        # 必须用 invoke：RunnableBinding._generate 不做工具注入，invoke 才做
        result_msg = model.invoke(messages, stop=stop)
        generation = ChatGeneration(
            message=result_msg,
            generation_info={"finish_reason": "stop"},
        )
        return ChatResult(generations=[generation])

    def _stream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs,
    ) -> Iterator[Any]:
        model = self._route(messages)
        for chunk in model.stream(messages, stop=stop):
            # _generate_with_cache 流式路径期望 ChatGenerationChunk（有 generation_info），
            # 底层 ChatOpenAI stream 返回 AIMessageChunk，需要包装
            if type(chunk).__name__ == "AIMessageChunk":
                chunk = ChatGenerationChunk(
                    message=chunk,
                    generation_info={"finish_reason": "stop"},
                )
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "routable-chat-model"

    @property
    def _identifying_params(self) -> dict:
        return {"router_configs": list(self.router.configs.keys())}
