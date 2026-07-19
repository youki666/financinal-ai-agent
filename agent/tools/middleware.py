"""增强中间件：查询重写监控、检索质量监控、引用溯源、兜底策略"""
from typing import Callable
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, after_model, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts, load_report_prompts


# ============================================================
# 1. 工具调用监控中间件
# ============================================================
@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})

    logger.info(f"[ToolMonitor] 执行工具: {tool_name}")
    logger.info(f"[ToolMonitor] 入参: {tool_args}")

    try:
        result = handler(request)
        # 截断过长结果用于日志
        result_preview = str(result)[:300] if result else "(empty)"
        logger.info(f"[ToolMonitor] {tool_name} 调用成功, 结果预览: {result_preview}")

        # 报告生成场景标记
        if tool_name == "generate_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"[ToolMonitor] {tool_name} 调用失败: {e}", exc_info=True)
        raise e


# ============================================================
# 2. 检索质量监控中间件
# ============================================================
@wrap_tool_call
def retrieval_quality_guard(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """当 RAG 检索结果为空或过短时记录告警"""
    result = handler(request)

    if request.tool_call["name"] in ("rag_summarize", "stock_brief", "industry_overview"):
        content = ""
        if isinstance(result, ToolMessage):
            content = result.content
        elif isinstance(result, str):
            content = result

        if not content or len(content) < 20:
            logger.warning(
                f"[QualityGuard] {request.tool_call['name']} 检索结果为空或过短, "
                f"query={request.tool_call.get('args', {})}"
            )
        elif "暂未检索到" in content or "未找到" in content:
            logger.warning(
                f"[QualityGuard] {request.tool_call['name']} 未检索到有效内容, "
                f"query={request.tool_call.get('args', {})}"
            )

    return result


# ============================================================
# 3. 引用溯源中间件
# ============================================================
from collections import defaultdict

# 跨工具调用的引用累积（同一对话中）
_citation_store: dict[str, list[str]] = defaultdict(list)


@wrap_tool_call
def citation_tracker(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """追踪工具调用的来源并累积引用"""
    result = handler(request)

    if request.tool_call["name"] in ("rag_summarize", "stock_brief", "industry_overview", "generate_report"):
        content = ""
        if isinstance(result, ToolMessage):
            content = result.content
        elif isinstance(result, str):
            content = result

        # 提取来源标记（匹配多种格式：参考资料：[文件名]、来源：[文件名]）
        import re
        sources = re.findall(r'【来源[:：]([^】]+)】', content)
        if not sources:
            sources = re.findall(r'来源[:：]([^】\n]+)', content)
        if not sources:
            # 新格式：> 参考资料：xxx.pdf
            sources = re.findall(r'参考资料[:：]\s*(.+?)(?:\n|$)', content)

        thread_id = getattr(request.runtime, "thread_id", "default")
        for src in sources:
            if src not in _citation_store[thread_id]:
                _citation_store[thread_id].append(src)

        if sources:
            logger.info(f"[CitationTracker] 追踪到 {len(sources)} 个来源")

    return result


def get_citations(thread_id: str = "default") -> list[str]:
    """获取累积的引用列表"""
    return _citation_store.get(thread_id, [])


def clear_citations(thread_id: str = "default"):
    """清除累积的引用"""
    _citation_store.pop(thread_id, None)


# ============================================================
# 4. 模型调用前日志中间件
# ============================================================
@before_model
def log_before_model(
        state: AgentState,
        runtime: Runtime,
):
    msg_count = len(state.get("messages", []))
    last_msg = state["messages"][-1] if state.get("messages") else None
    msg_type = type(last_msg).__name__ if last_msg else "N/A"
    logger.info(f"[BeforeModel] 即将调用模型, 消息数={msg_count}, 最后消息类型={msg_type}")
    return None


# ============================================================
# 5. 动态提示词切换中间件
# ============================================================
@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        logger.info("[PromptSwitch] 切换到研究报告提示词")
        return load_report_prompts()
    return load_system_prompts()


# ============================================================
# 6. 响应质量守卫中间件
# ============================================================
@after_model
def response_quality_guard(
        state: AgentState,
        runtime: Runtime,
) -> dict | None:
    """检查模型输出质量（引用检查、术语检查）"""
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    content_str = content if isinstance(content, str) else ""

    if not content_str.strip():
        logger.warning("[ResponseGuard] 模型输出为空")
        return None

    # 金融术语合理性检查
    suspicious_terms = ["暴涨", "暴跌", "保证收益", "稳赚", "必涨", "必跌"]
    for term in suspicious_terms:
        if term in content_str:
            logger.warning(f"[ResponseGuard] 检测到不专业表述: '{term}'")

    # 长度检查
    if len(content_str) < 30:
        logger.warning(f"[ResponseGuard] 输出过短 ({len(content_str)} 字符)")

    return None
