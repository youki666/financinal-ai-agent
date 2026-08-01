"""二级市场研究报告 ReAct Agent — 支持 SQLite 对话持久化"""
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Literal

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from agent.tools.agent_tools import rag_summarize, stock_quote_realtime, stock_history
from agent.tools.search_tools import web_search
from agent.tools.middleware import (
    monitor_tool,
    retrieval_quality_guard,
    citation_tracker,
    log_before_model,
    report_prompt_switch,
    response_quality_guard,
    hitl_guard,
)
from langchain.agents.middleware import (SummarizationMiddleware,
                                         ContextEditingMiddleware, ToolCallLimitMiddleware,
                                         ModelCallLimitMiddleware, PIIMiddleware, ModelFallbackMiddleware,
                                         LLMToolSelectorMiddleware, ToolRetryMiddleware, ModelRetryMiddleware,
                                         LLMToolEmulator,
                                         FilesystemFileSearchMiddleware, AgentMiddleware, ModelRequest
                                         )

# 2. 定义中间件，在 wrap_model_call 中注入工具
class DynamicToolMiddleware(AgentMiddleware):
    def wrap_model_call(self, request: ModelRequest, handler):
        # 关键：通过 request.override 注入新工具
        # 将新工具添加到现有工具列表的末尾
        if request.runtime.context["report"]:

            updated_request = request.override(
                tools=[*request.tools, stock_quote_realtime,stock_history]
            )
            logger.warning("添加工具成功")
            # 必须调用 handler 并传入更新后的请求
            return handler(updated_request)
        return handler(request)


dynamic_tool = DynamicToolMiddleware()

from dotenv import load_dotenv
load_dotenv()



# 1. 初始化一个用于生成摘要的模型

def _build_summarization_mw() -> SummarizationMiddleware:
    model = init_chat_model(
        model="qwen-plus",
        model_provider="openai",
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.0,
    )

    trigger: tuple[Literal["tokens"], int] = ("tokens", 4000)
    keep: tuple[Literal["messages"], int] = ("messages", 20)
    # 2. 创建 SummarizationMiddleware 实例
    return SummarizationMiddleware(
        model=model,
        trigger=trigger,
        keep=keep,
    )


DB_PATH = get_abs_path("data/conversations.db")


class ReactAgent:

    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_threads_table()
        self.checkpointer = SqliteSaver(self.conn)
        self._hitl_context: dict | None = None  # 存储待审批的 HITL 信息

        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize,stock_quote_realtime, stock_history,
                   web_search],
            middleware=[
                hitl_guard,          # 最先触发，需要审批的工具在此暂停
                monitor_tool,
                retrieval_quality_guard,
                citation_tracker,
                log_before_model,
                report_prompt_switch,
                response_quality_guard,
                _build_summarization_mw(),
                # 2. 然后用轻量模型筛选出最相关的3个工具
                #LLMToolSelectorMiddleware(max_tools=10),
                # 3. 限制模型调用次数，控制成本
                ModelCallLimitMiddleware(run_limit=10),
                # 4. 限制工具调用次数，防止死循环
                ToolCallLimitMiddleware(run_limit=5),

            ],
            checkpointer=self.checkpointer,
        )

    # ================================================================
    # 线程元数据表（存标题、时间戳）
    # ================================================================
    def _init_threads_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id   TEXT PRIMARY KEY,
                title       TEXT DEFAULT '新对话',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        self.conn.commit()

    # ================================================================
    # 线程管理
    # ================================================================
    def create_thread(self, title: str = "新对话") -> str:
        thread_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO threads (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (thread_id, title, now, now),
        )
        self.conn.commit()
        logger.info(f"[Thread] 创建对话: {thread_id} ({title})")
        return thread_id

    def list_threads(self) -> list[dict]:
        """返回所有线程，按更新时间倒序"""
        rows = self.conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM threads ORDER BY updated_at DESC"
        ).fetchall()
        return [
            {
                "thread_id": r[0],
                "title": r[1],
                "created_at": r[2],
                "updated_at": r[3],
            }
            for r in rows
        ]

    def update_thread_title(self, thread_id: str, title: str):
        self.conn.execute(
            "UPDATE threads SET title = ?, updated_at = ? WHERE thread_id = ?",
            (title, datetime.now().isoformat(), thread_id),
        )
        self.conn.commit()

    def delete_all_threads(self):
        """清空所有对话线程和检查点"""
        self.conn.execute("DELETE FROM threads")
        for table in ("checkpoints", "writes"):
            try:
                self.conn.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass
        self.conn.commit()
        logger.info("[Thread] 清空全部历史对话")

    def touch_thread(self, thread_id: str):
        """更新线程的最后活跃时间"""
        self.conn.execute(
            "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
            (datetime.now().isoformat(), thread_id),
        )
        self.conn.commit()

    def load_messages(self, thread_id: str) -> list[dict]:
        """加载指定线程的历史消息（供前端渲染用）"""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.agent.get_state(config)
        if not state or not state.values:
            return []

        raw_messages = state.values.get("messages", [])
        messages = []
        for msg in raw_messages:
            content = msg.content if hasattr(msg, "content") else str(msg)
            if not content:
                continue
            msg_type = type(msg).__name__
            if msg_type == "HumanMessage":
                messages.append({"role": "user", "content": content})
            elif msg_type == "AIMessage":
                messages.append({"role": "assistant", "content": content, "sources": []})
            # ToolMessage 不需要展示给用户
        return messages

    def auto_title(self, thread_id: str, query: str):
        """用第一条用户消息的前 20 字作为对话标题"""
        title = query[:20] + ("..." if len(query) > 20 else "")
        self.conn.execute(
            "UPDATE threads SET title = ?, updated_at = ? WHERE thread_id = ?",
            (title, datetime.now().isoformat(), thread_id),
        )
        self.conn.commit()

    # ================================================================
    # 流式执行
    # ================================================================
    def execute_stream(self, query: str, thread_id: str | None = None):
        """流式执行。thread_id 为 None 时自动创建新线程"""
        if not thread_id:
            thread_id = self.create_thread()

        # 第一条用户消息设为标题
        state = self.agent.get_state({"configurable": {"thread_id": thread_id}})
        if not state or not state.values or not state.values.get("messages"):
            self.auto_title(thread_id, query)

        input_dict = {
            "messages": [{"role": "user", "content": query}],
        }
        config = {"configurable": {"thread_id": thread_id}}

        # 工具名 → 用户可见的描述
        _tool_labels = {
            "rag_summarize": "正在检索研究报告...",
            "stock_brief": "正在查询股票概况...",
            "industry_overview": "正在分析行业数据...",
            "generate_report": "正在生成研究报告...",
            "stock_quote_realtime": "正在获取实时行情...",
            "stock_history": "正在查询历史走势...",
            "financial_news": "正在检索最新财经新闻...",
            "flash_news": "正在获取市场实时快讯...",
            "web_search": "正在联网搜索最新信息...",
        }

        def _tool_label(tool_name: str) -> str:
            return _tool_labels.get(tool_name, f"正在执行 {tool_name}...")

        try:
            for msg, metadata in self.agent.stream(
                    input_dict,
                    stream_mode="messages",
                    context={"report": False},
                    config=config,
            ):

                msg_type = type(msg).__name__

                # 工具执行完成 → 显示用户友好的操作提示
                if msg_type == "ToolMessage":
                    tool_name = getattr(msg, "name", "") or ""
                    label = _tool_label(tool_name)
                    yield f"\n> {label}\n\n"
                    continue

                # 模型逐 token 输出
                if msg_type == "AIMessageChunk":
                    # 外层 RoutableChatModel 包装会产生与内层 ChatOpenAI 重复的回调，
                    # 只保留内层模型（ls_provider != "routable-chat-model"）的 token
                    if metadata.get("ls_provider") == "routable-chat-model":
                        continue
                    has_tool_calls = getattr(msg, "tool_calls", None)
                    if has_tool_calls:
                        continue
                    content = getattr(msg, "content", "")
                    if isinstance(content, list):
                        content = "".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    if content:
                        yield content

            # 检查是否因 interrupt() 暂停（HITL）
            graph_state = self.agent.get_state(config)
            if graph_state and getattr(graph_state, "interrupts", None):
                first_interrupt = graph_state.interrupts[0]
                self._hitl_context = {
                    "thread_id": thread_id,
                    "interrupt_id": getattr(first_interrupt, "id", None),
                    "tool": first_interrupt.value.get("tool", ""),
                    "args": first_interrupt.value.get("args", {}),
                }
                logger.info(f"[HITL] 已暂停, 等待审批: {first_interrupt.value.get('tool')}")
            else:
                self.touch_thread(thread_id)
        except Exception as e:
            err = str(e) or type(e).__name__
            logger.error(f"[Agent] 执行异常: {type(e).__name__}: {err}", exc_info=True)

            # 识别常见 API 错误，给出可操作的提示
            low = err.lower()
            if "quota" in low and ("exhaust" in low or "free" in low):
                yield (
                    "模型 API 免费额度已用尽，请前往 [阿里云百炼控制台]"
                    "(https://bailian.console.aliyun.com/) 充值或关闭「仅使用免费额度」选项。\n\n"
                    "临时替代方案：编辑 `.env` 文件，将模型切换为其他可用模型。"
                )
            elif "403" in err or "forbidden" in low:
                yield (
                    f"模型 API 访问被拒绝（403），请检查 API Key 是否有效"
                    f"及对应模型是否已开通付费权限。\n\n"
                    f"详细错误: {err[:200]}"
                )
            elif "401" in err or "unauthorized" in low or "authentication" in low:
                yield (
                    "模型 API 认证失败，请检查 `.env` 文件中 DASHSCOPE_API_KEY 是否正确配置。"
                )
            elif "timeout" in low or "timed out" in low:
                yield "模型响应超时，请稍后重试。如持续出现此问题，可尝试切换为 qwen-turbo 等更轻量的模型。"
            else:
                yield f"分析过程中出现错误，请稍后重试。如需帮助，请检查运行日志。({type(e).__name__})"

    # ============================================================
    # HITL 人工介入方法
    # ============================================================
    def has_pending_interrupt(self) -> bool:
        return self._hitl_context is not None

    def get_interrupt_info(self) -> dict | None:
        return self._hitl_context

    def resume_stream(self, approved: bool):
        """恢复被 interrupt() 暂停的流，传入人工决策"""
        if not self._hitl_context:
            yield "没有待处理的审批请求。"
            return

        thread_id = self._hitl_context["thread_id"]
        interrupt_id = self._hitl_context.get("interrupt_id")
        config = {"configurable": {"thread_id": thread_id}}
        self._hitl_context = None

        _tool_labels = {
            "rag_summarize": "正在检索研究报告...",
            "stock_brief": "正在查询股票概况...",
            "industry_overview": "正在分析行业数据...",
            "generate_report": "正在生成研究报告...",
            "stock_quote_realtime": "正在获取实时行情...",
            "stock_history": "正在查询历史走势...",
            "financial_news": "正在检索最新财经新闻...",
            "flash_news": "正在获取市场实时快讯...",
            "web_search": "正在联网搜索最新信息...",
        }

        def _tool_label(tool_name: str) -> str:
            return _tool_labels.get(tool_name, f"正在执行 {tool_name}...")

        try:
            from langgraph.types import Command
            # 有 interrupt_id 时用 dict 形式，避免多个 pending interrupt 报错
            resume_value = {"approved": approved}
            cmd = Command(
                resume={interrupt_id: resume_value} if interrupt_id else resume_value,
            )
            for msg, metadata in self.agent.stream(
                    cmd,
                    stream_mode="messages",
                    context={"report": False},
                    config=config,
            ):
                msg_type = type(msg).__name__

                if msg_type == "ToolMessage":
                    tool_name = getattr(msg, "name", "") or ""
                    label = _tool_label(tool_name)
                    yield f"\n> {label}\n\n"
                    continue

                if msg_type == "AIMessageChunk":
                    if metadata.get("ls_provider") == "routable-chat-model":
                        continue
                    has_tool_calls = getattr(msg, "tool_calls", None)
                    if has_tool_calls:
                        continue
                    content = getattr(msg, "content", "")
                    if isinstance(content, list):
                        content = "".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    if content:
                        yield content

            self.touch_thread(thread_id)

            # 再次检查是否有新的 interrupt（如连续多次调用同一工具）
            graph_state = self.agent.get_state(config)
            if graph_state and getattr(graph_state, "interrupts", None):
                first = graph_state.interrupts[0]
                self._hitl_context = {
                    "thread_id": thread_id,
                    "interrupt_id": getattr(first, "id", None),
                    "tool": first.value.get("tool", ""),
                    "args": first.value.get("args", {}),
                }

        except Exception as e:
            err = str(e) or type(e).__name__
            logger.error(f"[Agent] 恢复执行异常: {type(e).__name__}: {err}", exc_info=True)
            yield f"恢复执行时出错（{err[:100]}）。"

    def close(self):
        self.conn.close()


if __name__ == '__main__':
    agent = ReactAgent()

    # 列出已有线程
    print("已有对话:")
    for t in agent.list_threads():
        print(f"  [{t['thread_id']}] {t['title']}  ({t['updated_at']})")

    # 新对话
    tid = agent.create_thread("测试对话")
    print(f"\n新对话 ID: {tid}")

    for chunk in agent.execute_stream("帮我分析一下紫金矿业", thread_id=tid):
        print(chunk, end="", flush=True)
    print()

    # 加载历史
    print("\n加载历史消息:")
    for msg in agent.load_messages(tid):
        print(f"  [{msg['role']}] {msg['content'][:80]}...")
