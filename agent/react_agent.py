"""二级市场研究报告 ReAct Agent — 支持 SQLite 对话持久化"""
import sqlite3
import uuid
from datetime import datetime

from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from agent.tools.agent_tools import rag_summarize, stock_brief, industry_overview, generate_report
from agent.tools.middleware import (
    monitor_tool,
    retrieval_quality_guard,
    citation_tracker,
    log_before_model,
    report_prompt_switch,
    response_quality_guard,
)

DB_PATH = get_abs_path("data/conversations.db")


class ReactAgent:

    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_threads_table()
        self.checkpointer = SqliteSaver(self.conn)

        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, stock_brief, industry_overview, generate_report],
            middleware=[
                monitor_tool,
                retrieval_quality_guard,
                citation_tracker,
                log_before_model,
                report_prompt_switch,
                response_quality_guard,
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

        try:
            # stream_mode="messages" 逐 token 实时输出
            has_tool_started = False  # 首轮思考阶段（含工具名）不展示
            for msg, metadata in self.agent.stream(
                    input_dict,
                    stream_mode="messages",
                    context={"report": False},
                    config=config,
            ):
                msg_type = type(msg).__name__

                # # 工具执行完成
                # if msg_type == "ToolMessage":
                #     has_tool_started = True
                #     yield "\n> 正在检索资料...\n\n"
                #     continue
                #
                # # 首轮思考阶段（调用工具前的自言自语）不展示，避免暴露工具名
                # if not has_tool_started:
                #     continue

                # 模型逐 token 输出
                if msg_type == "AIMessageChunk":
                    # 外层 RoutableChatModel 包装会产生与内层 ChatOpenAI 重复的回调，
                    # 只保留内层模型（ls_provider != "routablechatmodel"）的 token
                    if metadata.get("ls_provider") == "routablechatmodel":
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
        except Exception as e:
            err = str(e) or type(e).__name__
            logger.error(f"[Agent] 执行异常: {type(e).__name__}: {err}", exc_info=True)
            yield f"分析过程中出现错误（{err}）。请稍后重试或尝试其他查询。"

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
