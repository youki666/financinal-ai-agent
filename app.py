"""二级市场研究报告分析平台"""
import os
import re
import threading
import time
from datetime import datetime

import streamlit as st
from agent.react_agent import ReactAgent
from agent.tools.middleware import get_citations, clear_citations
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="ResearchAI · 二级市场研究报告分析",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 全局 CSS
# ============================================================
st.markdown("""
<style>
    /* ===== 隐藏 Streamlit 默认元素 ===== */
    #MainMenu          { visibility: hidden; }
    header             { visibility: hidden; }
    footer             { visibility: hidden; }
    .stDeployButton   { display: none; }
    #stDecoration     { display: none; }
    [data-testid="stToolbar"] { display: none; }
    [data-testid="stDecoration"] { display: none; }
    button[title="View fullscreen"] { display: none; }
    [data-testid="baseButton-header"] { display: none; }
    .stApp > header    { display: none !important; }

    /* ===== 根变量 ===== */
    :root {
        --bg-page:          #ffffff;
        --bg-card:          #fafafa;
        --bg-sidebar:       #f7f8fa;
        --border-light:     #e8eaed;
        --border-normal:    #d0d3d9;
        --accent:           #1a6fb5;
        --accent-light:     #e8f2fa;
        --accent-dark:      #12548a;
        --gold:             #c8972e;
        --gold-light:       #fdf6e8;
        --text-primary:     #1a1d23;
        --text-secondary:   #5a5f6b;
        --text-hint:        #9ba0ab;
        --red:              #d9304a;
        --green:            #0f8c5e;
        --shadow-card:      0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06);
        --radius:           8px;
    }

    /* ===== 全局 ===== */
    .stApp {
        background-color: var(--bg-page);
    }
    .stApp * {
        font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans SC", sans-serif;
    }

    /* ===== 侧边栏 ===== */
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar);
        border-right: 1px solid var(--border-light);
        padding-top: 0;
    }
    div[data-testid="stSidebarNav"]  { display: none; }
    div[data-testid="stSidebarHeader"] { display: none; }

    /* 侧边栏标题 */
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border-light);
    }

    /* ===== 统计卡片 ===== */
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }
    .stat-card {
        background: var(--bg-page);
        border: 1px solid var(--border-light);
        border-radius: var(--radius);
        padding: 14px 12px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .stat-card:hover {
        border-color: var(--accent);
        box-shadow: var(--shadow-card);
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: -0.02em;
    }
    .stat-label {
        font-size: 0.72rem;
        color: var(--text-hint);
        margin-top: 2px;
    }

    /* ===== 侧边栏按钮 ===== */
    [data-testid="stSidebar"] .stButton > button {
        font-size: 0.8rem;
        font-weight: 500;
        border-radius: 6px;
        border: 1px solid var(--border-normal);
        background: var(--bg-page);
        color: var(--text-primary);
        transition: all 0.15s ease;
        padding: 0.35rem 0.6rem;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
        background: var(--accent-light);
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: #fff;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: var(--accent-dark);
        border-color: var(--accent-dark);
    }

    /* ===== 小图标按钮 (删除/重新向量化) ===== */
    .st-key-del-btn .stButton > button,
    .st-key-revec-btn .stButton > button {
        font-size: 0.7rem;
        padding: 0.15rem 0.4rem;
        min-height: unset;
        line-height: 1;
        border-radius: 4px;
    }

    /* ===== 文件行 ===== */
    .file-row {
        padding: 6px 10px;
        margin: 2px 0;
        background: var(--bg-page);
        border: 1px solid var(--border-light);
        border-radius: 6px;
        font-size: 0.78rem;
        color: var(--text-primary);
    }
    .section-label {
        font-size: 0.72rem;
        color: var(--text-hint);
        margin: 0.8rem 0 0.3rem 0;
    }

    /* ===== 文件上传区 ===== */
    [data-testid="stFileUploader"] {
        border: 1px dashed var(--border-normal);
        border-radius: 6px;
        background: var(--bg-page);
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent);
    }

    /* ===== 对话 ===== */
    [data-testid="stChatMessage"] {
        border-radius: 6px !important;
        padding: 0.8rem 1rem !important;
        margin: 0.4rem 0 !important;
        font-size: 0.93rem;
        line-height: 1.7;
    }
    [data-testid="chat-message-user"] {
        background: #f5f7fa !important;
        border: 1px solid var(--border-light) !important;
        border-left: 3px solid var(--accent) !important;
    }
    [data-testid="chat-message-assistant"] {
        background: #ffffff !important;
        border: 1px solid var(--border-light) !important;
        border-left: 3px solid var(--gold) !important;
    }

    /* ===== 引用来源 ===== */
    .citation-block {
        margin-top: 0.6rem;
        padding: 8px 12px;
        background: var(--gold-light);
        border: 1px solid #f0dcb0;
        border-left: 3px solid var(--gold);
        border-radius: 4px;
        font-size: 0.75rem;
        color: #5a4a2a;
    }
    .citation-block .cite-header {
        color: var(--gold);
        font-size: 0.7rem;
        font-weight: 600;
        margin-bottom: 4px;
    }

    /* ===== 聊天输入 ===== */
    [data-testid="stChatInput"] textarea {
        font-size: 0.9rem !important;
        background: var(--bg-page) !important;
        border: 1px solid var(--border-normal) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--border-normal) !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-hint) !important;
    }

    /* ===== 分隔线 ===== */
    hr, .stDivider {
        border-color: var(--border-light) !important;
    }

    /* ===== Expander ===== */
    [data-testid="stExpander"] details {
        border: 1px solid var(--border-light);
        border-radius: 6px;
        background: var(--bg-page);
    }
    [data-testid="stExpander"] summary {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }

    /* ===== 滚动条 ===== */
    ::-webkit-scrollbar        { width: 5px; }
    ::-webkit-scrollbar-track  { background: transparent; }
    ::-webkit-scrollbar-thumb  { background: var(--border-normal); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-hint); }

    /* ===== 停止按钮 ===== */
    .stop-btn button {
        background: #d9304a !important;
        color: #fff !important;
        border: 1px solid #d9304a !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        border-radius: 6px !important;
        animation: pulse-stop 2s infinite !important;
    }
    .stop-btn button:hover {
        background: #c0283e !important;
        border-color: #c0283e !important;
    }
    @keyframes pulse-stop {
        0%, 100% { box-shadow: 0 0 0 0 rgba(217,48,74,0.4); }
        50%      { box-shadow: 0 0 0 6px rgba(217,48,74,0); }
    }

    /* ===== 运行状态居中 ===== */
    .run-status {
        text-align: center;
        margin-top: 4px;
        color: var(--text-secondary);
        font-size: 0.8rem;
    }

    /* ===== 状态列垂直居中 ===== */
    .status-align {
        display: flex;
        align-items: center;
        height: 100%;
    }

    /* ===== 示例查询按钮 ===== */
    .example-query .stButton > button {
        font-size: 0.8rem;
        color: var(--text-secondary);
        background: var(--bg-page);
        border: 1px solid var(--border-light);
        text-align: left;
        padding: 0.4rem 0.7rem;
        border-radius: 6px;
    }
    .example-query .stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
        background: var(--accent-light);
    }

    /* ===== Alert ===== */
    div[data-testid="stAlert"] {
        border-radius: 6px !important;
        font-size: 0.82rem;
    }

    /* ===== Spinner ===== */
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }

    /* ===== 主内容区顶部 margin 补偿 (header 被隐藏后) ===== */
    .block-container {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 初始化 Session State
# ============================================================
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 当前活跃的对话线程 ID
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None

if "vector_store" not in st.session_state:
    try:
        st.session_state["vector_store"] = VectorStoreService()
    except Exception:
        st.session_state["vector_store"] = None

# 侧边栏刷新计数器（删除/切换线程后触发刷新）
if "sidebar_tick" not in st.session_state:
    st.session_state["sidebar_tick"] = 0

# 后台运行状态（用于对话取消功能）
if "_run_active" not in st.session_state:
    st.session_state["_run_active"] = False
if "_run_chunks" not in st.session_state:
    st.session_state["_run_chunks"]: list[str] = []
if "_run_thread" not in st.session_state:
    st.session_state["_run_thread"]: threading.Thread | None = None


# ============================================================
# 辅助函数
# ============================================================
def get_kb_stats() -> dict:
    try:
        if st.session_state["vector_store"]:
            return st.session_state["vector_store"].get_collection_stats()
    except Exception:
        pass
    return {"total_chunks": 0, "collection_name": "N/A", "chunk_size": 0}


def get_loaded_files() -> list[str]:
    data_path = get_abs_path(chroma_conf.get("data_path", "data"))
    files = []
    if os.path.isdir(data_path):
        for f in sorted(os.listdir(data_path)):
            if f.endswith((".pdf", ".txt")):
                files.append(f)
    return files


def export_chat_history() -> str:
    lines = [
        "二级市场研究报告分析 · 对话记录\n",
        f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "=" * 50 + "\n\n",
    ]
    for msg in st.session_state["messages"]:
        role = "【用户】" if msg["role"] == "user" else "【分析报告】"
        lines.append(f"{role}\n\n{msg['content']}\n\n{'─' * 40}\n\n")
    return "".join(lines)


# ============================================================
# 后台执行辅助函数
# ============================================================
def _start_run(query: str):
    """把 Agent 执行放到后台线程，主线程通过 st.rerun() 轮询展示"""
    if not st.session_state["thread_id"]:
        st.session_state["thread_id"] = st.session_state["agent"].create_thread()

    st.session_state["messages"].append({"role": "user", "content": query})
    clear_citations()

    st.session_state["_run_chunks"] = []

    agent = st.session_state["agent"]
    thread_id = st.session_state["thread_id"]
    chunks_list: list[str] = st.session_state["_run_chunks"]

    def _agent_thread():
        try:
            for chunk in agent.execute_stream(query, thread_id=thread_id):
                chunks_list.append(chunk)
        except Exception as e:
            err = str(e) or type(e).__name__
            chunks_list.append(f"\n分析过程中出现错误（{err}）。请稍后重试或尝试其他查询。\n")

    thread = threading.Thread(target=_agent_thread, daemon=True)
    st.session_state["_run_thread"] = thread
    st.session_state["_run_active"] = True
    thread.start()
    st.rerun()


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    # --- 品牌 ---
    st.markdown(
        """
        <div style="
            font-size: 1.1rem; font-weight: 700; color: #1a1d23;
            margin-bottom: 0.15rem; letter-spacing: 0.03em;
        ">
            ■ ResearchAI
        </div>
        <div style="
            font-size: 0.72rem; color: #9ba0ab; margin-bottom: 0.8rem;
        ">
            二级市场研究报告分析平台
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # --- 知识库 ---
    st.markdown("#### 知识库")
    stats = get_kb_stats()

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">{stats["total_chunks"]}</div>
            <div class="stat-label">文档分片</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats["chunk_size"]}</div>
            <div class="stat-label">分片窗口</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- 文档列表 ---
    loaded_files = get_loaded_files()
    if loaded_files:
        st.markdown(f'<div class="section-label">已加载文档 · {len(loaded_files)} 个</div>', unsafe_allow_html=True)
        for f in loaded_files:
            st.markdown(f'<div class="file-row" title="{f}">{f}</div>', unsafe_allow_html=True)
    else:
        st.caption("暂无文档")

    st.divider()

    # --- 对话线程管理 ---
    st.markdown("#### 对话记录")

    # 新对话 + 清空
    c_new, c_clear = st.columns([3, 1])
    with c_new:
        if st.button("＋ 新对话", use_container_width=True, type="primary"):
            st.session_state["thread_id"] = None
            st.session_state["messages"] = []
            clear_citations()
            st.rerun()
    with c_clear:
        if threads := st.session_state["agent"].list_threads():
            if st.button("清空", use_container_width=True, help="删除全部历史对话"):
                st.session_state["agent"].delete_all_threads()
                st.session_state["thread_id"] = None
                st.session_state["messages"] = []
                st.session_state["sidebar_tick"] += 1
                st.rerun()

    # 获取线程列表
    threads = st.session_state["agent"].list_threads()

    if threads:
        st.markdown('<div class="section-label">历史对话</div>', unsafe_allow_html=True)
        for t in threads:
            tid = t["thread_id"]
            is_active = st.session_state["thread_id"] == tid
            title = t["title"]

            label = f"● {title}" if is_active else title
            if st.button(
                    label,
                    use_container_width=True,
                    key=f"thr_{tid}_{st.session_state['sidebar_tick']}",
                    help=f"创建: {t['created_at'][:16]}\n更新: {t['updated_at'][:16]}",
            ):
                st.session_state["thread_id"] = tid
                st.session_state["messages"] = st.session_state["agent"].load_messages(tid)
                clear_citations()
                st.rerun()
    else:
        st.caption("暂无历史对话")

    # 导出按钮（放在线程列表下方）
    if st.session_state["messages"]:
        txt = export_chat_history()
        st.download_button(
            "导出当前对话",
            data=txt,
            file_name=f"research_{st.session_state.get('thread_id', 'chat')}.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ============================================================
# 主内容区
# ============================================================
c_title, c_time = st.columns([5, 1])
with c_title:
    st.markdown(
        """
        <div style="font-size: 1.3rem; font-weight: 700; color: #1a1d23;">
            研究报告分析
        </div>
        <div style="font-size: 0.8rem; color: #9ba0ab; margin-bottom: 0.6rem;">
            ReAct Agent · RAG 检索增强 · 混合检索 · 引用溯源
        </div>
        """,
        unsafe_allow_html=True,
    )
with c_time:
    st.markdown(
        f'<div style="text-align:right; font-size:0.78rem; color:#9ba0ab; padding-top:0.3rem;">'
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")}'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# --- 示例查询 ---
st.markdown("#### 快速开始")
st.caption("选择一个示例，或直接在下方向输入框中输入您的研究问题")
st.markdown('<div class="example-query">', unsafe_allow_html=True)
examples = [
    "分析紫金矿业投资价值",
    "有色金属行业2025年展望",
    "贵州茅台近期走势如何",
    "生成阳光电源分析报告",
]
cols = st.columns(4)
for i, ex in enumerate(examples):
    with cols[i]:
        if st.button(ex, use_container_width=True, key=f"ex_{hash(ex)}"):
            st.session_state["_pending_query"] = ex
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# --- 渲染历史消息 ---
for i, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            src_html = '<div class="citation-block"><div class="cite-header">参考来源</div>'
            for s in message["sources"]:
                src_html += f'<div>· {s}</div>'
            src_html += '</div>'
            st.markdown(src_html, unsafe_allow_html=True)

# --- 处理快捷查询 ---
pending_query = st.session_state.pop("_pending_query", None)
if pending_query and not st.session_state["_run_active"]:
    _start_run(pending_query)

# --- 输入 ---
prompt = st.chat_input("请输入您的研究问题...")

if prompt and not st.session_state["_run_active"]:
    _start_run(prompt)

# --- 活跃运行中的展示 ---
if st.session_state["_run_active"]:
    stop_col, status_col = st.columns([1, 5])
    with stop_col:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        if st.button("■ 停止", key="stop_run", use_container_width=True):
            full = re.sub(
                r'\n?> 正在检索资料\.\.\.\n\n?', '',
                "".join(st.session_state["_run_chunks"]),
            ).strip()
            if full:
                citations = get_citations()
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": full + "\n\n*[已中止]*",
                    "sources": citations if citations else [],
                })
            st.session_state["_run_active"] = False
            st.session_state["_run_chunks"] = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with status_col:
        st.markdown('<div class="status-align">', unsafe_allow_html=True)
        tool_count = "".join(st.session_state["_run_chunks"]).count("> 正在检索资料...")
        if tool_count > 0:
            st.caption(f"已检索 {tool_count} 次，模型思考中...")
        else:
            st.caption("模型思考中...")
        st.markdown('</div>', unsafe_allow_html=True)

    # 实时展示已生成的内容
    current_text = "".join(st.session_state["_run_chunks"])
    if current_text.strip():
        st.chat_message("assistant").markdown(current_text)

    # 检查后台线程是否完成
    thread = st.session_state.get("_run_thread")
    if thread and not thread.is_alive():
        full = re.sub(
            r'\n?> 正在检索资料\.\.\.\n\n?', '',
            "".join(st.session_state["_run_chunks"]),
        ).strip()
        if full:
            citations = get_citations()
            st.session_state["messages"].append({
                "role": "assistant",
                "content": full,
                "sources": citations if citations else [],
            })
        st.session_state["_run_active"] = False
        st.session_state["_run_chunks"] = []
        st.rerun()
    else:
        # 仍在运行，短暂等待后通过 st.rerun() 刷新——实现实时交互
        time.sleep(0.5)
        st.rerun()