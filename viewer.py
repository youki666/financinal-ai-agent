import streamlit as st
import chromadb

st.set_page_config(page_title="ChromaDB 浏览", layout="wide")
st.title("📊 本地 ChromaDB 数据查看器")

# ---------- 1. 连接 ChromaDB ----------
@st.cache_resource
def get_client():
    try:
        # 优先使用 PersistentClient（无需额外服务）
        return chromadb.PersistentClient(path="./chroma_db")
    except Exception as e:
        st.error(f"连接失败：{e}")
        st.stop()

client = get_client()

# ---------- 2. 选择集合 ----------
collections = client.list_collections()
if not collections:
    st.warning("数据库中没有集合，请先插入数据。")
    st.stop()

selected = st.selectbox("选择集合", [c.name for c in collections])
coll = client.get_collection(selected)

# ---------- 3. 拉取数据（带分页） ----------
st.sidebar.markdown("### 分页控制")
page_size = st.sidebar.slider("每页条数", 10, 500, 100)
offset = st.sidebar.number_input("偏移量（从第几条开始）", min_value=0, step=10, value=0)

try:
    # 先获取总数
    count = coll.count()
    st.write(f"📦 集合 `{selected}` 共有 **{count}** 条记录")

    # 按偏移和限制拉取（需要 ChromaDB >= 1.5.0）
    results = coll.get(
        limit=page_size,
        offset=offset,
        include=["documents", "metadatas"]
    )
except Exception as e:
    st.error(f"获取数据失败：{e}")
    st.stop()

# ---------- 4. 展示数据 ----------
doc_ids = results.get("ids", [])
documents = results.get("documents", []) or []
metadatas = results.get("metadatas", []) or []

if not doc_ids:
    st.info(f"第 {offset+1} ~ {offset+page_size} 条无数据，请调整偏移量。")
else:
    st.success(f"显示第 {offset+1} ~ {offset+len(doc_ids)} 条")

    for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, documents, metadatas)):
        with st.expander(f"📄 #{offset+idx+1}  {doc_id[:20]}"):
            # 文档内容（限制长度避免页面卡顿）
            display_text = doc if doc else "(空文档)"
            if len(display_text) > 1000:
                display_text = display_text[:1000] + "... (截断)"
            st.text(display_text)

            if meta:
                st.json(meta)

# ---------- 5. 导出/高级功能（可选） ----------
if st.sidebar.button("📥 导出当前页为 JSON"):
    import json
    data = {
        "ids": doc_ids,
        "documents": documents,
        "metadatas": metadatas
    }
    st.sidebar.download_button(
        label="下载 JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name=f"{selected}_data.json",
        mime="application/json"
    )