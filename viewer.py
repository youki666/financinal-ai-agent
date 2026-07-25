import streamlit as st
import chromadb

st.set_page_config(page_title="ChromaDB 浏览", layout="wide")
st.title("📊 本地 ChromaDB 数据查看器")

# 连接你的 ChromaDB 服务
client = chromadb.HttpClient(host="127.0.0.1", port=8000)

collections = client.list_collections()
if not collections:
    st.warning("数据库中没有集合，请先插入数据。")
else:
    selected = st.selectbox("选择集合", [c.name for c in collections])
    coll = client.get_collection(selected)

    # 获取最多 100 条记录
    results = coll.get(include=["documents", "metadatas"])
    if results["ids"]:
        st.write(f"共 {len(results['ids'])} 条记录（显示前 100 条）")
        for idx, (doc_id, doc, meta) in enumerate(zip(results["ids"], results["documents"], results["metadatas"])):
            with st.expander(f"📄 {doc_id}"):
                st.text(doc if doc else "(空文档)")
                if meta:
                    st.json(meta)
    else:
        st.info("该集合为空。")