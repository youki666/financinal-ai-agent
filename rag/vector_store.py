import os
import numpy as np

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from model.factory import get_embed_model

from utils.config_handler import chroma_conf
from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class BM25Retriever:
    """BM25 关键词检索器，作为向量检索的补充"""

    def __init__(self):
        self.bm25_index: BM25Okapi | None = None
        self.documents: list[Document] = []
        self._dirty = True

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def build_index(self, documents: list[Document]):
        if not documents:
            return
        self.documents = documents
        tokenized = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25_index = BM25Okapi(tokenized)
        self._dirty = False
        logger.info(f"[BM25] 索引构建完成，共 {len(documents)} 篇文档")

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        if self._dirty or self.bm25_index is None:
            logger.warning("[BM25] 索引未就绪，返回空结果")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.documents[i], scores[i]) for i in top_indices if scores[i] > 0]


def rrf_fusion(
        vector_results: list[Document],
        bm25_results: list[tuple[Document, float]],
        k: int = 3,
        rrf_k: int = 60,
) -> list[Document]:
    """RRF (Reciprocal Rank Fusion) 混合检索融合算法"""
    scores: dict[str, float] = {}

    for rank, doc in enumerate(vector_results):
        doc_id = doc.page_content[:100]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)

    for rank, (doc, _) in enumerate(bm25_results):
        doc_id = doc.page_content[:100]
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)

    all_docs: dict[str, Document] = {}
    for doc in vector_results:
        all_docs[doc.page_content[:100]] = doc
    for doc, _ in bm25_results:
        all_docs[doc.page_content[:100]] = doc

    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [all_docs[doc_id] for doc_id, _ in sorted_ids[:k]]


class VectorStoreService:

    def __init__(self):
        self.vectors = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=get_embed_model(),
            persist_directory=chroma_conf["persist_directory"],
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

        self.bm25_retriever = BM25Retriever()
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self):
        """从 ChromaDB 中读取全部文档重建 BM25 索引"""
        try:
            results = self.vectors.get()
            if results and results["documents"]:
                documents = [
                    Document(page_content=text, metadata=meta or {})
                    for text, meta in zip(results["documents"], results["metadatas"])
                ]
                self.bm25_retriever.build_index(documents)
        except Exception as e:
            logger.warning(f"[BM25] 索引重建失败: {e}")

    def get_retriever(self):
        return self.vectors.as_retriever(search_kwargs={"k": chroma_conf.get("k", 5)})

    def hybrid_search(self, query: str, k: int | None = None) -> list[Document]:
        """混合检索：向量检索 + BM25 关键词检索 → RRF 融合"""
        if k is None:
            k = chroma_conf.get("k", 3)

        # 向量检索：取 top-2k 候选
        retriever = self.vectors.as_retriever(search_kwargs={"k": k * 2})
        vector_results = retriever.invoke(query)

        # BM25 检索：取 top-2k 候选
        bm25_results = self.bm25_retriever.search(query, k=k * 2)

        if not bm25_results:
            logger.info("[HybridSearch] BM25 无结果，回退到纯向量检索")
            return vector_results[:k]

        fused = rrf_fusion(vector_results, bm25_results, k=k)

        logger.info(
            f"[HybridSearch] 向量检索 {len(vector_results)} 篇, "
            f"BM25 检索 {len(bm25_results)} 篇, "
            f"融合后 {len(fused)} 篇"
        )
        return fused

    def get_collection_stats(self) -> dict:
        """获取知识库统计信息"""
        try:
            results = self.vectors.get()
            doc_count = len(results["ids"]) if results and results["ids"] else 0
            return {
                "total_chunks": doc_count,
                "collection_name": chroma_conf.get("collection_name", "agent"),
                "persist_directory": chroma_conf.get("persist_directory", "chroma_db"),
                "chunk_size": chroma_conf.get("chunk_size", 200),
            }
        except Exception as e:
            logger.error(f"[Stats] 获取统计信息失败: {e}")
            return {"error": str(e)}

    def load_document(self):
        """从数据文件夹内读取数据文件，转为向量存入向量库（MD5去重）"""

        def check_md5_hex(md5_for_check: str) -> bool:
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str) -> list[Document]:
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        new_docs_loaded = False

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                continue

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库] {path} 内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库] {path} 内没有有效文本内容，跳过")
                    continue

                # 注入来源文件名到元数据
                file_name = os.path.basename(path)
                for doc in documents:
                    if not doc.metadata:
                        doc.metadata = {}
                    doc.metadata["source"] = file_name

                split_document: list[Document] = self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库] {path} 分片后没有有效文本内容，跳过")
                    continue

                self.vectors.add_documents(split_document)
                save_md5_hex(md5_hex)
                new_docs_loaded = True
                logger.info(f"[加载知识库] {path} 内容加载成功，共 {len(split_document)} 个分片")
            except Exception as e:
                logger.error(f"[加载知识库] {path} 加载失败：{str(e)}", exc_info=True)
                continue

        if new_docs_loaded:
            self._rebuild_bm25_index()

    def rebuild_index(self):
        """强制重建 BM25 索引（用于前端手动刷新）"""
        self._rebuild_bm25_index()
        return self.get_collection_stats()


if __name__ == '__main__':
    vs = VectorStoreService()

    # 先加载文档
    vs.load_document()

    # 测试混合检索
    print("=" * 50)
    print("混合检索测试：紫金矿业")
    print("=" * 50)
    results = vs.hybrid_search("紫金矿业", k=3)
    for i, doc in enumerate(results):
        print(f"[{i+1}] {doc.page_content[:200]}...")
        print(f"    来源: {doc.metadata.get('source', 'unknown')}")
        print("-" * 40)

    print("\n知识库统计:")
    print(vs.get_collection_stats())
