"""RAG 总结服务：查询重写 → 混合检索 → 重排序 → 上下文组装 → LLM 总结"""
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

    def summarize(self, query: str) -> str:
        """完整的 RAG 总结流程"""
        logger.info(f"[RAG] 开始处理: '{query}'")

        context_docs = self.retrieve_docs(query)
        logger.info(f"[RAG] 检索到 {len(context_docs)} 篇相关文档")

        if not context_docs:
            return (
                "当前知识库中暂未检索到与您问题直接相关的资料。"
                "建议：1) 尝试使用更具体的检索词 2) 上传相关研究报告 PDF 到知识库"
            )

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
