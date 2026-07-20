"""重排序模块：基于 LLM 对检索结果进行相关性重排序和去重"""
from langchain_core.documents import Document
from utils.logger_handler import logger


class LLMReranker:
    """使用 LLM 作为重排序器，评估文档与查询的相关性"""

    def __init__(self, min_score: float = 0.3):
        self.min_score = min_score

    def _deduplicate(self, documents: list[Document], threshold: float = 0.7) -> list[Document]:
        """基于内容相似度去重（Jaccard相似度）"""
        if len(documents) <= 1:
            return documents

        def jaccard(text1: str, text2: str) -> float:
            set1 = set(text1)
            set2 = set(text2)
            if not set1 or not set2:
                return 0.0
            return len(set1 & set2) / len(set1 | set2)

        kept = [documents[0]]
        for doc in documents[1:]:
            is_dup = False
            for k in kept:
                if jaccard(doc.page_content[:200], k.page_content[:200]) > threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(doc)
        return kept

    def rerank(self, query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
        """重排序主入口：去重 + 按启发式规则排序 + min_score 过滤"""
        if not documents:
            return []

        if top_k is None:
            top_k = len(documents)

        # 去重
        deduped = self._deduplicate(documents)
        if len(deduped) < len(documents):
            logger.info(f"[Reranker] 去重: {len(documents)} → {len(deduped)}")

        # 启发式重排序：标题匹配 > 关键词密度 > 文档长度
        def score(doc: Document) -> float:
            s = 0.0
            content = doc.page_content
            query_lower = query.lower()

            # 标题/首句匹配加分
            first_line = content.split("\n")[0].lower()
            if any(term in first_line for term in query_lower.split()):
                s += 2.0

            # 关键词出现频率
            term_count = sum(1 for term in query_lower.split() if term in content.lower())
            s += term_count * 0.5

            # 适中长度加分（100-500字符最佳）
            length = len(content)
            if 100 <= length <= 500:
                s += 1.0
            elif length < 20:
                s -= 1.0

            return s

        scored = [(score(doc), doc) for doc in deduped]

        # 过滤低分文档（得分 ≤ 0 说明与查询无关）
        filtered = [(s, doc) for s, doc in scored if s > self.min_score]
        if len(filtered) < len(scored):
            logger.info(f"[Reranker] min_score 过滤 ({self.min_score}): {len(scored)} → {len(filtered)}")

        # 如果全被过滤，保留最高分的那一篇（至少让 LLM 有东西判断）
        if not filtered:
            filtered = scored[:1]

        filtered.sort(key=lambda x: x[0], reverse=True)
        result = [doc for _, doc in filtered[:top_k]]
        logger.info(f"[Reranker] 重排序完成，返回 top-{len(result)}")
        return result


reranker = LLMReranker()
