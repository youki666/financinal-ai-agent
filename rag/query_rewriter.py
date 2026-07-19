"""查询重写模块：将用户口语化查询改写为精准检索词，支持多角度检索"""
from functools import lru_cache
from model.factory import chat_model
from utils.prompt_loader import load_query_rewrite_prompts
from utils.logger_handler import logger


class QueryRewriter:

    def __init__(self):
        self.prompt_template = load_query_rewrite_prompts()
        self.model = chat_model

    @lru_cache(maxsize=128)
    def _cached_rewrite(self, query: str) -> str:
        """缓存热点查询的改写结果"""
        return self._do_rewrite(query)

    def _do_rewrite(self, query: str) -> str:
        try:
            prompt = self.prompt_template.format(query=query)
            response = self.model.invoke(prompt)
            return response.content.strip() if hasattr(response, "content") else str(response).strip()
        except Exception as e:
            logger.warning(f"[QueryRewriter] 改写失败，使用原始查询: {e}")
            return query

    def rewrite(self, query: str, use_cache: bool = True) -> list[str]:
        """将原始查询改写为 1-3 个精准检索词"""
        if use_cache:
            raw = self._cached_rewrite(query)
        else:
            raw = self._do_rewrite(query)

        queries = [q.strip() for q in raw.split("\n") if q.strip()]
        queries = [q for q in queries if q][:3]

        if not queries:
            queries = [query]

        logger.info(f"[QueryRewriter] '{query}' → {queries}")
        return queries


query_rewriter = QueryRewriter()
