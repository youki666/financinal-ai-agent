"""上下文组装模块：将检索到的文档片段组装为结构化上下文"""
from langchain_core.documents import Document
from collections import defaultdict
from utils.logger_handler import logger


class ContextAssembler:
    """按来源分组、Token预算控制、优先级排序"""

    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
        self._chars_per_token = 2.0  # 中文每 token 约 2 字符

    @property
    def max_chars(self) -> int:
        return int(self.max_tokens * self._chars_per_token)

    def _group_by_source(self, documents: list[Document]) -> dict[str, list[Document]]:
        groups: dict[str, list[Document]] = defaultdict(list)
        for doc in documents:
            source = doc.metadata.get("source", "未知来源")
            groups[source].append(doc)
        return dict(groups)

    def _sort_groups(self, groups: dict[str, list[Document]], query: str) -> list[tuple[str, list[Document]]]:
        """按与查询的相关性排序来源组"""
        def group_score(item):
            source, docs = item[0], item[1]
            # 标题匹配
            if any(term in source.lower() for term in query.lower().split()):
                return 3
            # 文档内容匹配度
            total_score = sum(
                sum(1 for term in query.lower().split() if term in doc.page_content.lower())
                for doc in docs
            )
            return total_score

        return sorted(groups.items(), key=group_score, reverse=True)

    def assemble(self, query: str, documents: list[Document]) -> str:
        """组装上下文"""
        if not documents:
            return ""

        groups = self._group_by_source(documents)
        sorted_groups = self._sort_groups(groups, query)

        context_parts: list[str] = []
        total_chars = 0
        counter = 0

        for source, docs in sorted_groups:
            source_text = ""
            for doc in docs:
                chunk_text = doc.page_content.replace("\n", " ").strip()
                if not chunk_text:
                    continue

                remaining = self.max_chars - total_chars
                if remaining <= 0:
                    break

                if len(chunk_text) > remaining:
                    chunk_text = chunk_text[:remaining] + "..."

                source_text += chunk_text + "\n"
                total_chars += len(chunk_text)

            if source_text.strip():
                counter += 1
                context_parts.append(f"--- 文档{counter}：{source} ---\n{source_text}")

        context = "\n".join(context_parts)
        logger.info(f"[ContextAssembler] 组装完成: {len(documents)} 文档 → {counter} 组, {total_chars} 字符")
        return context


context_assembler = ContextAssembler()
