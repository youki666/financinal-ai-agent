"""Tavily Search 工具：联网搜索获取实时信息"""
import os
from langchain_core.tools import tool
from utils.logger_handler import logger


@tool(description="搜索互联网获取实时公开信息。入参 query 为搜索关键词（如 2026年光伏新政 2026年7月LPR利率），返回标题、URL 和内容摘要。适用于知识库无法覆盖的最新资讯和政策。")
def web_search(query: str) -> str:
    try:
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return "联网搜索功能需要配置 TAVILY_API_KEY（在 .env 文件中设置），请访问 https://tavily.com 注册获取免费 API Key。"

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=10,
            include_answer=True,
            include_raw_content=False,
        )

        lines = [f"## '{query}' 搜索结果", ""]

        # Tavily 生成的综合摘要
        answer = response.get("answer", "")
        if answer:
            lines.append(f"**摘要**：{answer}")
            lines.append("")

        results = response.get("results", [])
        if not results:
            return f"未找到与 '{query}' 相关的搜索结果。"

        lines.append(f"**共 {len(results)} 条结果**：")
        lines.append("")
        for i, r in enumerate(results[:8], 1):
            title = r.get("title", "无标题")
            url = r.get("url", "")
            content = (r.get("content", "") or "")[:150]
            lines.append(f"**{i}. {title}**")
            if content:
                lines.append(f"   {content}")
            if url:
                lines.append(f"   [来源]({url})")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[WebSearch] 搜索失败: {e}")
        return f"联网搜索失败（{e}）。请稍后重试。"
