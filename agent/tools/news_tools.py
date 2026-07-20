"""新闻检索工具：NewsAPI 财经新闻 + AKShare 财联社快讯"""
import os
from datetime import datetime, timedelta
from langchain_core.tools import tool
from utils.logger_handler import logger


@tool(description="检索近期财经新闻。入参 query 为检索关键词（如 光伏、新能源汽车），days 为最近天数（默认 7）。返回标题、来源、发布时间和摘要。")
def financial_news(query: str, days: int = 7) -> str:
    try:
        import requests

        api_key = os.getenv("NEWSAPI_KEY", "")
        if not api_key:
            return "新闻检索功能需要配置 NEWSAPI_KEY（在 .env 文件中设置），请访问 https://newsapi.org/ 注册获取。"

        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "language": "zh",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "apiKey": api_key,
        }

        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if data.get("status") != "ok":
            return f"新闻检索失败: {data.get('message', '未知错误')}"

        articles = data.get("articles", [])
        if not articles:
            return f"最近 {days} 天内未找到与 '{query}' 相关的新闻。"

        lines = [f"## '{query}' 近期新闻（最近 {days} 天）", ""]
        for i, art in enumerate(articles[:8], 1):
            title = art.get("title", "无标题")
            source = art.get("source", {}).get("name", "未知来源")
            published = (art.get("publishedAt", "") or "")[:10]
            desc = (art.get("description", "") or "")[:120]
            lines.append(f"**{i}. {title}**")
            lines.append(f"   来源: {source} | {published}")
            if desc:
                lines.append(f"   {desc}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[FinancialNews] 检索失败: {e}")
        return f"新闻检索失败（{e}）。请稍后重试。"


@tool(description="获取东方财富最新市场热点新闻。入参 limit 为返回条数（默认 20）。返回最新市场快讯列表。")
def flash_news(limit: int = 20) -> str:
    try:
        import akshare as ak

        # 使用东方财富市场热点新闻
        df = ak.stock_news_em(symbol="000001")
        if df.empty:
            return "暂未获取到市场快讯。"

        recent = df.head(min(limit, len(df)))
        lines = [f"## 市场最新新闻（最新 {len(recent)} 条）", ""]
        for i, (_, row) in enumerate(recent.iterrows(), 1):
            title = str(row.get("新闻标题", ""))
            ctime = str(row.get("发布时间", ""))[:19]
            source = str(row.get("文章来源", ""))
            lines.append(f"**{i}.** {title}")
            lines.append(f"    _{source} | {ctime}_")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[FlashNews] 获取快讯失败: {e}")
        return f"获取实时快讯失败（{e}）。请稍后重试。"
