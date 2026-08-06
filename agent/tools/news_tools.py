"""新闻检索工具：NewsAPI 财经新闻 + AKShare 东方财富快讯（双源自动切换）"""
import os
from datetime import datetime, timedelta
from langchain_core.tools import tool
from utils.logger_handler import logger


def _search_news_akshare(query: str, days: int) -> str:
    """AKShare 东方财富全球财经新闻搜索（NewsAPI 不可用时的兜底方案）"""
    import akshare as ak

    df = ak.stock_info_global_em()
    if df.empty:
        return "暂未获取到财经新闻。"

    cutoff = datetime.now() - timedelta(days=days)
    lines = [f"## '{query}' 近期新闻（最近 {days} 天，东方财富）", ""]
    count = 0
    for _, row in df.iterrows():
        title = str(row.get("标题", ""))
        summary = str(row.get("摘要", ""))[:150]
        pub_time_str = str(row.get("发布时间", ""))[:19]
        url = str(row.get("链接", ""))

        # 按关键词模糊匹配
        text = title + summary
        if not any(kw in text for kw in query.split()):
            continue

        # 时间过滤
        try:
            pub_time = datetime.strptime(pub_time_str, "%Y-%m-%d %H:%M:%S")
            if pub_time < cutoff:
                continue
        except ValueError:
            pass

        count += 1
        lines.append(f"**{count}. {title}**")
        if summary:
            lines.append(f"   {summary}")
        if url:
            lines.append(f"   [来源]({url})")
        lines.append("")

        if count >= 8:
            break

    if count == 0:
        return f"最近 {days} 天内未找到与 '{query}' 相关的新闻。"
    return "\n".join(lines)


@tool(description="检索近期财经新闻。入参 query 为检索关键词（如 光伏、新能源汽车），days 为最近天数（默认 7）。返回标题、来源、发布时间和摘要。自动在 NewsAPI 和东方财富之间切换。")
def financial_news(query: str, days: int = 7) -> str:
    # 优先尝试 NewsAPI
    api_key = os.getenv("NEWSAPI_KEY", "")
    if api_key:
        try:
            import requests

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

            if data.get("status") == "ok":
                articles = data.get("articles", [])
                if articles:
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
                else:
                    return f"最近 {days} 天内未找到与 '{query}' 相关的新闻。"
        except Exception as e:
            logger.warning(f"[FinancialNews] NewsAPI 不可用，切换到东方财富: {e}")

    # 兜底：东方财富全球财经新闻
    try:
        return _search_news_akshare(query, days)
    except Exception as e:
        logger.error(f"[FinancialNews] 东方财富检索也失败: {e}")
        return f"新闻检索失败（{e}）。NewsAPI 和东方财富均不可用，请稍后重试。"


@tool(description="获取东方财富最新市场热点新闻。入参 limit 为返回条数（默认 20）。返回最新市场快讯列表。")
def flash_news(limit: int = 20) -> str:
    try:
        import akshare as ak

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
