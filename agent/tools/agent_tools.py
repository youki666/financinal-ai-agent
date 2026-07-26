"""金融研究 Agent 工具集"""
from langchain_core.tools import tool

from utils.logger_handler import logger

from rag.rag_service import RagSummarizeService

rag = RagSummarizeService()


@tool(description="从研究报告知识库中检索专业资料。入参 query 为检索词字符串，返回相关研究报告摘要。")
def rag_summarize(query: str) -> str:
    return rag.summarize(query)




def _format_quote(series) -> str:
    """将 AKShare 返回的行情 Series 格式化为 Markdown"""
    fields = {
        "最新价": series.get("最新价"),
        "涨跌幅": f"{_safe_float(series.get('涨跌幅', 0)):.2f}%",
        "涨跌额": series.get("涨跌额"),
        "成交量": series.get("成交量"),
        "成交额": series.get("成交额"),
        "换手率": f"{_safe_float(series.get('换手率', 0)):.2f}%",
        "今开": series.get("今开"),
        "昨收": series.get("昨收"),
        "最高": series.get("最高"),
        "最低": series.get("最低"),
        "市盈率": series.get("市盈率-动态"),
        "市净率": series.get("市净率"),
    }
    lines = [
        f"## {series.get('名称', 'N/A')}（{series.get('代码', 'N/A')}）实时行情",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
    ]
    for label, value in fields.items():
        if value is not None:
            lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


import requests

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def _safe_float(value, default=0.0):
    """安全转换为 float，失败返回默认值"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_quote_sina(parts: list[str], code: str) -> str:
    """将新浪实时行情数据格式化为 Markdown
    parts 字段: 0=名称, 1=今开, 2=昨收, 3=现价, 4=最高, 5=最低,
                8=成交量(股), 9=成交额(元), 30=日期, 31=时间
    """
    name = parts[0]
    price = _safe_float(parts[3])
    prev_close = _safe_float(parts[2])
    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
    vol = _safe_float(parts[8])
    amount = _safe_float(parts[9])
    date = parts[30] if len(parts) > 30 else ""
    time = parts[31] if len(parts) > 31 else ""

    lines = [
        f"## {name}（{code}）实时行情",
        f"更新时间：{date} {time}",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 最新价 | {parts[3]} |",
        f"| 涨跌幅 | {change_pct:.2f}% |",
        f"| 今开 | {parts[1]} |",
        f"| 昨收 | {parts[2]} |",
        f"| 最高 | {parts[4]} |",
        f"| 最低 | {parts[5]} |",
        f"| 成交量(股) | {vol:,.0f} |",
        f"| 成交额(元) | {amount:,.0f} |",
    ]
    return "\n".join(lines)


@tool(description="获取 A 股实时行情。入参 stock_code 为股票代码（如 601899、600519）,当用户输入股票简称时你需要转换为对应的股票代码（如 紫金矿业、贵州茅台）。返回现价、涨跌幅、成交量等数据。")
def stock_quote_realtime(stock_code: str) -> str:
    try:
        code = str(stock_code).strip()
        symbol = f"sh{code}" if code.startswith("6") else f"sz{code}"

        headers = {**_BROWSER_HEADERS, "Referer": "https://finance.sina.com.cn/"}
        r = requests.get(
            f"https://hq.sinajs.cn/list={symbol}",
            headers=headers, timeout=10,
        )
        text = r.text
        if "=" not in text or len(text) < 30:
            return f"未找到与 '{stock_code}' 匹配的股票，请检查代码后重试。"

        # 提取引号内的数据
        data_str = text.split('"')[1] if '"' in text else ""
        if not data_str:
            return f"未获取到 {code} 的行情数据。"

        parts = data_str.split(",")
        if len(parts) < 6:
            return f"未获取到 {code} 的行情数据。"

        return _format_quote_sina(parts, code)

    except Exception as e:
        logger.error(f"[StockQuote] 获取行情失败: {e}")
        return "获取行情数据失败，请稍后重试。"


@tool(description="获取 A 股历史 K 线数据。入参 stock_code 为股票代码（如 601899），当用户输入股票简称时你需要转换为对应的股票代码（如 紫金矿业对应 601899），period 可选 'day'/'week'/'month'，默认 'month'。返回近期开盘价、收盘价、最高、最低、涨跌幅、成交量等。")
def stock_history(stock_code: str, period: str = "month") -> str:
    try:
        code = str(stock_code).strip()

        # 根据代码判断交易所前缀
        if code.startswith("6"):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"

        scale_map = {"day": "240", "week": "1200", "month": "7200"}
        scale = scale_map.get(period, "7200")

        url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
        params = {"symbol": symbol, "scale": scale, "ma": "no", "datalen": "10"}
        r = requests.get(url, params=params, headers=_BROWSER_HEADERS, timeout=10)
        klines = r.json()

        if not klines:
            return f"未找到股票 {code} 的历史数据。"

        period_label = {"day": "日", "week": "周", "month": "月"}.get(period, "月")

        lines = [
            f"## {code} 近期历史行情（{period_label}K）",
            "",
            "| 日期 | 开盘 | 收盘 | 最高 | 最低 | 涨跌幅 | 成交量 |",
            "|------|------|------|------|------|--------|--------|",
        ]
        for row in klines[-10:]:
            date = row.get("day", "")
            open_p = float(row.get("open", 0))
            close_p = float(row.get("close", 0))
            high = row.get("high", "")
            low = row.get("low", "")
            vol = row.get("volume", "")
            pct = ((close_p - open_p) / open_p * 100) if open_p else 0
            lines.append(f"| {date} | {row.get('open', '')} | {row.get('close', '')} | {high} | {low} | {pct:.2f}% | {vol} |")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[StockHistory] 获取历史数据失败: {e}")
        return "获取历史数据失败，请稍后重试。"

