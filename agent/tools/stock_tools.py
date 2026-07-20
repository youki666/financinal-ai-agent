"""实时行情工具：AKShare 接入 A 股行情数据"""
from langchain_core.tools import tool
from utils.logger_handler import logger


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


def _safe_float(value, default=0.0):
    """安全转换为 float，失败返回默认值"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@tool(description="获取 A 股实时行情。入参 stock_code 为股票代码（如 601899、600519）或股票简称（如 紫金矿业、贵州茅台）。返回现价、涨跌幅、成交量、换手率、PE/PB 等数据。")
def stock_quote_realtime(stock_code: str) -> str:
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        code_str = str(stock_code).strip()

        # 按代码或名称匹配
        match = df[df["代码"].str.contains(code_str) | df["名称"].str.contains(code_str)]
        if match.empty:
            return f"未找到与 '{stock_code}' 匹配的股票，请检查代码或简称后重试。"

        row = match.iloc[0]
        return _format_quote(row)

    except Exception as e:
        logger.error(f"[StockQuote] 获取行情失败: {e}")
        return f"获取行情数据失败（{e}）。请稍后重试或检查股票代码是否正确。"


@tool(description="获取 A 股历史 K 线数据。入参 stock_code 为股票代码（如 601899），period 可选 'day'/'week'/'month'，默认 'month'。返回近期收盘价、涨跌幅、成交量等。")
def stock_history(stock_code: str, period: str = "month") -> str:
    try:
        import akshare as ak

        period_map = {"day": "daily", "week": "weekly", "month": "monthly"}
        ak_period = period_map.get(period, "monthly")

        df = ak.stock_zh_a_hist(symbol=str(stock_code).strip(), period=ak_period, adjust="qfq")
        if df.empty:
            return f"未找到股票 {stock_code} 的历史数据。"

        recent = df.tail(10).iloc[::-1]
        lines = [f"## {stock_code} 近期历史行情（{period}）", "", "| 日期 | 收盘 | 涨跌幅 | 成交量 |", "|------|------|--------|--------|"]
        for _, row in recent.iterrows():
            date = str(row.get("日期", ""))[:10]
            close = row.get("收盘")
            pct = f"{_safe_float(row.get('涨跌幅', 0)):.2f}%"
            vol = row.get("成交量")
            lines.append(f"| {date} | {close} | {pct} | {vol} |")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[StockHistory] 获取历史数据失败: {e}")
        return f"获取历史数据失败（{e}）。请稍后重试。"
