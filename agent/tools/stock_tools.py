"""实时行情工具：新浪财经数据源（AKShare 封装）"""
import pandas as pd
from langchain_core.tools import tool
from utils.logger_handler import logger


def _to_sina_code(stock_code: str) -> str:
    """将纯数字代码转为新浪格式（sh/sz 前缀）"""
    code = str(stock_code).strip()
    if code.startswith(("sh", "sz", "bj")):
        return code
    if code.startswith(("6", "68")):
        return f"sh{code}"
    if code.startswith(("0", "3", "2")):
        return f"sz{code}"
    return f"sh{code}"


def _format_quote(row) -> str:
    """将行情数据格式化为 Markdown 表格"""
    name = row.get("名称", "N/A")
    code = row.get("代码", "N/A")

    fields = {
        "最新价": row.get("最新价"),
        "涨跌幅": f"{_safe_float(row.get('涨跌幅', 0)):.2f}%",
        "涨跌额": row.get("涨跌额"),
        "今开": row.get("今开"),
        "昨收": row.get("昨收"),
        "最高": row.get("最高"),
        "最低": row.get("最低"),
        "成交量": row.get("成交量"),
        "成交额": row.get("成交额"),
    }
    lines = [
        f"## {name}（{code}）实时行情",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
    ]
    for label, value in fields.items():
        if value is not None:
            val = f"{float(value):.2f}" if isinstance(value, (int, float)) else str(value)
            lines.append(f"| {label} | {val} |")
    return "\n".join(lines)


def _format_history(df, stock_code: str, period: str) -> str:
    """将历史数据格式化为 Markdown 表格"""
    period_label = {"day": "日线", "week": "周线", "month": "月线"}.get(period, "日线")
    lines = [f"## {stock_code} 近期历史行情（{period_label}）", "", "| 日期 | 开盘 | 收盘 | 最高 | 最低 | 涨跌幅 | 成交量 |", "|------|------|------|------|------|--------|--------|"]
    for _, row in df.iterrows():
        date = str(row.get("date", ""))[:10]
        open_v = row.get("open")
        close = row.get("close")
        high = row.get("high")
        low = row.get("low")
        pct = _calc_pct(row)
        vol = row.get("volume")
        lines.append(f"| {date} | {open_v} | {close} | {high} | {low} | {pct} | {vol} |")
    return "\n".join(lines)


def _calc_pct(row) -> str:
    """根据 open/close 计算涨跌幅"""
    try:
        o, c = float(row.get("open", 0)), float(row.get("close", 0))
        if o == 0:
            return "0.00%"
        return f"{((c - o) / o * 100):.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@tool(description="获取 A 股实时行情。入参 stock_code 为股票代码（如 601899、600519）或股票简称（如 紫金矿业、贵州茅台）。返回现价、涨跌幅、成交量等数据。")
def stock_quote_realtime(stock_code: str) -> str:
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot()
        code_str = str(stock_code).strip()

        # 去掉前缀再匹配（Sina 用 sh601899 格式）
        clean = code_str
        for prefix in ("sh", "sz", "bj"):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break

        match = df[df["代码"].str.contains(clean) | df["名称"].str.contains(code_str)]
        if match.empty:
            return f"未找到与 '{stock_code}' 匹配的股票，请检查代码或简称后重试。"

        row = match.iloc[0]
        return _format_quote(row)

    except Exception as e:
        logger.error(f"[StockQuote] 获取行情失败: {e}")
        return f"获取行情数据失败（{e}）。请稍后重试或检查股票代码是否正确。"


@tool(description="获取 A 股历史 K 线数据。入参 stock_code 为股票代码（如 601899），period 可选 'day'/'week'/'month'，默认 'month'。返回近期开盘价、收盘价、最高/最低、涨跌幅、成交量等。")
def stock_history(stock_code: str, period: str = "month") -> str:
    try:
        import akshare as ak

        sina_code = _to_sina_code(str(stock_code).strip())

        if period == "month":
            df = ak.stock_zh_a_daily(symbol=sina_code, adjust="qfq")
            # 按月采样：每月最后一个交易日
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df["ym"] = df["date"].dt.strftime("%Y-%m")
            df = df.groupby("ym").last().reset_index()
            df = df.tail(10)
        elif period == "week":
            df = ak.stock_zh_a_daily(symbol=sina_code, adjust="qfq")
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df = df.set_index("date").resample("W").last().dropna().reset_index()
            df = df.tail(10)
        else:
            df = ak.stock_zh_a_daily(symbol=sina_code, adjust="qfq")
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(10)

        if df.empty:
            return f"未找到股票 {stock_code} 的历史数据。"

        return _format_history(df.iloc[::-1], stock_code, period)

    except Exception as e:
        logger.error(f"[StockHistory] 获取历史数据失败: {e}")
        return f"获取历史数据失败（{e}）。请稍后重试。"
