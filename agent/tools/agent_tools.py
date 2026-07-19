"""金融研究 Agent 工具集"""
from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService

rag = RagSummarizeService()


@tool(description="从研究报告知识库中检索专业资料。入参 query 为检索词字符串，返回相关研究报告摘要。适用于需要补充专业信息、行业数据、公司基本面等场景。")
def rag_summarize(query: str) -> str:
    return rag.summarize(query)


@tool(description="检索指定股票的概况信息，包括主营业务、近期动态、关键财务数据。入参 stock_code 为股票代码或简称。")
def stock_brief(stock_code: str) -> str:
    query = f"{stock_code} 主营业务 财务数据 近期动态 公司概况"
    return rag.summarize(query)


@tool(description="检索指定行业的整体概况，包括产业链结构、竞争格局、政策动向、发展趋势。入参 industry 为行业名称。")
def industry_overview(industry: str) -> str:
    query = f"{industry} 产业链 竞争格局 政策 发展趋势 行业分析"
    return rag.summarize(query)


@tool(description="基于已检索的研究资料，生成结构化 Markdown 研究报告。入参 topic 为报告主题。需先完成相关检索后再调用此工具。")
def generate_report(topic: str) -> str:
    query = f"{topic} 综合分析 投资价值 风险评估 行业前景"
    return rag.summarize(query)
