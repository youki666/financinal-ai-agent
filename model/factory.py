import os

from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

from model.router import LLMConfig, ModelRouter, RoutableChatModel
from utils.config_handler import rag_conf

load_dotenv()

# ============================================================
# 嵌入模型（不变，始终用同一个）
# ============================================================
embed_model = DashScopeEmbeddings(model=rag_conf["embedding_model_name"])

# ============================================================
# 多模型配置（可按需扩展）
# ============================================================
model_configs: dict[str, LLMConfig] = {
    "fast": LLMConfig(
        provider="openai",
        model_name="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.0,
        description="标准模型 (qwen-plus) — 简单问答",
    ),
    "standard": LLMConfig(
        provider="openai",
        model_name="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.0,
        description="标准模型 (qwen-plus) — 个股/行业分析",
    ),
    "powerful": LLMConfig(
        provider="openai",
        model_name="deepseek-v4-pro",
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.0,
        description="强力模型 (qwen-plus) — 研报生成/复杂分析",
    ),
}

# ============================================================
# 可路由模型（对上层透明，自动按查询内容切换）
# ============================================================
chat_model = RoutableChatModel(router=ModelRouter(configs=model_configs))

if __name__ == "__main__":
    queries = [
        "今日金价多少",
        "分析紫金矿业投资价值",
        "生成一份有色金属行业研究报告",
    ]
    for q in queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {q}")
        print(chat_model.invoke(q))
