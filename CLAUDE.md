# ResearchAI — 二级市场研究报告分析平台

## 项目概述

基于 LangChain ReAct Agent + RAG 的 A 股二级市场研究分析平台。用户用中文自然语言提问个股/行业问题，Agent 从本地研报知识库（PDF）中检索相关内容，结合实时行情（AKShare）、新闻（NewsAPI/Eastmoney）、联网搜索（Tavily），生成带来源引用的分析回答或结构化研报。

部署为 **Streamlit 网页应用**：`streamlit run app.py`

## 技术栈

- **UI**: Streamlit 1.40.1
- **Agent 框架**: LangChain 1.3 + LangGraph 1.2.8（`create_agent`，9 个工具 + 6 个中间件）
- **LLM**: 阿里云百炼 DashScope（qwen-max，OpenAI 兼容接口），通过 `RoutableChatModel` 按查询关键词自动路由到 fast/standard/powerful 三个配置
- **Embedding**: DashScope text-embedding-v2
- **向量库**: ChromaDB 1.5.9 + BM25 混合检索 + RRF 融合
- **持久化**: SQLite（LangGraph checkpointer + 会话线程元数据）
- **调度**: APScheduler（定时生成日报/周报，推送到邮件/钉钉/飞书）
- **评估**: RAGAS
- **配置**: YAML（`config/`）+ python-dotenv

## 目录结构

```
├── app.py                     # Streamlit 主入口（UI、会话管理、流式展示）
├── run_eval.py                # RAGAS 评估 CLI 入口
├── step.txt                   # RAG 全流程技术文档（6 个 Phase 详解）
├── requirements.txt
├── .env / .env.example
│
├── agent/
│   ├── react_agent.py         # ReactAgent 类：构建 Agent、工具注册、线程管理、流式输出
│   ├── scheduler.py           # ReportScheduler：定时任务调度（APScheduler）
│   ├── notifier.py            # 推送通知：EmailNotifier / WebhookNotifier（钉钉/飞书）
│   └── tools/
│       ├── agent_tools.py     # RAG 工具：rag_summarize, stock_brief, industry_overview, generate_report
│       ├── stock_tools.py     # 行情工具：stock_quote_realtime, stock_history（AKShare）
│       ├── news_tools.py      # 新闻工具：financial_news（NewsAPI）, flash_news（AKShare/Eastmoney）
│       ├── search_tools.py    # web_search（Tavily）
│       └── middleware.py      # LangGraph 中间件：监控、引用追踪、研报提示词切换、质量守护
│
├── model/
│   ├── factory.py             # 模型配置（fast/standard/powerful）、chat_model 单例、Embedding 工厂
│   └── router.py              # ModelRouter + RoutableChatModel：多模型路由，对上层透明
│
├── rag/
│   ├── rag_service.py         # RagSummarizeService：完整 RAG 流水线（改写→检索→重排→组装→生成）
│   ├── vector_store.py        # VectorStoreService：ChromaDB + BM25 混合检索、RRF 融合、文档加载
│   ├── query_rewriter.py      # LLM 查询改写（1-3 个子查询），LRU 缓存 128 条
│   ├── reranker.py            # Jaccard 去重 + 启发式评分重排序
│   └── context_assembler.py   # 按来源分组 + Token 预算控制（~3000 tokens）组装上下文
│
├── config/                    # YAML 配置文件
│   ├── chroma.yaml            # 向量库：chunk_size=200, overlap=20, k=3
│   ├── rag.yaml               # 模型名：chat=qwen3.7-plus, embedding=text-embedding-v2
│   ├── scheduler.yaml         # 定时任务定义
│   ├── prompts.yml            # 提示词文件路径
│   └── agent.yaml
│
├── prompts/                   # 提示词模板（.txt）
│   ├── main_prompt.txt        # ReAct Agent 系统提示词
│   ├── rag_summarize.txt      # RAG 总结提示词
│   ├── report_prompt.txt      # 结构化研报生成提示词
│   └── query_rewrite.txt      # 查询改写提示词
│
├── utils/                     # 工具函数：配置加载、提示词加载、路径解析、文件加载、日志
├── evaluation/                # RAGAS 评估：eval_runner.py + test_dataset.py（12 条测试样本）
├── data/                      # 知识库 PDF + conversations.db（SQLite）
├── chroma_db/                 # ChromaDB 持久化目录
├── embedding/bge-large-zh-v1.5/  # 本地 BGE 中文 Embedding 模型（当前未使用，用 DashScope 云端）
├── assets/                    # 生成的研报样本
└── logs/                      # 日志文件（agent_YYYYMMDD.log）
```

## 常用命令

```bash
# 启动开发服务器
streamlit run app.py

# 运行 RAGAS 评估
python run_eval.py                          # 全量评估
python run_eval.py --metrics faithfulness   # 单指标
python run_eval.py --verbose                # 查看每题明细
python run_eval.py --output result.json     # 导出 JSON

# 安装依赖
pip install -r requirements.txt
```

## 核心架构与数据流

### Agent 执行流程

```
用户提问 → ReactAgent.execute_stream()
  ├── create_agent(model=RoutableChatModel, tools=[9个工具], middleware=[6个中间件])
  ├── LangGraph 线程管理（SqliteSaver → data/conversations.db）
  └── 流式输出 → Streamlit UI（tool_agent 状态标签 + AIMessageChunk 文本）
```

### RAG 流水线（`rag_service.py`）

```
query → QueryRewriter.rewrite() → 1~3 个子查询
  → 每个子查询: hybrid_search() [向量 + BM25 → RRF 融合] → Reranker 重排序
  → 多查询结果合并去重
  → ContextAssembler 按来源分组 + Token 预算控制
  → LLM 生成（rag_summarize.txt 提示词）
```

### 模型路由（`model/router.py`）

`RoutableChatModel` 实现 `BaseChatModel`，对 Agent 和 RAG 透明。每次调用根据查询关键词规则路由：
- 报告/研报/深度分析 → powerful（qwen-max）
- 分析/投资/估值/行业 → standard（qwen-max）
- 其他 → fast（qwen-max）

注意：当前三个配置都指向 `qwen-max`，描述中的 qwen-plus 未实际使用。`config/rag.yaml` 中的 `chat_model_name: qwen3.7-plus` 也未在 factory.py 中被引用。

## 开发规范与注意事项

### 代码风格
- 所有注释、日志、UI 文本使用中文
- 变量/函数名使用英文 snake_case
- 类型注解仅用于关键函数签名，不追求全覆盖
- 不写文档字符串，代码通过命名自解释

### 关键约定
- **API Key 获取**：优先 `os.getenv()`，回退 `st.secrets`（Streamlit Cloud 部署兼容）
- **路径解析**：使用 `utils/path_tool.py` 的 `get_abs_path()` 基于项目根目录解析
- **配置加载**：通过 `utils/config_handler.py` 的 `load_yaml()` 加载 `config/` 下的 YAML
- **日志**：使用 `utils/logger_handler.py` 的 `logger`，按天轮转，输出到 `logs/`
- **LLM 调用**：统一使用 `model/factory.py` 的 `chat_model`（RoutableChatModel 单例）和 `get_embed_model()`
- **提示词**：通过 `utils/prompt_loader.py` 从 `prompts/` 目录加载

### 数据库
- `data/conversations.db`：SQLite，存储 LangGraph checkpoints + 线程元数据表
- `chroma_db/`：ChromaDB 向量库持久化目录
- 两者都加入 `.gitignore`

### 易错点
- `RoutableChatModel` 流式输出会产生外层 provider 的重复 AIMessageChunk，`react_agent.py` 中有去重逻辑，修改时注意不要破坏
- BM25 索引在文档加载/删除时需重建，存储在内存中，每次启动从 ChromaDB 重新构建
- `config/rag.yaml` 的 `chat_model_name` 当前未被 factory.py 使用，factory.py 硬编码了模型名
- `embedding/bge-large-zh-v1.5/` 是本地 Embedding 模型文件，但当前实际使用 DashScope 云端 Embedding，本地模型未接入代码
- `docker-compose.yml` 为空文件，尚未配置
- 定时任务依赖 `config/scheduler.yaml`，使用 Asia/Shanghai 时区

### 依赖版本敏感项
- `langchain==1.3.11` + `langgraph==1.2.8`：使用 `create_agent` API，非旧版 `create_react_agent`
- `langgraph-checkpoint-sqlite==3.1.0`：`SqliteSaver.from_conn_string()` 用法
- `chromadb==1.5.9`：注意与 `langchain-chroma==0.2.6` 的兼容性
