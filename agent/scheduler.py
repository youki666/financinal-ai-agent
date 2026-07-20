"""定时报告调度器：基于 APScheduler + YAML 配置"""
import os
from datetime import datetime

import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from agent.notifier import EmailNotifier, WebhookNotifier
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class ReportScheduler:
    """读取 config/scheduler.yaml，定时生成报告并推送到指定渠道"""

    def __init__(self, rag_service):
        self.rag = rag_service
        self.scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
        self.email = EmailNotifier()
        self.webhook = WebhookNotifier()
        self.jobs_config = []
        self._job_run_history: dict[str, str] = {}

    def load_config(self, config_path: str | None = None):
        if config_path is None:
            config_path = get_abs_path("config/scheduler.yaml")
        if not os.path.exists(config_path):
            logger.warning(f"[Scheduler] 配置文件不存在: {config_path}，跳过")
            return
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.jobs_config = cfg.get("jobs", [])
        logger.info(f"[Scheduler] 加载了 {len(self.jobs_config)} 个定时任务")

    def start(self):
        if not self.jobs_config:
            self.load_config()
        if not self.jobs_config:
            logger.warning("[Scheduler] 无定时任务，调度器未启动")
            return

        for job_cfg in self.jobs_config:
            name = job_cfg.get("name", "unnamed")
            job_type = job_cfg.get("type", "rag")
            topic = job_cfg.get("topic", "")
            cron = job_cfg.get("cron", "0 9 * * *")
            channels = job_cfg.get("channels", ["email"])

            if not topic:
                logger.warning(f"[Scheduler] 跳过无效任务: {name}（缺少 topic）")
                continue

            self.scheduler.add_job(
                func=self._generate_and_send,
                trigger=CronTrigger.from_crontab(cron),
                args=[name, job_type, topic, channels],
                id=name,
                replace_existing=True,
            )
            logger.info(f"[Scheduler] 已注册: {name} | cron={cron} | channels={channels}")

        self.scheduler.start()
        logger.info(f"[Scheduler] 调度器已启动，共 {len(self.jobs_config)} 个任务")

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("[Scheduler] 调度器已停止")

    def run_now(self, job_name: str | None = None):
        """立即手动触发某个任务（用于测试或 UI 手动触发）"""
        if job_name:
            for job_cfg in self.jobs_config:
                if job_cfg.get("name") == job_name:
                    logger.info(f"[Scheduler] 手动触发: {job_name}")
                    self._generate_and_send(
                        job_cfg["name"],
                        job_cfg.get("type", "rag"),
                        job_cfg["topic"],
                        job_cfg.get("channels", ["email"]),
                    )
                    return
            logger.warning(f"[Scheduler] 未找到任务: {job_name}")
        else:
            for job_cfg in self.jobs_config:
                logger.info(f"[Scheduler] 手动触发: {job_cfg['name']}")
                self._generate_and_send(
                    job_cfg["name"],
                    job_cfg.get("type", "rag"),
                    job_cfg["topic"],
                    job_cfg.get("channels", ["email"]),
                )

    def _generate_and_send(self, name: str, job_type: str, topic: str, channels: list[str]):
        """执行单个定时任务：生成报告 → 多渠道推送

        job_type:
          - \"rag\":  使用 RAG 知识库检索 + LLM 总结（适合日报/周报/研报）
          - \"news\": 使用 flash_news + financial_news 获取实时新闻，RAG 做摘要（适合快讯推送）
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[Scheduler] ┌─ 开始执行: {name}")
        logger.info(f"[Scheduler] │  type={job_type}  topic={topic[:50]}...")

        if job_type == "news":
            report = self._build_news_report(topic)
        else:
            logger.info(f"[Scheduler] │  [RAG] 调用 rag_summarize → {topic[:50]}...")
            try:
                report = self.rag.summarize(topic)
                logger.info(f"[Scheduler] │  [RAG] 报告生成完成 ({len(report)} 字符)")
            except Exception as e:
                report = f"报告生成失败: {e}"
                logger.error(f"[Scheduler] │  [RAG] 报告生成异常: {e}")

        subject = f"[ResearchAI] {name} — {now_str}"

        results: dict[str, bool] = {}
        if "email" in channels:
            logger.info(f"[Scheduler] │  [EmailNotifier] 发送邮件 → {subject}")
            results["email"] = self.email.send(subject, report)
        if "dingtalk" in channels:
            logger.info(f"[Scheduler] │  [Webhook] 发送钉钉 → {name}")
            results["dingtalk"] = self.webhook.send_dingtalk(name, report)
        if "feishu" in channels:
            logger.info(f"[Scheduler] │  [Webhook] 发送飞书 → {name}")
            results["feishu"] = self.webhook.send_feishu(name, report)

        self._job_run_history[name] = now_str

        ok = [ch for ch, success in results.items() if success]
        failed = [ch for ch, success in results.items() if not success]
        if ok:
            logger.info(f"[Scheduler] └─ 推送成功: {ok}")
        if failed:
            logger.warning(
                f"[Scheduler] └─ 推送失败: {failed}，"
                f"请检查 .env 中的对应配置"
            )

    def _build_news_report(self, topic: str) -> str:
        """获取实时新闻并生成摘要报告"""
        from agent.tools.news_tools import flash_news, financial_news

        parts = []

        # 1. 实时快讯
        try:
            logger.info(f"[Scheduler] │  [flash_news] 获取实时快讯 (limit=15)")
            flash = flash_news.invoke({"limit": 15})
            lines = flash.count("\n") if flash else 0
            logger.info(f"[Scheduler] │  [flash_news] 返回 {lines} 行")
            parts.append(flash)
        except Exception as e:
            logger.error(f"[Scheduler] │  [flash_news] 失败: {e}")
            parts.append("快讯获取失败")

        # 2. 关键词新闻
        try:
            logger.info(f"[Scheduler] │  [financial_news] 检索新闻 (query={topic[:30]}..., days=1)")
            search = financial_news.invoke({"query": topic, "days": 1})
            logger.info(f"[Scheduler] │  [financial_news] 返回 {len(search)} 字符")
            parts.append(search)
        except Exception as e:
            logger.error(f"[Scheduler] │  [financial_news] 失败: {e}")

        raw_news = "\n\n".join(parts)

        # 3. 用 RAG 的 LLM 做摘要（不查知识库，直接用新闻内容）
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import PromptTemplate

            prompt = PromptTemplate.from_template(
                "你是金融新闻编辑。请将以下新闻内容整理成简洁的快讯摘要"
                "（5条核心新闻，每条不超过100字），用邮件正文格式输出,标题加粗：\n\n"
                "{news}"
            )
            chain = prompt | self.rag.chain.middle[0] | StrOutputParser()
            logger.info(f"[Scheduler] │  [LLM] 生成新闻摘要 ({min(len(raw_news), 4000)} 字符输入)")
            summary = chain.invoke({"news": raw_news[:4000]})
            logger.info(f"[Scheduler] │  [LLM] 摘要完成 ({len(summary)} 字符)")
            return f"## {topic} 快讯摘要\n\n{summary}\n\n---\n\n### 原始快讯\n\n{raw_news}"
        except Exception as e:
            logger.error(f"[Scheduler] │  [LLM] 新闻摘要生成失败: {e}")
            return f"## {topic} 快讯\n\n{raw_news}"

    def get_status(self) -> dict:
        """返回调度器状态（供 Streamlit UI 使用）"""
        return {
            "running": self.scheduler.running,
            "job_count": len(self.scheduler.get_jobs()),
            "jobs": [
                {
                    "name": job.id,
                    "next_run": str(job.next_run_time)[:19] if job.next_run_time else "N/A",
                    "last_manual": self._job_run_history.get(job.id, "N/A"),
                }
                for job in self.scheduler.get_jobs()
            ],
            "email_available": self.email.available,
        }
