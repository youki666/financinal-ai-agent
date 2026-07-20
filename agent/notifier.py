"""通知推送模块：邮箱 + 钉钉/飞书 Webhook"""
import os
from utils.logger_handler import logger


class EmailNotifier:
    """SMTP 邮件推送（每次发送时动态读取环境变量，支持运行时更新 .env 配置）"""

    @staticmethod
    def _read_config() -> dict:
        return {
            "host": os.getenv("SMTP_HOST", ""),
            "port": int(os.getenv("SMTP_PORT", "465")),
            "user": os.getenv("SMTP_USER", ""),
            "password": os.getenv("SMTP_PASS", ""),
            "to_addr": os.getenv("SMTP_TO", "") or os.getenv("SMTP_USER", ""),
        }

    @property
    def available(self) -> bool:
        cfg = self._read_config()
        return bool(cfg["host"] and cfg["user"] and cfg["password"])

    def send(self, subject: str, body: str) -> bool:
        cfg = self._read_config()
        if not cfg["host"] or not cfg["user"] or not cfg["password"]:
            logger.warning(
                f"[EmailNotifier] SMTP 未完整配置 (host={bool(cfg['host'])} "
                f"user={bool(cfg['user'])} pass={bool(cfg['password'])})，跳过邮件发送"
            )
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = cfg["user"]
            msg["To"] = cfg["to_addr"]

            html_body = _markdown_to_html(body)
            msg.attach(MIMEText(body, "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=15) as server:
                server.login(cfg["user"], cfg["password"])
                server.sendmail(cfg["user"], [cfg["to_addr"]], msg.as_string())

            logger.info(f"[EmailNotifier] 邮件发送成功: {subject}")
            return True

        except Exception as e:
            logger.error(f"[EmailNotifier] 邮件发送失败: {e}")
            return False


class WebhookNotifier:
    """钉钉 / 飞书 Webhook 推送"""

    def __init__(self):
        self.dingtalk_url = os.getenv("DINGTALK_WEBHOOK_URL", "")
        self.feishu_url = os.getenv("FEISHU_WEBHOOK_URL", "")

    def send_dingtalk(self, title: str, content: str) -> bool:
        if not self.dingtalk_url:
            return False

        try:
            import requests

            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": title,
                    "text": f"## {title}\n\n{content}",
                },
            }
            resp = requests.post(self.dingtalk_url, json=payload, timeout=10)
            if resp.status_code == 200 and resp.json().get("errcode") == 0:
                logger.info(f"[Webhook] 钉钉推送成功: {title}")
                return True
            else:
                logger.warning(f"[Webhook] 钉钉推送返回异常: {resp.text}")
                return False

        except Exception as e:
            logger.error(f"[Webhook] 钉钉推送失败: {e}")
            return False

    def send_feishu(self, title: str, content: str) -> bool:
        if not self.feishu_url:
            return False

        try:
            import requests

            payload = {
                "msg_type": "interactive",
                "card": {
                    "header": {"title": {"tag": "plain_text", "content": title}},
                    "elements": [
                        {"tag": "markdown", "content": content},
                    ],
                },
            }
            resp = requests.post(self.feishu_url, json=payload, timeout=10)
            if resp.status_code == 200 and resp.json().get("StatusCode") == 0:
                logger.info(f"[Webhook] 飞书推送成功: {title}")
                return True
            else:
                logger.warning(f"[Webhook] 飞书推送返回异常: {resp.text}")
                return False

        except Exception as e:
            logger.error(f"[Webhook] 飞书推送失败: {e}")
            return False

    def send(self, title: str, content: str, channels: list[str]) -> dict[str, bool]:
        """多渠道推送，返回各渠道发送结果"""
        results = {}
        if "dingtalk" in channels:
            results["dingtalk"] = self.send_dingtalk(title, content)
        if "feishu" in channels:
            results["feishu"] = self.send_feishu(title, content)
        return results


def _markdown_to_html(md_text: str) -> str:
    """简单的 Markdown → HTML 转换（用于邮件正文）"""
    import re

    html = md_text
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\|(.+)\|", lambda m: f"<tr>{''.join(f'<td>{c.strip()}</td>' for c in m.group(1).split('|'))}</tr>", html)
    html = re.sub(r"\n\n", "<br><br>", html)
    html = f'<div style="font-family: sans-serif; max-width: 700px;">{html}</div>'
    return html
