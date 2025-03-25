import os
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class EmailNotifier:
    """邮件通知发送器，用于监控警报"""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """从环境变量加载邮件配置"""
        load_dotenv()
        
        config = {
            "smtp_server": os.getenv("EMAIL_SMTP_SERVER"),
            "smtp_port": int(os.getenv("EMAIL_SMTP_PORT", "587")),
            "smtp_user": os.getenv("EMAIL_SMTP_USER"),
            "smtp_password": os.getenv("EMAIL_SMTP_PASSWORD"),
            "from_email": os.getenv("EMAIL_FROM"),
            "default_recipients": [
                email.strip() for email in os.getenv("EMAIL_DEFAULT_RECIPIENTS", "").split(",") if email.strip()
            ]
        }
        
        # 验证必要的配置项
        missing = []
        for key in ["smtp_server", "smtp_user", "smtp_password"]:
            if not config.get(key):
                missing.append(key)
        
        if missing:
            logger.warning(f"Missing email configuration: {', '.join(missing)}")
        
        return config
    
    def send(self, message: str, to_emails: Optional[List[str]] = None,
             subject: str = "Crypto Monitoring Alert", 
             html_format: bool = True, **kwargs) -> bool:
        """
        发送邮件通知
        
        Args:
            message: 邮件内容
            to_emails: 收件人列表，如果为None则使用默认收件人
            subject: 邮件主题
            html_format: 是否以HTML格式发送
            **kwargs: 其他SMTP参数
        
        Returns:
            bool: 是否发送成功
        """
        # 获取SMTP配置
        smtp_server = self.config.get("smtp_server")
        smtp_port = self.config.get("smtp_port", 587)
        smtp_user = self.config.get("smtp_user")
        smtp_password = self.config.get("smtp_password")
        
        if not all([smtp_server, smtp_user, smtp_password]):
            logger.error("SMTP configuration is incomplete")
            return False
        
        # 确定发件人和收件人
        from_email = kwargs.get("from_email") or self.config.get("from_email", smtp_user)
        recipients = to_emails or self.config.get("default_recipients", [])
        
        if not recipients:
            logger.error("No recipients specified for email")
            return False
        
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject
            
            # 添加内容
            if html_format:
                # 如果消息已经包含HTML标签，直接使用它
                if not (message.startswith('<') and message.endswith('>')):
                    # 转换换行符为<br>
                    message = message.replace('\n', '<br>')
                    # 添加基本HTML结构
                    message = f"""
                    <html>
                    <body>
                    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                        {message}
                    </div>
                    </body>
                    </html>
                    """
                msg.attach(MIMEText(message, 'html'))
            else:
                msg.attach(MIMEText(message, 'plain'))
            
            # 连接SMTP服务器
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                
                # 发送邮件
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {recipients}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False