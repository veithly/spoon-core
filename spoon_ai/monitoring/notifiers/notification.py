# spoon_ai/monitoring/notifiers/notification.py
import logging
from typing import Dict, Any, List, Optional
import asyncio
import inspect

logger = logging.getLogger(__name__)

class NotificationManager:
    """通知管理器，管理多个通知渠道，调用social_media目录中的通知类"""
    
    def __init__(self):
        self.channels = {}
        self._load_channels()
    
    def _load_channels(self):
        """加载所有可用的通知渠道"""
        # 加载Telegram
        try:
            from spoon_ai.social_media.telegram import TelegramClient
            from spoon_ai.agents.toolcall import ToolCallAgent
            
            class NotificationAgent(ToolCallAgent):
                """简化的Agent，仅用于发送通知"""
                def __init__(self):
                    pass
                
                async def run(self, text):
                    return "Notification only"
                
                def clear(self):
                    pass
                
                @property
                def memory(self):
                    class DummyMemory:
                        def get_messages(self):
                            return []
                    return DummyMemory()
                
                @property
                def state(self):
                    return None
                
                @state.setter
                def state(self, value):
                    pass
            
            self.channels["telegram"] = {
                "instance": TelegramClient(NotificationAgent())
            }
            logger.info("Registered Telegram notification channel")
        except Exception as e:
            logger.warning(f"Failed to register Telegram channel: {str(e)}")
        
        # 加载Twitter
        try:
            from spoon_ai.social_media.twitter import TwitterClient
            self.channels["twitter"] = {
                "instance": TwitterClient()
            }
            logger.info("Registered Twitter notification channel")
        except Exception as e:
            logger.warning(f"Failed to register Twitter channel: {str(e)}")
        
        # 加载Email
        try:
            from spoon_ai.social_media.email import EmailNotifier
            self.channels["email"] = {
                "instance": EmailNotifier()
            }
            logger.info("Registered Email notification channel")
        except Exception as e:
            logger.warning(f"Failed to register Email channel: {str(e)}")
    
    async def _run_async_method(self, method, *args, **kwargs):
        """运行异步方法并等待结果"""
        return await method(*args, **kwargs)
        
    def send(self, channel: str, message: str, **kwargs) -> bool:
        """通过指定渠道发送通知"""
        if channel not in self.channels:
            logger.error(f"Notification channel not available: {channel}")
            return False
            
        try:
            logger.info(f"Attempting to send notification via {channel}")
            logger.info(f"Notification channels available: {self.channels.keys()}")
            
            instance = self.channels[channel]["instance"]
            logger.info(f"Using {channel} instance: {type(instance).__name__}")
            
            # 记录参数
            safe_kwargs = kwargs.copy()
            if "password" in safe_kwargs:
                safe_kwargs["password"] = "******"  # 隐藏密码
            logger.info(f"Notification params: {safe_kwargs}")
            
            # 根据不同渠道调用不同的方法
            if channel == "telegram":
                # Telegram使用异步的send_proactive_message方法
                chat_id = kwargs.get("chat_id")
                method = instance.send_proactive_message
                
                # 检查是否需要传递chat_id
                if chat_id:
                    # 运行异步方法
                    logger.info(f"Sending Telegram message with chat_id: {chat_id}")
                    loop = asyncio.get_event_loop()
                    if not loop.is_running():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    loop.run_until_complete(method(message, chat_id))
                else:
                    logger.info("Sending Telegram message without specific chat_id")
                    loop = asyncio.get_event_loop()
                    if not loop.is_running():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    loop.run_until_complete(method(message))
                
                logger.info(f"Telegram notification sent successfully")
                return True
            else:
                # Twitter和Email使用同步的send方法
                method = instance.send
                logger.info(f"Calling {type(instance).__name__}.send method")
                
                # 记录发送的消息摘要
                msg_preview = message[:100] + "..." if len(message) > 100 else message
                logger.info(f"Message preview: {msg_preview}")
                
                result = method(message, **kwargs)
                logger.info(f"Send result: {result}")
                return result
                    
        except Exception as e:
            logger.error(f"Failed to send notification via {channel}: {str(e)}")
            # 打印完整的错误堆栈
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_available_channels(self) -> List[str]:
        """获取所有可用的通知渠道"""
        return list(self.channels.keys())

    def send_to_all(self, message: str, channels: Optional[List[str]] = None, **kwargs) -> Dict[str, bool]:
        """
        向多个渠道发送相同的通知
        
        Args:
            message: 通知内容
            channels: 要使用的渠道列表，如果为None则使用所有可用渠道
            **kwargs: 渠道特定的参数
        
        Returns:
            Dict[str, bool]: 每个渠道的发送结果
        """
        if channels is None:
            channels = self.get_available_channels()
            
        results = {}
        for channel in channels:
            channel_kwargs = kwargs.get(channel, {})
            results[channel] = self.send(channel, message, **channel_kwargs)
            
        return results