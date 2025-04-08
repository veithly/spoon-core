#!/usr/bin/env python
# btc_price_agent/advanced_monitor.py

import os
import sys
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import requests
import json

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 导入 SpoonAI 组件
from spoon_ai.chat import ChatBot, Memory
from spoon_ai.schema import Message
from spoon_ai.monitoring.clients.base import DataClient
from spoon_ai.monitoring.clients.cex import get_cex_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("btc-advanced-monitor")

class BTCAdvancedMonitor:
    """比特币高级价格监控，使用 LLM 进行趋势分析"""
    
    def __init__(
        self,
        llm_provider: str = "anthropic",
        model_name: str = "claude-3-7-sonnet-20250219",
        market: str = "cex",
        provider: str = "binance",
        symbol: str = "BTCUSDT"
    ):
        # 初始化 ChatBot
        self.chatbot = ChatBot(
            llm_provider=llm_provider,
            model_name=model_name,
            api_key=os.getenv("ANTHROPIC_API_KEY") if llm_provider == "anthropic" else os.getenv("OPENAI_API_KEY")
        )
        
        # 初始化记忆
        self.memory = Memory()
        
        # 初始化数据客户端
        self.client = get_cex_client(provider)
        self.market = market
        self.provider = provider
        self.symbol = symbol
        
        # 系统消息
        self.system_message = """你是一个专业的加密货币市场分析师，擅长技术分析和市场趋势预测。
        你需要基于提供的价格和交易数据，分析比特币的市场状况并提供见解。
        分析需要包括:
        1. 价格趋势和动量
        2. 主要支撑和阻力位
        3. 交易量分析
        4. 市场周期定位
        5. 短期价格走势预测
        
        请使用专业但易懂的语言，避免过于复杂的术语，并尽量提供具体的价格水平和百分比。
        """
        
        logger.info(f"BTC 高级分析监控已初始化，使用 {provider} 数据源监控 {symbol}")
    
    async def get_market_data(self) -> Dict[str, Any]:
        """获取市场数据，包括当前价格、24小时统计和K线数据"""
        try:
            # 获取当前价格
            price_data = self.client.get_ticker_price(self.symbol)
            
            # 获取24小时统计
            stats_24h = self.client.get_ticker_24h(self.symbol)
            
            # 获取K线数据 (1 天 7 根 K 线)
            klines = self.client.get_klines(self.symbol, "1d", 7)
            
            # 格式化 K 线数据
            formatted_klines = []
            for k in klines:
                formatted_klines.append({
                    "open_time": datetime.fromtimestamp(k[0]/1000).strftime("%Y-%m-%d"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            
            # 返回整合后的数据
            return {
                "current_price": float(price_data["price"]),
                "price_change_24h": float(stats_24h["priceChange"]),
                "price_change_percent_24h": float(stats_24h["priceChangePercent"]),
                "volume_24h": float(stats_24h["volume"]),
                "high_24h": float(stats_24h["highPrice"]),
                "low_24h": float(stats_24h["lowPrice"]),
                "klines_daily": formatted_klines,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"获取市场数据时出错: {str(e)}")
            raise
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """获取市场情绪数据（可以从外部 API 获取或使用简单的启发式方法）"""
        try:
            # 这里是简单的启发式方法，实际应用中可以集成外部 API
            market_data = await self.get_market_data()
            
            # 基于价格变化计算简单情绪评分
            price_change = market_data["price_change_percent_24h"]
            
            # 简单的情绪评分计算
            if price_change > 5:
                sentiment = "极度乐观"
                score = 90
            elif price_change > 2:
                sentiment = "乐观"
                score = 70
            elif price_change > 0:
                sentiment = "略微乐观"
                score = 60
            elif price_change > -2:
                sentiment = "略微悲观"
                score = 40
            elif price_change > -5:
                sentiment = "悲观"
                score = 30
            else:
                sentiment = "极度悲观"
                score = 10
            
            return {
                "sentiment": sentiment,
                "score": score,
                "based_on": "价格变化百分比",
                "price_change": price_change
            }
        except Exception as e:
            logger.error(f"获取市场情绪数据时出错: {str(e)}")
            return {
                "sentiment": "未知",
                "score": 50,
                "based_on": "无数据",
                "price_change": 0
            }
    
    async def analyze_market(self) -> str:
        """使用 LLM 分析市场状况"""
        try:
            # 获取市场数据
            market_data = await self.get_market_data()
            sentiment_data = await self.get_market_sentiment()
            
            # 创建提示信息
            user_message = Message(
                role="user", 
                content=f"""请基于以下比特币市场数据提供详细分析:

交易对: {self.symbol}
当前价格: {market_data['current_price']} USD
24小时价格变化: {market_data['price_change_24h']} USD ({market_data['price_change_percent_24h']}%)
24小时交易量: {market_data['volume_24h']} USD
24小时最高价: {market_data['high_24h']} USD
24小时最低价: {market_data['low_24h']} USD
市场情绪: {sentiment_data['sentiment']} (评分: {sentiment_data['score']}/100)
时间: {market_data['timestamp']}

近7天K线数据:
{json.dumps(market_data['klines_daily'], indent=2, ensure_ascii=False)}

请提供分析，包括:
1. 价格趋势概述
2. 重要支撑/阻力位
3. 近期交易量分析
4. 短期价格预测
5. 具体的操作建议
"""
            )
            
            # 清除历史记忆，确保每次分析基于最新数据
            self.memory.clear()
            self.memory.add_message(user_message)
            
            # 获取 LLM 回复
            response = await self.chatbot.ask(self.memory.get_messages(), system_msg=self.system_message)
            
            # 记录分析结果
            logger.info(f"生成了比特币市场分析报告")
            
            return response
        except Exception as e:
            logger.error(f"分析市场时出错: {str(e)}")
            return f"市场分析生成失败: {str(e)}"
    
    async def run_scheduled_analysis(self, interval_hours: int = 6):
        """按计划运行市场分析"""
        logger.info(f"开始按计划运行市场分析，间隔 {interval_hours} 小时")
        
        while True:
            try:
                analysis = await self.analyze_market()
                
                # 这里可以添加发送分析结果的逻辑，例如发送到 Telegram
                # 例如: await self.send_to_telegram(analysis)
                
                logger.info(f"下一次分析将在 {interval_hours} 小时后进行")
                await asyncio.sleep(interval_hours * 60 * 60)  # 转换为秒
                
            except Exception as e:
                logger.error(f"调度分析时出错: {str(e)}")
                # 出错后等待一段时间再重试
                await asyncio.sleep(60 * 10)  # 10分钟后重试
    
    async def send_to_telegram(self, message: str, chat_id: str = None) -> bool:
        """发送分析结果到 Telegram"""
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        if not telegram_token or not chat_id:
            logger.error("缺少 Telegram 配置，无法发送消息")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            logger.info(f"成功发送分析结果到 Telegram")
            return True
            
        except Exception as e:
            logger.error(f"发送到 Telegram 时出错: {str(e)}")
            return False

async def main():
    """主函数"""
    try:
        # 创建高级监控器
        monitor = BTCAdvancedMonitor(
            provider="binance",
            symbol="BTCUSDT"
        )
        
        # 直接运行一次分析
        analysis = await monitor.analyze_market()
        print("\n===== 比特币市场分析 =====")
        print(analysis)
        print("==========================\n")
        
        # 可选：发送到 Telegram
        # await monitor.send_to_telegram(analysis)
        
        # 开始定期分析
        await monitor.run_scheduled_analysis(interval_hours=6)
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 