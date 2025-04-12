from .base import DexBaseTool
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class ImpermanentLossCalculator(DexBaseTool):
    name: str = "impermanent_loss_calculator"
    description: str = "计算Uniswap等AMM上的无偿损失(Impermanent Loss)"
    parameters: dict = {
        "type": "object",
        "properties": {
            "nft_id": {
                "type": "string",
                "description": "Uniswap v3 NFT ID"
            },
            "pool_address": {
                "type": "string",
                "description": "池子合约地址"
            },
            "mode": {
                "type": "string",
                "description": "计算模式: 'simple' 或 'position'"
            },
            "initial_price": {
                "type": "number",
                "description": "初始价格 (simple模式)"
            },
            "current_price": {
                "type": "number",
                "description": "当前价格 (simple模式)"
            }
        },
        "required": ["mode"]
    }
    
    # 获取NFT仓位信息的查询模板
    position_template: str = """
query UniswapV3PositionQuery {{
  ethereum(network: ethereum) {{
    address(address: {{is: "{nft_id}"}}) {{
      balances {{
        currency {{
          symbol
          address
        }}
        value
      }}
      smartContract {{
        attributes {{
          name
          value
        }}
      }}
    }}
    dexTrades(
      smartContractAddress: {{is: "{pool_address}"}}
      options: {{limit: 1, desc: "block.height"}}
    ) {{
      block {{
        height
        timestamp {{
          time
        }}
      }}
      buyAmount
      buyCurrency {{
        symbol
        address
      }}
      sellAmount
      sellCurrency {{
        symbol
        address
      }}
      transaction {{
        hash
      }}
    }}
  }}
}}
"""

    # 获取池子流动性分布的查询模板
    liquidity_template: str = """
query UniswapV3LiquidityQuery {{
  ethereum(network: ethereum) {{
    dexLiquidity(
      pool: {{is: "{pool_address}"}}
      options: {{limit: 100}}
    ) {{
      tickLower
      tickUpper
      liquidityNet
      liquidityGross
      timestamp {{
        time
      }}
    }}
  }}
}}
"""
    
    def calculate_il(self, initial_price, current_price):
        """
        计算基本的无偿损失百分比
        
        Args:
            initial_price: 提供流动性时的价格
            current_price: 当前价格
        
        Returns:
            float: 无偿损失百分比(负值)
        """
        price_ratio = current_price / initial_price
        il_percent = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
        return il_percent * 100
    
    async def get_position_data(self, nft_id, pool_address):
        """获取NFT仓位的详细信息"""
        response = await super().execute_query(
            self.position_template.format(nft_id=nft_id, pool_address=pool_address)
        )
        return response['data']['ethereum']
    
    async def get_liquidity_distribution(self, pool_address):
        """获取池子的流动性分布"""
        response = await super().execute_query(
            self.liquidity_template.format(pool_address=pool_address)
        )
        return response['data']['ethereum']['dexLiquidity']
    
    async def calculate_position_il(self, nft_id, pool_address):
        """
        计算特定Uniswap v3 NFT仓位的无偿损失
        
        Args:
            nft_id: Uniswap v3 NFT ID
            pool_address: 池子合约地址
        
        Returns:
            dict: 包含无偿损失信息的字典
        """
        # 获取NFT仓位信息
        position_data = await self.get_position_data(nft_id, pool_address)
        
        # 获取流动性分布
        liquidity_data = await self.get_liquidity_distribution(pool_address)
        
        # 解析NFT仓位数据
        nft_info = position_data['address']
        current_trade = position_data['dexTrades'][0]
        
        # 提取代币余额和当前价格
        token0 = nft_info['balances'][0]['currency']['symbol']
        token0_amount = float(nft_info['balances'][0]['value'])
        token1 = nft_info['balances'][1]['currency']['symbol']
        token1_amount = float(nft_info['balances'][1]['value'])
        
        # 计算当前价格
        current_price = float(current_trade['sellAmount']) / float(current_trade['buyAmount'])
        
        # 从NFT属性中获取初始价格(这需要根据实际数据格式调整)
        initial_price_attr = next((attr for attr in nft_info['smartContract']['attributes'] if attr['name'] == 'initialPrice'), None)
        initial_price = float(initial_price_attr['value']) if initial_price_attr else current_price * 0.9  # 如果找不到，使用默认值
        
        # 计算如果持有原始代币的价值(假设价格变化只影响token1的价格)
        token0_value = token0_amount
        token1_value_if_held = token1_amount * (current_price / initial_price)
        hodl_value = token0_value + token1_value_if_held
        
        # 计算当前LP价值
        current_value = token0_amount + token1_amount
        
        # 计算无偿损失
        il_usd = current_value - hodl_value
        il_percent = (il_usd / hodl_value) * 100
        
        # 分析流动性分布
        df = pd.DataFrame(liquidity_data)
        current_tick = np.log(current_price) / np.log(1.0001)
        
        # 分类流动性
        below_range = df[df['tickUpper'] < current_tick]['liquidityGross'].sum()
        in_range = df[(df['tickLower'] <= current_tick) & (df['tickUpper'] >= current_tick)]['liquidityGross'].sum()
        above_range = df[df['tickLower'] > current_tick]['liquidityGross'].sum()
        
        total_liquidity = below_range + in_range + above_range
        distribution = {
            "below_range_percent": (below_range / total_liquidity) * 100 if total_liquidity > 0 else 0,
            "in_range_percent": (in_range / total_liquidity) * 100 if total_liquidity > 0 else 0,
            "above_range_percent": (above_range / total_liquidity) * 100 if total_liquidity > 0 else 0
        }
        
        return {
            "nft_id": nft_id,
            "pool_address": pool_address,
            "token0": token0,
            "token1": token1,
            "initial_tokens": {
                "token0": token0_amount,
                "token1": token1_amount
            },
            "initial_price": initial_price,
            "current_price": current_price,
            "hodl_value": hodl_value,
            "current_value": current_value,
            "impermanent_loss_usd": il_usd,
            "impermanent_loss_percent": il_percent,
            "liquidity_distribution": distribution
        }
    
    async def execute(self, **params):
        """
        执行无偿损失计算
        
        Args:
            params: 包含必要参数的字典
                - mode: 'simple' 或 'position'
                - initial_price: 如果mode是'simple'
                - current_price: 如果mode是'simple'
                - nft_id: 如果mode是'position'
                - pool_address: 如果mode是'position'
        
        Returns:
            dict: 计算结果
        """
        mode = params.get('mode')
        
        if mode == 'simple':
            initial_price = params.get('initial_price')
            current_price = params.get('current_price')
            
            if not initial_price or not current_price:
                return {"error": "Simple mode requires initial_price and current_price"}
                
            il_percent = self.calculate_il(initial_price, current_price)
            return {
                "impermanent_loss_percent": il_percent,
                "initial_price": initial_price,
                "current_price": current_price
            }
            
        elif mode == 'position':
            nft_id = params.get('nft_id')
            pool_address = params.get('pool_address')
            
            if not nft_id or not pool_address:
                return {"error": "Position mode requires nft_id and pool_address"}
                
            return await self.calculate_position_il(nft_id, pool_address)
            
        else:
            return {"error": "Invalid mode specified. Use 'simple' or 'position'"}