"""
LLM-Driven Real Cryptocurrency Analysis System

This system is completely driven by LLM with intelligent analysis decisions at each step:
1. Step 1: Call real Binance API to get market data (no simulated data)
2. Step 2: LLM intelligently selects tokens to analyze based on real data
3. Step 3: Use PowerData to get technical indicators for selected tokens
4. Step 4: LLM analyzes data and decides next actions
5. Final: LLM summarizes all information to generate investment recommendations

Features:
- Real Binance API - no simulated data
- Completely LLM-driven decision flow
- Real API calls (Binance + PowerData)
- Intelligent routing and analysis
- Enhanced Graph system
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Set, Annotated
from datetime import datetime

from spoon_ai.graph import (
    StateGraph,
    NodeContext,
    NodeResult,
    RouterResult,
    node_decorator,
    router_decorator,
    merge_dicts,
    append_history,
    union_sets,
    validate_range,
    validate_enum,
    operator
)

from spoon_ai.llm.manager import get_llm_manager
from spoon_ai.schema import Message
from spoon_ai.tools.crypto_tools import get_crypto_tools
from spoon_ai.tools.tool_manager import ToolManager
from spoon_ai.tools.mcp_tool import MCPTool

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce log output
logger = logging.getLogger(__name__)


def format_price(price: float) -> str:
    """Format price with appropriate precision for display"""
    if price == 0:
        return "$0.00"

    if price >= 1:
        return f"${price:,.2f}"
    elif price >= 0.01:
        return f"${price:.4f}"
    elif price >= 0.0001:
        return f"${price:.6f}"
    else:
        # For very small numbers like PEPE, show 2 significant digits
        import math
        if price == 0:
            return "$0.00"

        # Find the number of decimal places needed for 2 significant digits
        log_val = math.floor(math.log10(abs(price)))
        decimal_places = max(2, 2 - log_val)
        return f"${price:.{int(decimal_places)}f}"


def save_analysis_report(result: Dict[str, Any], filename: str = None) -> str:
    """Save analysis report as Markdown document"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crypto_analysis_report_{timestamp}.md"

    # Generate markdown content
    md_content = generate_markdown_report(result)

    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(md_content)

    return filename


def generate_markdown_report(result: Dict[str, Any]) -> str:
    """Generate markdown content for the analysis report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_lines = [
        "# Cryptocurrency Analysis Report",
        f"*Generated on {timestamp}*",
        "",
        "## Executive Summary",
        ""
    ]

    # Add executive summary
    confidence_score = result.get('confidence_score', 0)
    risk_level = result.get('risk_level', 'N/A')

    md_lines.extend([
        f"- **Overall Confidence**: {confidence_score:.1%}",
        f"- **Risk Level**: {risk_level}",
        f"- **Data Sources**: Binance API + PowerData + LLM Analysis",
        ""
    ])

    # Add market overview
    binance_data = result.get('binance_market_data', {})
    if binance_data:
        md_lines.extend([
            "## Market Overview",
            "",
            f"- **Total Trading Pairs Available**: {binance_data.get('total_pairs_available', 0)}",
            f"- **Selected Active Pairs**: {binance_data.get('selected_pairs_count', 0)}",
            f"- **Data Collection Time**: {binance_data.get('timestamp', 'N/A')[:19]}",
            ""
        ])

        top_pairs = binance_data.get('top_pairs', [])[:5]
        if top_pairs:
            md_lines.extend([
                "### Top 5 Most Active Trading Pairs",
                ""
            ])
            for pair in top_pairs:
                change = pair['priceChangePercent']
                volume = pair['quoteVolume']
                md_lines.append(f"- **{pair['symbol']}**: {change:+.2f}% (Volume: {volume:,.0f} USDT)")
            md_lines.append("")

    # Add news analysis
    news_data = result.get('news_data', {})
    if news_data:
        md_lines.extend([
            "## News Analysis",
            ""
        ])

        analysis_type = news_data.get('analysis_type', 'unknown')
        if analysis_type == 'llm_generated':
            news_analysis = news_data.get('data', {})
            market_analysis = news_analysis.get('market_news_analysis', {})

            if market_analysis:
                sentiment = market_analysis.get('overall_sentiment', 'neutral').upper()
                md_lines.append(f"- **Overall Sentiment**: {sentiment}")

                key_themes = market_analysis.get('key_themes', [])
                if key_themes:
                    md_lines.append(f"- **Key Themes**: {', '.join(key_themes)}")

                market_drivers = market_analysis.get('market_drivers', [])
                if market_drivers:
                    md_lines.append(f"- **Market Drivers**: {', '.join(market_drivers)}")

                md_lines.append("")

    # Add comprehensive analysis
    comprehensive_analysis = result.get('comprehensive_analysis', {})
    if comprehensive_analysis:
        analysis_result = comprehensive_analysis.get('analysis_result', {})
        comprehensive = analysis_result.get('comprehensive_analysis', {})

        if comprehensive:
            md_lines.extend([
                "## LLM Comprehensive Analysis",
                "",
                f"- **Overall Trend**: {comprehensive.get('overall_market_trend', 'N/A').upper()}",
                f"- **Market Sentiment**: {comprehensive.get('market_sentiment', 'N/A').upper()}",
                f"- **Risk Level**: {comprehensive.get('risk_level', 'N/A').upper()}",
                ""
            ])

            key_insights = comprehensive.get('key_insights', [])
            if key_insights:
                md_lines.extend([
                    "### Key Insights",
                    ""
                ])
                for insight in key_insights:
                    md_lines.append(f"- {insight}")
                md_lines.append("")

            market_summary = analysis_result.get('market_summary', '')
            if market_summary:
                md_lines.extend([
                    "### Market Summary",
                    "",
                    market_summary,
                    ""
                ])

    # Add LLM trading advice
    trading_advice = result.get('trading_advice', '')
    if trading_advice and trading_advice.strip():
        md_lines.extend([
            "## Professional Trading Recommendations",
            "",
            trading_advice,
            "",
            "---",
            ""
        ])
    else:
            md_lines.extend([
            "## Trading Recommendations",
            "",
            "Unable to generate specific trading recommendations.",
            "",
            "---",
            ""
        ])

    # Add risk disclaimer
    md_lines.extend([
        "## Risk Disclaimer",
        "",
        "⚠️ **Important Notice**:",
        "",
        "- This analysis is for informational purposes only and does not constitute investment advice",
        "- Cryptocurrency investments carry high risk - please make decisions carefully",
        "- It is recommended to combine multiple analysis methods and risk management strategies",
        "- Past performance does not guarantee future results",
        "",
        "---",
        f"*Report generated by LLM-Driven Crypto Analysis System on {timestamp}*"
    ])

    return "\n".join(md_lines)


# State Schema
class LLMCryptoAnalysisState(Dict[str, Any]):
    """LLM-driven cryptocurrency analysis state"""
    # Basic query
    query: str

    # Real market data
    binance_market_data: Annotated[Dict[str, Any], merge_dicts]

    # Token selection and details
    selected_tokens: Annotated[List[str], operator.add]
    token_details: Annotated[Dict[str, Any], merge_dicts]

    # Technical and news data
    market_data: Annotated[Dict[str, Any], merge_dicts]
    news_data: Annotated[Dict[str, Any], merge_dicts]

    # LLM analysis results
    comprehensive_analysis: Annotated[Dict[str, Any], merge_dicts]
    technical_analysis: Annotated[Dict[str, Any], merge_dicts]

    # Execution tracking
    execution_history: Annotated[List[Dict[str, Any]], append_history]
    analysis_flags: Annotated[Set[str], union_sets]

    # Final results
    final_recommendation: str
    confidence_score: Annotated[float, validate_range(0.0, 1.0)]
    risk_level: Annotated[str, validate_enum(["LOW", "MEDIUM", "HIGH", "EXTREME"])]


class LLMCryptoAnalyzer:
    """LLM-driven cryptocurrency analyzer"""

    def __init__(self):
        """Initialize the analyzer"""
        logger.info("Initializing LLM-driven cryptocurrency analyzer")

        # Initialize LLM manager
        self.llm_manager = get_llm_manager()

        # Initialize tools
        self.crypto_tools = get_crypto_tools()
        self.tool_manager = ToolManager(self.crypto_tools)

        # Initialize Tavily search MCP tool with better error handling
        self.tavily_search_tool = None
        try:
            tavily_key = os.getenv("TAVILY_API_KEY", "")
            if tavily_key and "your-tavily-api-key-here" not in tavily_key:
                # Test if npx is available
                import subprocess
                try:
                    result = subprocess.run(["npx", "--version"], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        self.tavily_search_tool = MCPTool(
                            name="tavily-search",
                            description="Performs a web search using the Tavily API for cryptocurrency news and market sentiment.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Search query for cryptocurrency news and analysis"
                                    },
                                    "max_results": {
                                        "type": "integer",
                                        "description": "Maximum number of search results to return",
                                        "default": 5
                                    },
                                    "search_depth": {
                                        "type": "string",
                                        "description": "Search depth: basic or advanced",
                                        "default": "basic"
                                    }
                                },
                                "required": ["query"]
                            },
                            mcp_config={
                                "command": "npx",
                                "args": ["--yes", "tavily-mcp"],
                                "env": {"TAVILY_API_KEY": tavily_key},
                                "timeout": 30,  # Increased timeout
                                "max_retries": 2  # Add retry support
                            }
                        )
                        logger.info("Tavily search MCP tool initialized successfully")
                    else:
                        logger.warning("npx not available, Tavily search tool disabled")
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    logger.warning("npx not found or timeout, Tavily search tool disabled")
            else:
                logger.warning("TAVILY_API_KEY not set or is placeholder, Tavily search tool unavailable")

        except Exception as e:
            logger.warning(f"Tavily MCP tool initialization failed: {e}")
            self.tavily_search_tool = None

        # Import crypto_powerdata CEX tool directly
        try:
            from spoon_toolkits.crypto.crypto_powerdata.tools import CryptoPowerDataCEXTool
            self.powerdata_cex_tool = CryptoPowerDataCEXTool()
            logger.info("Successfully imported crypto_powerdata CEX tool")
        except ImportError as e:
            logger.error(f"Cannot import crypto_powerdata tool: {e}")
            self.powerdata_cex_tool = None

        logger.info(f"Initialization complete, LLM manager ready, PowerData CEX tool: {'found' if self.powerdata_cex_tool else 'not found'}, Tavily search: {'found' if self.tavily_search_tool else 'not found'}")

    def create_analysis_graph(self) -> StateGraph:
        """Create graph with true parallel branches for per-token analysis.

        Flow:
        1) fetch_binance_data -> prepare_token_list
        2) analyze_token_0..9 (parallel group: token_analysis) – each node picks a token by index from state
        3) llm_final_aggregation -> END
        """
        graph = StateGraph(LLMCryptoAnalysisState)

        # Sequential preparation nodes
        graph.add_node("fetch_binance_data", self._wrap_method(self.fetch_binance_market_data))
        graph.add_node("prepare_token_list", self._wrap_method(self.prepare_token_list))

        # Create 10 parallel per-index analyzer nodes using the graph's native parallel group feature
        # Each node will dynamically pick a token from state based on its index
        analyzer_entry_node = None
        for index in range(10):
            node_name = f"analyze_token_{index}"
            if analyzer_entry_node is None:
                analyzer_entry_node = node_name
            graph.add_node(
                name=node_name,
                action=self._wrap_method(self.create_token_analyzer_by_index(index)),
                parallel_group="token_analysis"
            )

        # Final aggregation and ranking
        graph.add_node("llm_final_aggregation", self._wrap_method(self.llm_final_aggregation))

        # Sequential preparation flow
        graph.add_edge("fetch_binance_data", "prepare_token_list")

        # Connect to the entry node of the parallel branch; the graph engine will execute the whole branch
        if analyzer_entry_node:
            graph.add_edge("prepare_token_list", analyzer_entry_node)
            graph.add_edge(analyzer_entry_node, "llm_final_aggregation")

        # Final aggregation
        graph.add_edge("llm_final_aggregation", "END")

        # Set entry point and monitoring
        graph.set_entry_point("fetch_binance_data")
        graph.enable_monitoring([
            "execution_time",
            "llm_response_quality",
            "api_success_rate",
            "analysis_depth",
            "parallel_branch_efficiency"
        ])

        return graph

    def _wrap_method(self, method):
        """Wrap method to handle self binding"""
        # Create wrapper function without decorators
        if hasattr(method, '__name__') and 'routing_decision' in method.__name__:
            # Router method
            async def router_wrapper(state, context):
                return await method(state, context)
            return router_decorator(router_wrapper)
        else:
            # Regular node method
            async def node_wrapper(state, context):
                return await method(state, context)
            return node_decorator(node_wrapper)

    async def _call_llm(self, messages: List[Message], provider: str = None) -> str:
        """Call LLM and return response content"""
        try:
            response = await self.llm_manager.chat(messages, provider=provider)
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"LLM call failed: {str(e)}"

    async def fetch_binance_market_data(self, state: Dict[str, Any], context: NodeContext) -> NodeResult:
        """Step 1: Call Binance API to get real market data and select top 10 by volume (excluding stablecoins)."""
        logger.info(f"[{context.node_name}] Calling real Binance API to get market data")

        try:
            # Use direct HTTP request to call Binance API, avoiding tool parameter issues
            import aiohttp

            async with aiohttp.ClientSession() as session:
                # Get 24-hour price change statistics
                url = "https://api.binance.com/api/v3/ticker/24hr"
                async with session.get(url) as response:
                    if response.status == 200:
                        binance_data = await response.json()
                    else:
                        return NodeResult(
                            error=f"Binance API call failed, status code: {response.status}",
                            confidence=0.0
                        )

            # Define stablecoin list (to be filtered out)
            stablecoins = {
                'USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'BUSDUSDT', 'DAIUSDT',
                'USDPUSDT', 'FRAXUSDT', 'LUSDUSDT', 'SUSDUSDT', 'USTCUSDT',
                'USDDUSDT', 'GUSDUSDT', 'PAXGUSDT', 'USTUSDT'
            }

            # Filter and sort data - keep only USDT pairs, exclude stablecoins
            usdt_pairs = []
            for item in binance_data:
                if isinstance(item, dict) and item.get('symbol', '').endswith('USDT'):
                    symbol = item.get('symbol', '')

                    # Skip stablecoin trading pairs
                    if symbol in stablecoins:
                        continue

                    try:
                        # Ensure data integrity
                        if all(key in item for key in ['symbol', 'priceChangePercent', 'volume', 'lastPrice']):
                            usdt_pairs.append({
                                'symbol': symbol,
                                'priceChangePercent': float(item['priceChangePercent']),
                                'volume': float(item['volume']),
                                'lastPrice': float(item['lastPrice']),
                                'count': int(item.get('count', 0)),
                                'quoteVolume': float(item.get('quoteVolume', 0))
                            })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid data item: {e}")
                        continue

            # Sort by trading volume, take top 10 (after excluding stablecoins)
            top_pairs_by_volume = sorted(usdt_pairs, key=lambda x: x['quoteVolume'], reverse=True)[:10]

            logger.info(f"After filtering stablecoins, selected top 10 tokens by volume: {[p['symbol'] for p in top_pairs_by_volume]}")

            return NodeResult(
                updates={
                    "binance_market_data": {
                        "top_pairs": top_pairs_by_volume,
                        "total_pairs_available": len(usdt_pairs),
                        "selected_pairs_count": len(top_pairs_by_volume),
                        "timestamp": datetime.now().isoformat(),
                        "source": "binance_api_real",
                        "api_endpoint": "https://api.binance.com/api/v3/ticker/24hr",
                        "stablecoins_filtered": True,
                        "selection_criteria": "top_10_by_volume_excluding_stablecoins"
                    },
                    "execution_history": {
                        "action": "fetch_binance_data",
                        "pairs_fetched": len(top_pairs_by_volume),
                        "api_used": "binance_official",
                        "real_data": True,
                        "stablecoins_filtered": True
                    },
                    "analysis_flags": {"binance_data_fetched"}
                },
                metadata={
                    "api_source": "binance_official",
                    "pairs_count": len(top_pairs_by_volume),
                    "data_type": "24hr_ticker",
                    "real_data": True,
                    "stablecoins_excluded": len(stablecoins)
                },
                logs=[f"Successfully fetched top 15 tokens by volume from Binance API (excluded {len(stablecoins)} stablecoins)"],
                confidence=0.95,
                reasoning="Direct call to Binance official API to get real market data, excluding stablecoin trading pairs"
            )

        except Exception as e:
            logger.error(f"Binance API call failed: {e}")
            return NodeResult(
                error=f"Binance API call failed: {str(e)}",
                confidence=0.0
            )

    async def prepare_token_list(self, state: Dict[str, Any], context: NodeContext) -> NodeResult:
        """Step 2: Prepare top-10 token list by volume (stablecoins excluded)."""
        logger.info(f"[{context.node_name}] Preparing top 10 token list by volume")

        try:
            # Get real Binance data
            binance_data = state.get("binance_market_data", {})
            top_pairs = binance_data.get("top_pairs", [])

            if not top_pairs:
                return NodeResult(
                    error="No Binance market data available",
                    confidence=0.0
                )

            # Extract token symbols (remove USDT suffix)
            selected_tokens = [pair["symbol"].replace("USDT", "") for pair in top_pairs]

            # Build detailed token information
            token_details = {}
            for pair in top_pairs:
                token = pair["symbol"].replace("USDT", "")
                token_details[token] = {
                    "symbol": pair["symbol"],
                    "price_change_24h": pair["priceChangePercent"],
                    "volume_usdt": pair["quoteVolume"],
                    "last_price": pair["lastPrice"],
                    "trade_count": pair.get("count", 0),
                    "rank_by_volume": top_pairs.index(pair) + 1
                }

            logger.info(f"Prepared top 10 tokens for analysis: {selected_tokens}")

            return NodeResult(
                updates={
                    "selected_tokens": selected_tokens,
                    "token_details": token_details,
                    "selection_method": "top_15_by_volume_excluding_stablecoins",
                    "execution_history": {
                        "action": "prepare_token_list",
                        "tokens_count": len(selected_tokens),
                        "selection_criteria": "top_10_volume_no_stablecoins"
                    },
                    "analysis_flags": {"tokens_prepared"}
                },
                metadata={
                    "tokens_count": len(selected_tokens),
                    "selection_type": "volume_based",
                    "stablecoins_excluded": True
                },
                logs=[f"Prepared {len(selected_tokens)} tokens for analysis (stablecoins excluded)"],
                confidence=1.0,
                reasoning="Selected top 15 tokens based on volume ranking, stablecoins excluded"
            )

        except Exception as e:
            logger.error(f"Token list preparation failed: {e}")
            return NodeResult(
                error=f"Token list preparation failed: {str(e)}",
                confidence=0.0
            )

    def create_token_analyzer_by_index(self, token_index: int):
        """Create a per-index analyzer that selects the token dynamically from state."""
        async def analyzer(state: Dict[str, Any], context: NodeContext) -> NodeResult:
            selected_tokens: List[str] = state.get("selected_tokens", [])
            if token_index >= len(selected_tokens):
                return NodeResult(updates={}, confidence=1.0, reasoning=f"No token at index {token_index}")

            token = selected_tokens[token_index]
            # Reuse the per-token analyzer with token symbol
            return await self.create_token_analyzer(token)(state, context)

        return analyzer

    def create_token_analyzer(self, token: str):
        """Create analyzer coroutine for a specific token symbol."""
        async def token_analyzer(state: Dict[str, Any], context: NodeContext) -> NodeResult:
            """Analyze a single token end-to-end: kline + news + LLM recommendations."""
            logger.info(f"[{context.node_name}] Start analyzing token: {token}")
            start_time = datetime.now()

            try:
                token_details = state.get("token_details", {})

                # 1) Fetch kline data and news concurrently
                async def fetch_kline_data():
                    """Fetch kline technical data via PowerData toolkit."""
                    try:
                        if not self.powerdata_cex_tool:
                            return {"error": "PowerData tool not available", "data": None}

                        symbol = f"{token}/USDT"
                        indicators_config = {
                            "rsi": [{"timeperiod": 14}],
                            "ema": [{"timeperiod": 12}, {"timeperiod": 26}, {"timeperiod": 50}],
                            "macd": [{"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}],
                            "bbands": [{"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2}]
                        }

                        # Fetch both 1d and 4h timeframes
                        daily_result = await self.powerdata_cex_tool.execute(
                            exchange="binance",
                            symbol=symbol,
                            timeframe="1d",
                            limit=100,
                            indicators_config=json.dumps(indicators_config),
                            use_enhanced=True
                        )

                        h4_result = await self.powerdata_cex_tool.execute(
                            exchange="binance",
                            symbol=symbol,
                            timeframe="4h",
                            limit=100,
                            indicators_config=json.dumps(indicators_config),
                            use_enhanced=True
                        )

                        return {
                            "daily_data": daily_result.output if daily_result and not daily_result.error else None,
                            "h4_data": h4_result.output if h4_result and not h4_result.error else None,
                            "error": None
                        }

                    except Exception as e:
                        logger.error(f"Failed to fetch kline for {token}: {e}")
                        return {"error": str(e), "data": None}

                async def fetch_news_data():
                    """Fetch latest news via Tavily MCP for sentiment/context."""
                    try:
                        if not self.tavily_search_tool:
                            return {"error": "Tavily tool not available", "data": None}

                        search_query = f"{token} cryptocurrency news price analysis market trends"
                        result = await self.tavily_search_tool.execute(
                            query=search_query,
                            max_results=3,
                            search_depth="basic"
                        )

                        if hasattr(result, 'error') and result.error:
                            return {"error": result.error, "data": None}
                        elif hasattr(result, 'output'):
                            return {"data": result.output, "error": None}
                        else:
                            return {"data": str(result), "error": None}

                    except Exception as e:
                        logger.error(f"Failed to fetch news for {token}: {e}")
                        return {"error": str(e), "data": None}

                # Run both coroutines concurrently
                kline_task = fetch_kline_data()
                news_task = fetch_news_data()

                kline_result, news_result = await asyncio.gather(kline_task, news_task)

                # 2) Immediately run LLM analysis per token
                token_detail = token_details.get(token, {})
                current_price = token_detail.get('last_price', 0)
                price_change_24h = token_detail.get('price_change_24h', 0)

                # Build LLM prompts (EN) for clarity and consistency
                ta_section = json.dumps(kline_result, indent=2, ensure_ascii=False) if kline_result.get('daily_data') else 'TA unavailable'
                news_section = json.dumps(news_result, indent=2, ensure_ascii=False) if news_result.get('data') else 'News unavailable'

                ta_prompt = f"""You are a professional crypto trading analyst. Provide a concise, actionable TECHNICAL trading recommendation for {token}.

Context:
- Token: {token}
- Current Price: ${current_price}
- 24h Change: {price_change_24h:+.2f}%

Technical Data (1d and 4h):
{ta_section}

Requirements:
1) Summarize technical trend and momentum
2) Key support/resistance levels (prices)
3) Entry plan (zones or triggers)
4) Stop loss
5) Targets (multiple if applicable)
6) Risk rating (Low/Medium/High) and a 1-10 opportunity score
Output in English, concise, bullet style."""

                news_prompt = f"""You are a professional crypto news/sentiment analyst. Based on recent NEWS for {token}, provide a concise, actionable NEWS-DRIVEN trading recommendation.

Context:
- Token: {token}
- Current Price: ${current_price}

News Data (headlines, snippets, sources):
{news_section}

Requirements:
1) Summarize overall sentiment and key catalysts/risks
2) Potential impact on price (short-term vs mid-term)
3) Entry considerations based on news (if any)
4) Risk warnings linked to news uncertainty
5) Risk rating (Low/Medium/High) and a 1-10 opportunity score
Output in English, concise, bullet style."""

                ta_messages = [
                    Message(role="system", content="You are a professional crypto trading analyst focusing on technical analysis."),
                    Message(role="user", content=ta_prompt)
                ]
                news_messages = [
                    Message(role="system", content="You are a professional crypto analyst focusing on news and sentiment."),
                    Message(role="user", content=news_prompt)
                ]

                try:
                    ta_analysis = await self._call_llm(ta_messages)
                except Exception as e:
                    ta_analysis = f"TA LLM analysis failed: {str(e)}"

                try:
                    news_analysis = await self._call_llm(news_messages)
                except Exception as e:
                    news_analysis = f"News LLM analysis failed: {str(e)}"

                end_time = datetime.now()
                analysis_time = (end_time - start_time).total_seconds()

                # Store per-token analysis result back into state
                analysis_result = {
                    "token": token,
                    "current_price": current_price,
                    "price_change_24h": price_change_24h,
                    "kline_data": kline_result,
                    "news_data": news_result,
                    "llm_ta": ta_analysis,
                    "llm_news": news_analysis,
                    "analysis_time": analysis_time,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }

                logger.info(f"Finished {token} analysis in {analysis_time:.2f}s")

                return NodeResult(
                    updates={
                        f"token_analysis_{token.lower()}": analysis_result,
                        "execution_history": {
                            "action": f"analyze_{token.lower()}",
                            "token": token,
                            "analysis_time": analysis_time,
                            "status": "completed",
                            "parallel_execution": True
                        },
                        "analysis_flags": {f"{token.lower()}_analysis_completed"}
                    },
                    metadata={
                        "token": token,
                        "analysis_time": analysis_time,
                        "execution_type": "parallel_graph_node",
                        "kline_success": kline_result.get('error') is None,
                        "news_success": news_result.get('error') is None
                    },
                    logs=[f"Completed {token} end-to-end analysis"],
                    confidence=0.9 if (isinstance(ta_analysis, str) and "failed" not in ta_analysis.lower()) and (isinstance(news_analysis, str) and "failed" not in news_analysis.lower()) else 0.3,
                    reasoning=f"Parallel branch node completed kline + news + LLM for {token}"
                )

            except Exception as e:
                logger.error(f"Token {token} analysis failed: {e}")
                return NodeResult(
                    error=f"Token {token} analysis failed: {str(e)}",
                    confidence=0.0
                )

        return token_analyzer

    async def llm_final_aggregation(self, state: Dict[str, Any], context: NodeContext) -> NodeResult:
        """Aggregate all per-token recommendations and rank 3-5 best opportunities."""
        logger.info(f"[{context.node_name}] Aggregating all token analyses and ranking opportunities")

        try:
            # Collect all token analyses dynamically from state
            token_analyses = []
            for key, value in state.items():
                if isinstance(key, str) and key.startswith("token_analysis_") and isinstance(value, dict):
                    token_analyses.append(value)

            if not token_analyses:
                return NodeResult(
                    error="No token analyses available for aggregation",
                    confidence=0.0
                )

            # Compute parallel execution stats
            total_time = max([analysis.get('analysis_time', 0) for analysis in token_analyses])
            avg_time = sum([analysis.get('analysis_time', 0) for analysis in token_analyses]) / len(token_analyses)
            parallel_stats = {
                "total_tokens": len(token_analyses),
                "successful_count": len([a for a in token_analyses if a.get('status') == 'completed']),
                "total_execution_time": total_time,
                "average_time_per_token": avg_time,
                "parallel_efficiency": "Graph-level true parallel execution"
            }

            # Build final aggregation prompt
            aggregation_prompt = f"""Based on the following {len(token_analyses)} token analyses, produce a final summary:

## Execution Stats:
- Tokens analyzed: {parallel_stats.get('total_tokens', 0)}
- Success count: {parallel_stats.get('successful_count', 0)}
- Total execution time: {parallel_stats.get('total_execution_time', 0):.2f}s
- Execution mode: {parallel_stats.get('parallel_efficiency', 'parallel')}

## Per-Token Analyses:
"""

            # Append each token analysis details to the prompt
            for i, analysis in enumerate(token_analyses, 1):
                token = analysis.get('token', 'Unknown')
                price = analysis.get('current_price', 0)
                change = analysis.get('price_change_24h', 0)
                llm_ta = analysis.get('llm_ta', 'No TA analysis')
                llm_news = analysis.get('llm_news', 'No news analysis')
                analysis_time = analysis.get('analysis_time', 0)

                aggregation_prompt += f"""
### {i}. {token}
- Current Price: ${price}
- 24h Change: {change:+.2f}%
- Analysis Time: {analysis_time:.2f}s

**Technical Recommendation (LLM):**
{llm_ta}

**News Recommendation (LLM):**
{llm_news}

---
"""

            aggregation_prompt += """
## Your Tasks:
1) Rank the top 3-5 trading opportunities by risk-adjusted potential
   - For each: token, current price, opportunity score, key reasons
   - Entry plan, stop loss, targets, timeframe, risk level
2) Market overview: trend, sentiment, and rotations
3) Risk warnings: key risks to monitor
4) Portfolio suggestion: weighting and risk management tips

Keep the output in English, concise and directly actionable.
"""

            messages = [
                Message(role="system", content=(
                    "You are a senior crypto investment advisor. Combine all token analyses to identify the best opportunities, "
                    "rank them by risk-adjusted potential, and provide clear, actionable plans along with risk management."
                )),
                Message(role="user", content=aggregation_prompt)
            ]

            # Call LLM for final aggregation
            try:
                final_recommendation = await self._call_llm(messages)
            except Exception as e:
                final_recommendation = f"Final aggregation analysis failed: {str(e)}"

            # Compute overall confidence
            success_rate = parallel_stats.get('successful_count', 0) / max(1, parallel_stats.get('total_tokens', 1))
            final_confidence = min(0.95, 0.6 + success_rate * 0.3)

            return NodeResult(
                updates={
                    "final_recommendation": final_recommendation,
                    "trading_advice": final_recommendation,  # Compatible with original display logic
                    "aggregation_stats": {
                        "tokens_analyzed": len(token_analyses),
                        "successful_analyses": parallel_stats.get('successful_count', 0),
                        "total_execution_time": parallel_stats.get('total_execution_time', 0),
                        "average_analysis_time": parallel_stats.get('average_time_per_token', 0),
                        "parallel_efficiency": "True parallel execution per token analysis"
                    },
                    "confidence_score": final_confidence,
                    "risk_level": "MEDIUM",  # Can be dynamically adjusted based on analysis results
                    "execution_history": {
                        "action": "llm_final_aggregation",
                        "tokens_processed": len(token_analyses),
                        "aggregation_method": "ranking_by_opportunity",
                        "llm_used": True
                    },
                    "analysis_flags": {"final_aggregation_completed"}
                },
                metadata={
                    "aggregation_type": "opportunity_ranking",
                    "tokens_included": [analysis.get('token') for analysis in token_analyses],
                    "recommendation_length": len(final_recommendation),
                    "success_rate": success_rate,
                    "execution_efficiency": "parallel_per_token_analysis"
                },
                logs=[f"Aggregated {len(token_analyses)} token analyses and ranked opportunities"],
                confidence=final_confidence,
                reasoning=f"Final aggregation over {len(token_analyses)} tokens with ranking"
            )

        except Exception as e:
            logger.error(f"Final aggregation analysis failed: {e}")
            return NodeResult(
                error=f"Final aggregation analysis failed: {str(e)}",
                confidence=0.0
            )


async def run_llm_driven_analysis():
    """Run LLM-driven real cryptocurrency analysis"""
    logger.info("🚀 Starting LLM-driven real cryptocurrency analysis system")

    try:
        # Create analyzer
        analyzer = LLMCryptoAnalyzer()

        # Create analysis graph
        graph = analyzer.create_analysis_graph()

        # Compile graph
        compiled = graph.compile()

        # Initial state
        initial_state = {
            "query": "LLM-driven cryptocurrency analysis based on real Binance API data",
            "binance_market_data": {},
            "selected_tokens": [],
            "token_details": {},
            "market_data": {},
            "news_data": {},
            "comprehensive_analysis": {},
            "technical_analysis": {},
            "execution_history": [],
            "analysis_flags": set(),
            "final_recommendation": "",
            "confidence_score": 0.0,
            "risk_level": "MEDIUM"
        }

        logger.info("📊 Starting LLM-driven real data analysis workflow...")
        start_time = datetime.now()

        # Execute analysis
        result = await compiled.invoke(
            initial_state,
            config={"configurable": {"thread_id": "llm_crypto_analysis"}}
        )

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Display results
        print("\n" + "="*80)
        print("🔍 LLM-Driven Real Cryptocurrency Analysis Results")
        print("="*80)

        print(f"\n⏱️ Total Execution Time: {execution_time:.2f} seconds")
        print(f"🔧 Data Sources: Binance Official API + PowerData + LLM Analysis")

        # Display Binance data overview
        binance_data = result.get('binance_market_data', {})
        if binance_data:
            print(f"\n📊 Binance Real Data:")
            print(f"  • Total Available Trading Pairs: {binance_data.get('total_pairs_available', 0)}")
            print(f"  • Selected Active Pairs: {binance_data.get('selected_pairs_count', 0)}")
            print(f"  • Data Collection Time: {binance_data.get('timestamp', 'N/A')[:19]}")

            top_pairs = binance_data.get('top_pairs', [])[:5]
            if top_pairs:
                print(f"  • Top 5 Most Active Trading Pairs:")
                for pair in top_pairs:
                    print(f"    - {pair['symbol']}: {pair['priceChangePercent']:+.2f}% (Volume: {pair['quoteVolume']:,.0f} USDT)")

        # Show PowerData analysis results (simplified)
        market_data = result.get('market_data', {})
        if market_data:
            print(f"\n📈 PowerData Technical Analysis:")
            successful_count = sum(1 for data in market_data.values() if data.get('status') == 'success')
            print(f"  • Successfully analyzed {successful_count}/{len(market_data)} tokens")

        # Show news analysis (simplified)
        news_data = result.get('news_data', {})
        if news_data:
            print(f"\n📰 News Analysis:")
            analysis_type = news_data.get('analysis_type', 'unknown')
            if analysis_type == 'llm_generated':
                news_analysis = news_data.get('data', {})
                market_analysis = news_analysis.get('market_news_analysis', {})
                if market_analysis:
                    sentiment = market_analysis.get('overall_sentiment', 'neutral').upper()
                    print(f"  • Overall Sentiment: {sentiment}")
            else:
                print(f"  • News Search Status: {news_data.get('news_search_status', 'unknown')}")

        # Show LLM comprehensive analysis (simplified)
        trend_analysis = result.get('comprehensive_analysis', {})
        if trend_analysis:
            print(f"\n🤖 LLM Comprehensive Analysis:")
            analysis_result = trend_analysis.get('analysis_result', {})
            comprehensive = analysis_result.get('comprehensive_analysis', {})

            if comprehensive:
                overall_trend = comprehensive.get('overall_market_trend', 'N/A')
                market_sentiment = comprehensive.get('market_sentiment', 'N/A')
                risk_level = comprehensive.get('risk_level', 'N/A')
                print(f"  • Overall Trend: {overall_trend.upper()}")
                print(f"  • Market Sentiment: {market_sentiment.upper()}")
                print(f"  • Risk Level: {risk_level.upper()}")

                # Show top opportunities
                top_opportunities = comprehensive.get('top_opportunities', [])
                if top_opportunities:
                    print(f"  • Top Opportunities: {', '.join(top_opportunities[:3])}")

        # Show direct LLM trading advice
        trading_advice = result.get('trading_advice', '')
        confidence_score = result.get('confidence_score', 0)
        risk_level = result.get('risk_level', 'N/A')

        print(f"\n🏆 LLM Trading Recommendations:")
        print(f"  • Overall Confidence: {confidence_score:.1%}")
        print(f"  • Risk Level: {risk_level}")
        print("="*80)

        if trading_advice and trading_advice.strip():
            print(f"\n📈 Professional Trading Advice:")
            print("="*80)
            # Display the raw LLM trading advice with proper formatting
            print(trading_advice)
            print("="*80)
        else:
            print("   ❌ Unable to generate trading recommendations")

        return result

    except Exception as e:
        logger.error(f"❌ LLM-driven analysis failed: {e}")
        print(f"\n💥 Error encountered during analysis: {e}")
        print("Possible causes:")
        print("  • Binance API access issues")
        print("  • PowerData tool configuration issues")
        print("  • LLM service unavailable")
        print("  • Network connection issues")
        raise


if __name__ == "__main__":
    # Run LLM-driven real cryptocurrency analysis
    asyncio.run(run_llm_driven_analysis())