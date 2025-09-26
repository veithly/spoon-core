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
    merge_dicts,
    append_history,
    union_sets,
    validate_range,
    validate_enum,
    END,
)
import operator

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
        self.llm_manager = get_llm_manager()
        self.crypto_tools = get_crypto_tools()
        self.tool_manager = ToolManager(self.crypto_tools)

        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if tavily_key:
            try:
                from spoon_ai.tools.mcp_tool import MCPTool
                self.tavily_search_tool = MCPTool(
                    name="tavily-search",
                    description="Performs a web search using the Tavily API for cryptocurrency news and market sentiment.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query for cryptocurrency news and analysis"},
                            "max_results": {"type": "integer", "description": "Maximum number of search results to return", "default": 5},
                            "search_depth": {"type": "string", "description": "Search depth: basic or advanced", "default": "basic"}
                        },
                        "required": ["query"]
                    },
                    mcp_config={
                        "command": "npx",
                        "args": ["--yes", "tavily-mcp"],
                        "env": {"TAVILY_API_KEY": tavily_key}
                    }
                )
            except:
                self.tavily_search_tool = None
        else:
            self.tavily_search_tool = None


        try:
            from spoon_toolkits.crypto.crypto_powerdata.tools import CryptoPowerDataCEXTool
            self.powerdata_cex_tool = CryptoPowerDataCEXTool()
        except:
            self.powerdata_cex_tool = None

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
        token_nodes: List[str] = []
        for index in range(10):
            node_name = f"analyze_token_{index}"
            if analyzer_entry_node is None:
                analyzer_entry_node = node_name
            graph.add_node(
                node_name,
                self._wrap_method(self.create_token_analyzer_by_index(index)),
            )
            token_nodes.append(node_name)

        # Register the parallel execution group for token analysis
        if token_nodes:
            graph.add_parallel_group(
                "token_analysis",
                token_nodes,
                {"join_strategy": "all_complete", "error_strategy": "ignore_errors"},
            )

        # Final aggregation and ranking
        graph.add_node("llm_final_aggregation", self._wrap_method(self.llm_final_aggregation))

        # Sequential preparation flow
        graph.add_edge("fetch_binance_data", "prepare_token_list")

        # Connect prepare_token_list to each analyzer to execute in parallel
        if token_nodes:
            for n in token_nodes:
                graph.add_edge("prepare_token_list", n)
            # After any analyzer completes, allow flow to final aggregation
            # Connect all analyzers to final aggregation; duplicates are harmless
            for n in token_nodes:
                graph.add_edge(n, "llm_final_aggregation")

        # Final aggregation
        graph.add_edge("llm_final_aggregation", END)

        # Set entry point and monitoring
        graph.set_entry_point("fetch_binance_data")
        # Optional monitoring (no-op if engine doesn't implement it)
        if hasattr(graph, "enable_monitoring"):
            graph.enable_monitoring([
                "execution_time",
                "llm_response_quality",
                "api_success_rate",
                "analysis_depth",
                "parallel_branch_efficiency"
            ])

        return graph

    def _wrap_method(self, method):
        """Wrap instance method into a coroutine accepting only (state), constructing a default NodeContext.
        Converts NodeResult to Command/update dict for the minimal engine.
        """
        async def wrapper(state):
            node_name = getattr(method, '__name__', 'node')
            ctx = NodeContext(node_name=node_name, iteration=0, thread_id="example")
            result = await method(state, ctx)
            # Normalize return for engine: support NodeResult by mapping to Command/update dict
            if isinstance(result, NodeResult):
                if result.error:
                    raise RuntimeError(result.error)
                # Always return plain updates dict for invoke() path
                return result.updates or {}
            return result
        return wrapper

    async def _call_llm(self, messages: List[Message], provider: str = None) -> str:
        response = await self.llm_manager.chat(messages, provider=provider)
        return response.content

    async def fetch_binance_market_data(self, state: Dict[str, Any], context: NodeContext) -> NodeResult:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            async with session.get(url) as response:
                if response.status != 200:
                    return NodeResult(error=f"Binance API failed: {response.status}", confidence=0.0)
                binance_data = await response.json()

        stablecoins = {'USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'BUSDUSDT', 'DAIUSDT', 'USDPUSDT', 'FRAXUSDT', 'LUSDUSDT', 'SUSDUSDT', 'USTCUSDT', 'USDDUSDT', 'GUSDUSDT', 'PAXGUSDT', 'USTUSDT'}

        usdt_pairs = []
        for item in binance_data:
            if isinstance(item, dict) and item.get('symbol', '').endswith('USDT'):
                symbol = item.get('symbol', '')
                if symbol not in stablecoins and all(key in item for key in ['symbol', 'priceChangePercent', 'volume', 'lastPrice']):
                    usdt_pairs.append({
                        'symbol': symbol,
                        'priceChangePercent': float(item['priceChangePercent']),
                        'volume': float(item['volume']),
                        'lastPrice': float(item['lastPrice']),
                        'count': int(item.get('count', 0)),
                        'quoteVolume': float(item.get('quoteVolume', 0))
                    })

        top_pairs_by_volume = sorted(usdt_pairs, key=lambda x: x['quoteVolume'], reverse=True)[:10]

        return NodeResult(
            updates={
                "binance_market_data": {
                    "top_pairs": top_pairs_by_volume,
                    "total_pairs_available": len(usdt_pairs),
                    "selected_pairs_count": len(top_pairs_by_volume),
                    "timestamp": datetime.now().isoformat(),
                    "source": "binance_api_real"
                },
                "execution_history": {
                    "action": "fetch_binance_data",
                    "pairs_fetched": len(top_pairs_by_volume),
                    "real_data": True
                },
                "analysis_flags": {"binance_data_fetched"}
            },
            confidence=0.95
        )

    async def prepare_token_list(self, state: Dict[str, Any], context: NodeContext) -> NodeResult:
        binance_data = state.get("binance_market_data", {})
        top_pairs = binance_data.get("top_pairs", [])

        if not top_pairs:
            return NodeResult(error="No Binance market data available", confidence=0.0)

        selected_tokens = [pair["symbol"].replace("USDT", "") for pair in top_pairs]
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

        return NodeResult(
            updates={
                "selected_tokens": selected_tokens,
                "token_details": token_details,
                "execution_history": {
                    "action": "prepare_token_list",
                    "tokens_count": len(selected_tokens)
                },
                "analysis_flags": {"tokens_prepared"}
            },
            confidence=1.0
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

                    except Exception:
                        return {"error": "kline fetch failed", "data": None}

                async def fetch_news_data():
                    if self.tavily_search_tool:
                        result = await self.tavily_search_tool.execute(
                            query=f"{token} cryptocurrency news price analysis market trends",
                            max_results=3,
                            search_depth="basic"
                        )
                        return {"data": str(result), "error": None}
                    return {"error": "no tavily mcp tool", "data": None}

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

                ta_analysis = await self._call_llm(ta_messages)
                news_analysis = await self._call_llm(news_messages)

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
        final_recommendation = await self._call_llm(messages)

        # Compute overall confidence
        success_rate = parallel_stats.get('successful_count', 0) / max(1, parallel_stats.get('total_tokens', 1))
        final_confidence = min(0.95, 0.6 + success_rate * 0.3)

        return NodeResult(
            updates={
                "final_recommendation": final_recommendation,
                "trading_advice": final_recommendation,
                "aggregation_stats": {
                    "tokens_analyzed": len(token_analyses),
                    "successful_analyses": parallel_stats.get('successful_count', 0),
                    "total_execution_time": parallel_stats.get('total_execution_time', 0),
                    "average_analysis_time": parallel_stats.get('average_time_per_token', 0)
                },
                "confidence_score": final_confidence,
                "risk_level": "MEDIUM",
                "execution_history": {
                    "action": "llm_final_aggregation",
                    "tokens_processed": len(token_analyses)
                },
                "analysis_flags": {"final_aggregation_completed"}
            },
            confidence=final_confidence
        )


async def run_llm_driven_analysis():
    analyzer = LLMCryptoAnalyzer()
    graph = analyzer.create_analysis_graph()
    compiled = graph.compile()

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

    start_time = datetime.now()
    result = await compiled.invoke(
        initial_state,
        config={"configurable": {"thread_id": "llm_crypto_analysis"}}
    )

    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    trading_advice = result.get('trading_advice', '')
    if trading_advice:
        print(trading_advice)

    return result


if __name__ == "__main__":
    # Run LLM-driven real cryptocurrency analysis
    asyncio.run(run_llm_driven_analysis())