"""
SpoonOS Advanced Graph Demo - Parallel Processing & Intelligent Routing (Simplified)

Highlights:
- Intelligent query routing: general_qa, short_term_trend, macro_trend, deep_research
- TRUE parallel data fetching (15m, 30m, 1h, 4h, daily, weekly)
- LLM-powered routing and summarization
- Real market data via PowerData toolkit

Simplified design:
- Removed external web/news search and reflection loops
- Centralized symbol extraction (no hard-coded valid symbols)
- Reduced prints and redundant try/except checks
"""

import asyncio
import json
from typing import Dict, Any, List, Annotated, TypedDict, Optional
from datetime import datetime
from dotenv import load_dotenv

from spoon_ai.graph import (
    StateGraph, END
)

# Import PowerData tool for real data integration (simplified)
from spoon_toolkits.crypto.crypto_powerdata.tools import CryptoPowerDataCEXTool
powerdata_tool = CryptoPowerDataCEXTool()

from spoon_ai.llm.manager import get_llm_manager
from spoon_ai.schema import Message
from spoon_ai.tools.mcp_tool import MCPTool
from spoon_toolkits.crypto.evm import EvmSwapTool

# Load environment variables
load_dotenv()

# Initialize Tavily MCP tool (simplified, like graph_crypto_analysis)
import os
import subprocess
tavily_search_tool = None
try:
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key and "your-tavily-api-key-here" not in tavily_key:
        cmd = None
        try:
            r = subprocess.run(["npx", "--version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                cmd = "npx"
        except Exception:
            pass
        if cmd is None and os.name == "nt":
            try:
                r = subprocess.run(["npx.cmd", "--version"], capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    cmd = "npx.cmd"
            except Exception:
                pass
        if cmd:
            tavily_search_tool = MCPTool(
                name="tavily-search",
                description="Performs a web search using the Tavily API for cryptocurrency news and market sentiment.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query for cryptocurrency news and analysis"},
                        "max_results": {"type": "integer", "description": "Maximum results", "default": 5},
                        "search_depth": {"type": "string", "description": "basic or advanced", "default": "basic"}
                    },
                    "required": ["query"]
                },
                mcp_config={
                    "command": cmd,
                    "args": ["--yes", "tavily-mcp"],
                    "env": {"TAVILY_API_KEY": tavily_key},
                    "timeout": 30,
                    "max_retries": 2
                }
            )
except Exception:
    tavily_search_tool = None


def _extract_indicators(data: Any) -> Dict[str, Any]:
    """Safely extract indicator-like fields from PowerData result.
    Kept minimal to avoid errors; returns empty dict on unknown shapes.
    """
    indicators: Dict[str, Any] = {}
    try:
        if isinstance(data, dict):
            for key, value in data.items():
                key_l = str(key).lower()
                if any(k in key_l for k in ("rsi", "ema", "macd", "bbands", "stoch", "adx", "cci")):
                    indicators[key] = value
        elif isinstance(data, list) and data:
            last = data[-1]
            if isinstance(last, dict):
                for key, value in last.items():
                    key_l = str(key).lower()
                    if any(k in key_l for k in ("rsi", "ema", "macd", "bbands", "stoch", "adx", "cci")):
                        indicators[key] = value
    except Exception:
        return {}
    return indicators


async def _fetch_kline_data(symbol: str, timeframe: str, limit: int, indicators_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generic PowerData fetch returning a normalized timeframe payload."""
    symbol_pair = f"{symbol}/USDT"
    result = await powerdata_tool.execute(
        exchange="binance",
        symbol=symbol_pair,
        timeframe=timeframe,
        limit=limit,
        indicators_config=json.dumps(indicators_config),
        use_enhanced=True,
    )
    return {
        "symbol": symbol_pair,
        "timeframe": timeframe,
        "data": result.output,
        "indicators": _extract_indicators(result.output),
        "timestamp": datetime.now().isoformat(),
    }




# ==========================================================================
# CONFIG PRESETS AND NODE FACTORY HELPERS
# ==========================================================================

# Indicator config presets for short-term and macro analyses
SHORT_TERM_PRESETS: Dict[str, Dict[str, Any]] = {
    "15m": {
        "limit": 50,
        "indicators": {
            "rsi": [{"timeperiod": 14}],
            "ema": [{"timeperiod": 12}, {"timeperiod": 26}],
            "macd": [{"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}],
            "bbands": [{"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2}],
        },
    },
    "30m": {
        "limit": 40,
        "indicators": {
            "rsi": [{"timeperiod": 14}],
            "ema": [{"timeperiod": 12}, {"timeperiod": 26}],
            "macd": [{"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}],
            "stoch": [{"fastkperiod": 14, "slowkperiod": 3, "slowdperiod": 3}],
        },
    },
    "1h": {
        "limit": 30,
        "indicators": {
            "rsi": [{"timeperiod": 14}],
            "ema": [{"timeperiod": 12}, {"timeperiod": 26}],
            "macd": [{"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}],
            "adx": [{"timeperiod": 14}],
            "cci": [{"timeperiod": 20}],
        },
    },
}

MACRO_PRESETS: Dict[str, Dict[str, Any]] = {
    "4h": {
        "limit": 30,
        "indicators": {
            "rsi": [{"timeperiod": 14}],
            "ema": [{"timeperiod": 50}, {"timeperiod": 200}],
            "macd": [{"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}],
            "adx": [{"timeperiod": 14}],
        },
    },
    "1d": {
        "limit": 30,
        "indicators": {
            "rsi": [{"timeperiod": 14}],
            "ema": [{"timeperiod": 50}, {"timeperiod": 200}],
            "macd": [{"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}],
            "adx": [{"timeperiod": 14}],
            "bbands": [{"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2}],
        },
    },
    "1w": {
        "limit": 20,
        "indicators": {
            "rsi": [{"timeperiod": 14}],
            "ema": [{"timeperiod": 50}, {"timeperiod": 200}],
            "macd": [{"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}],
            "adx": [{"timeperiod": 14}],
            "stoch": [{"fastkperiod": 14, "slowkperiod": 3, "slowdperiod": 3}],
        },
    },
}


def make_fetch_node(timeframe: str, limit: int, indicators_config: Dict[str, Any], result_key: str):
    async def _node(state: AdvancedComprehensiveState) -> Dict[str, Any]:
        symbol = state.get("symbol", "BTC")
        payload = await _fetch_kline_data(symbol, timeframe, limit, indicators_config)
        return {
            result_key: payload,
            "parallel_tasks_completed": state.get("parallel_tasks_completed", 0) + 1,
            "execution_log": state.get("execution_log", []) + [f"{timeframe} data fetched for {payload['symbol']}"],
        }
    return _node


# ==========================================================================
# LIGHTWEIGHT GRAPH MEMORY (JSON persistence per user)
# ==========================================================================

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "advanced_graph_memory.json")


def _read_json_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json_file(path: str, data: Dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _user_key(user_name: str) -> str:
    return (user_name or "User").strip().lower()


async def load_memory_node(state: "AdvancedComprehensiveState") -> Dict[str, Any]:
    user_name = state.get("user_name", "User")
    all_mem = _read_json_file(MEMORY_FILE)
    ukey = _user_key(user_name)
    user_mem = all_mem.get(ukey, {})
    memory_state: MemoryState = {
        "session_id": state.get("session_id", ""),
        "conversation_history": user_mem.get("conversation_history", [])[-50:],
        "user_context": user_mem.get("user_context", {}),
        "learned_patterns": user_mem.get("learned_patterns", {}),
    }
    return {
        "memory_state": memory_state,
        "execution_log": state.get("execution_log", []) + [f"Memory loaded for {user_name}"],
        "current_step": "memory_loaded",
    }


async def update_memory_node(state: "AdvancedComprehensiveState") -> Dict[str, Any]:
    user_name = state.get("user_name", "User")
    user_query = state.get("user_query", "")
    symbol = state.get("symbol", "")
    qa = state.get("query_analysis", {}) or {}
    query_type = qa.get("query_type", "general_qa")
    final_output = state.get("final_output", "")

    all_mem = _read_json_file(MEMORY_FILE)
    ukey = _user_key(user_name)
    user_mem = all_mem.get(ukey, {"conversation_history": [], "user_context": {}, "learned_patterns": {}})

    # Update conversation history
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": user_query,
        "query_type": query_type,
        "symbol": symbol,
    }
    user_mem.setdefault("conversation_history", []).append(entry)
    # Trim
    user_mem["conversation_history"] = user_mem["conversation_history"][-200:]

    # Update learned patterns
    lp = user_mem.setdefault("learned_patterns", {})
    qtc = lp.setdefault("query_type_counts", {})
    qtc[query_type] = int(qtc.get(query_type, 0)) + 1
    sc = lp.setdefault("symbol_counts", {})
    if symbol:
        sc[symbol] = int(sc.get(symbol, 0)) + 1

    # Update user context
    uc = user_mem.setdefault("user_context", {})
    uc["last_summary"] = final_output[:800]
    if symbol:
        per_symbol = uc.setdefault("last_symbol_summaries", {})
        per_symbol[symbol] = final_output[:800]

    all_mem[ukey] = user_mem
    _write_json_file(MEMORY_FILE, all_mem)

    memory_state: MemoryState = {
        "session_id": state.get("session_id", ""),
        "conversation_history": user_mem["conversation_history"],
        "user_context": uc,
        "learned_patterns": lp,
    }
    return {
        "memory_state": memory_state,
        "execution_log": state.get("execution_log", []) + ["Memory updated"],
        "current_step": "memory_updated",
    }

# ============================================================================
# STATE SCHEMAS
# ============================================================================

class QueryAnalysisState(TypedDict):
    """State for query analysis and routing"""
    user_query: str
    query_type: str  # 'general_qa', 'short_term_trend', 'macro_trend', 'deep_research'
    confidence: float
    analysis_metadata: Annotated[Dict[str, Any], dict]


class ShortTermTrendState(TypedDict):
    """State for short-term trend analysis (15m/30m/1h parallel processing)"""
    user_query: str
    symbol: str
    timeframe_data: Annotated[Dict[str, Any], dict]  # {'15m': data, '30m': data, '1h': data}
    analysis_summary: str


class MacroTrendState(TypedDict):
    """State for macro trend analysis (4h/daily/weekly + news)"""
    user_query: str
    symbol: str
    timeframe_data: Annotated[Dict[str, Any], dict]
    news_data: Annotated[List[Dict[str, Any]], list]
    macro_analysis: str


class DeepResearchState(TypedDict):
    """Simplified deep research state"""
    user_query: str
    final_insights: str


class MemoryState(TypedDict):
    """State for memory management"""
    session_id: str
    conversation_history: Annotated[List[Dict[str, str]], list]
    user_context: Annotated[Dict[str, Any], dict]
    learned_patterns: Annotated[Dict[str, Any], dict]


class AdvancedComprehensiveState(TypedDict):
    """Advanced comprehensive state supporting all query types"""
    # Core query information
    user_query: str
    user_name: str
    session_id: str
    symbol: str
    trade_params: Annotated[Optional[Dict[str, Any]], None]  # Trading parameters extracted from query

    # Query analysis
    query_analysis: Annotated[Optional[QueryAnalysisState], None]

    # Analysis states (only one will be populated based on query type)
    short_term_trend: Annotated[Optional[ShortTermTrendState], None]
    macro_trend: Annotated[Optional[MacroTrendState], None]
    deep_research: Annotated[Optional[DeepResearchState], None]

    # Memory and context
    memory_state: Annotated[Optional[MemoryState], None]

    # Execution tracking
    execution_log: Annotated[List[str], list]
    routing_decisions: Annotated[List[str], list]
    current_step: str
    final_output: str

    # Trading signals
    trade_signal: str  # "BUY" or "NO_BUY"

    # Performance metrics
    processing_time: float
    parallel_tasks_completed: int


# ============================================================================
# WORKFLOW NODES
# ============================================================================

async def initialize_session_node(state: AdvancedComprehensiveState) -> Dict[str, Any]:
    """Initialize session and memory state"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return {
        "session_id": session_id,
        "current_step": "session_initialized",
        "execution_log": ["Session initialized"],
        "processing_time": 0.0,
        "parallel_tasks_completed": 0
    }


async def analyze_query_intent_node(state: AdvancedComprehensiveState) -> Dict[str, Any]:
    """Analyze user query and determine query type using LLM"""
    query = state.get("user_query", "")
    llm_manager = get_llm_manager()

    intent_prompt = f"""
    Analyze this cryptocurrency-related query and classify it into EXACTLY ONE category:

    Query: "{query}"

    Categories:
    1. general_qa - General questions about crypto concepts, how things work, basic info
       Examples: "What is blockchain?", "How does DeFi work?"

    2. short_term_trend - Questions about immediate/short-term price movements and trends
       Examples: "What's ETH doing now?", "Short-term BTC momentum", "Next hour trend"

    3. macro_trend - Questions about long-term trends, analysis, forecasts, weekly/monthly outlook
       Examples: "Long-term trends for ETH", "Macro analysis of BTC", "Weekly/monthly outlook"

    4. deep_research - Complex research requiring multiple sources and deep analysis
       Examples: "Research DeFi protocols", "Analyze market developments", "Investigate new projects"

    CLASSIFICATION RULES:
    - If query mentions "long-term", "macro", "trend analysis", "forecast", "weekly", "monthly" → macro_trend
    - If query mentions "short-term", "now", "today", "momentum", "next hours" → short_term_trend
    - If query asks to "research", "analyze thoroughly", "investigate" → deep_research
    - If query is general question about crypto concepts → general_qa

    Return ONLY: category|confidence
    Example: macro_trend|0.95
    """

    messages = [Message(role="user", content=intent_prompt)]
    response = await llm_manager.chat(messages)
    result = response.content.strip()

    # Parse result
    try:
        category, confidence_str = result.split("|")
        confidence = float(confidence_str)
    except:
        category = "general_qa"
        confidence = 0.5

    # Ensure valid category
    valid_categories = ["general_qa", "short_term_trend", "macro_trend", "deep_research"]
    if category not in valid_categories:
        category = "general_qa"

    # Debug logging
    print(f"DEBUG: Query classification - Category: {category}, Confidence: {confidence}, Raw LLM response: '{result}'")

    return {
        "query_analysis": {
            "user_query": query,
            "query_type": category,
            "confidence": confidence,
            "analysis_metadata": {
                "llm_response": result,
                "analysis_timestamp": datetime.now().isoformat()
            }
        },
        "current_step": f"intent_analyzed_{category}",
        "execution_log": state.get("execution_log", []) + [f"Query classified as: {category} (confidence: {confidence:.2f})"]
    }


async def general_qa_node(state: AdvancedComprehensiveState) -> Dict[str, Any]:
    """Handle general Q&A queries directly with LLM"""
    query = state.get("user_query", "")
    llm_manager = get_llm_manager()

    qa_prompt = f"""
    You are a helpful cryptocurrency assistant. Answer this general question about cryptocurrency:

    Question: {query}

    Provide a clear, concise, and helpful response. If this involves financial advice,
    remind the user that this is not financial advice and they should do their own research.
    """

    messages = [Message(role="user", content=qa_prompt)]
    response = await llm_manager.chat(messages)

    return {
        "final_output": response.content.strip(),
        "current_step": "general_qa_completed",
        "execution_log": state.get("execution_log", []) + ["General Q&A response generated"]
    }


async def extract_symbol_node(state: AdvancedComprehensiveState) -> Dict[str, Any]:
    """Extract trading parameters from user query for analysis and potential execution"""
    query = state.get("user_query", "")
    llm_manager = get_llm_manager()

    # Extract symbol for analysis (default to ETH) and trading parameters
    extraction_prompt = f"""
    Analyze this crypto-related query and extract:

    1. ANALYSIS_SYMBOL: The primary cryptocurrency to analyze for trend (e.g., BTC, ETH, SOL)

    2. TRADE_PARAMS: If trading/buying/swapping is mentioned, extract in JSON format:
       {{
         "from_token": "token to sell (address like 0x... or symbol like USDC)",
         "to_token": "token to buy (address like 0x... or symbol like ETH)",
         "amount": "amount as string (e.g., '0.1', '100')"
       }}

    Look for phrases like:
    - "swap X for Y"
    - "buy X amount of Y"
    - "trade X to Y"
    - "if bullish, swap/buy..."

    Query: "{query}"

    Return format:
    ANALYSIS_SYMBOL: [SYMBOL]
    TRADE_PARAMS: [JSON or "NONE"]
    """

    messages = [Message(role="user", content=extraction_prompt)]
    response = await llm_manager.chat(messages)
    result = response.content.strip()

    # Parse results
    analysis_symbol = "ETH"  # Default to ETH
    trade_params = None

    for line in result.split('\n'):
        line = line.strip()
        if line.startswith('ANALYSIS_SYMBOL:'):
            symbol_part = line.replace('ANALYSIS_SYMBOL:', '').strip()
            if symbol_part and len(symbol_part) <= 10:  # Reasonable symbol length
                analysis_symbol = symbol_part.upper()
        elif line.startswith('TRADE_PARAMS:'):
            params_part = line.replace('TRADE_PARAMS:', '').strip()
            if params_part != "NONE" and params_part.startswith('{'):
                try:
                    trade_params = json.loads(params_part)
                except:
                    trade_params = None

    return {
        "symbol": analysis_symbol,
        "trade_params": trade_params,
        "current_step": f"symbol_extracted_{analysis_symbol}",
        "execution_log": state.get("execution_log", []) + [
            f"Analysis symbol: {analysis_symbol}",
            f"Trade params: {trade_params if trade_params else 'None'}"
        ]
    }


# ============================================================================
# SHORT-TERM TREND ANALYSIS NODES (PARALLEL PROCESSING)
# ============================================================================

fetch_15m_data_node = make_fetch_node(
    "15m",
    SHORT_TERM_PRESETS["15m"]["limit"],
    SHORT_TERM_PRESETS["15m"]["indicators"],
    "timeframe_data_15m",
)


fetch_30m_data_node = make_fetch_node(
    "30m",
    SHORT_TERM_PRESETS["30m"]["limit"],
    SHORT_TERM_PRESETS["30m"]["indicators"],
    "timeframe_data_30m",
)


fetch_1h_data_node = make_fetch_node(
    "1h",
    SHORT_TERM_PRESETS["1h"]["limit"],
    SHORT_TERM_PRESETS["1h"]["indicators"],
    "timeframe_data_1h",
)



# preview function removed by simplification


async def analyze_short_term_trend_node(state: AdvancedComprehensiveState) -> Dict[str, Any]:
    """Analyze short-term trend from parallel data"""
    # Combine data from all timeframes
    timeframe_data = {}
    indicators = {}

    # Collect data from parallel nodes
    if "timeframe_data_15m" in state:
        timeframe_data["15m"] = state["timeframe_data_15m"]
        indicators.update(state["timeframe_data_15m"].get("indicators", {}))

    if "timeframe_data_30m" in state:
        timeframe_data["30m"] = state["timeframe_data_30m"]
        indicators.update(state["timeframe_data_30m"].get("indicators", {}))

    if "timeframe_data_1h" in state:
        timeframe_data["1h"] = state["timeframe_data_1h"]
        indicators.update(state["timeframe_data_1h"].get("indicators", {}))

    llm_manager = get_llm_manager()

    analysis_prompt = f"""
    You are a market analyst. Analyze short-term cryptocurrency trend using the raw multi-timeframe K-line data (candles). Write a concise, human-readable summary WITHOUT any JSON. Include:
    - A clear conclusion (bullish/neutral/bearish) for the next few hours
    - 2-4 key reasons grounded in price action across 15m/30m/1h (e.g., higher highs/lows, momentum, volatility, ranges, breakouts, reactions to levels)
    - Notable support/resistance or key levels if inferable
    - A brief risk note

    User Query: {state.get("user_query", "")}
    Timeframe Data: {json.dumps(timeframe_data, indent=2)}
    Indicators: not used

    IMPORTANT: After your analysis, also provide a simple BUY/NO_BUY recommendation based on bullish signals.
    Format: End your response with "TRADE_SIGNAL: BUY" or "TRADE_SIGNAL: NO_BUY"
    """

    messages = [Message(role="user", content=analysis_prompt)]
    response = await llm_manager.chat(messages)
    ai_summary = response.content.strip()


    trade_signal = "NO_BUY"
    if "TRADE_SIGNAL: BUY" in ai_summary.upper():
        trade_signal = "BUY"
    elif "TRADE_SIGNAL: NO_BUY" in ai_summary.upper():
        trade_signal = "NO_BUY"

    symbol = state.get("symbol", "ETH")

    return {
        "short_term_trend": {
            "user_query": state.get("user_query", ""),
            "symbol": symbol,
            "timeframe_data": timeframe_data,
            "indicators": {},
            "trade_signal": trade_signal,
        },
        "final_output": ai_summary,
        "trade_signal": trade_signal,
        "current_step": "short_term_analysis_completed",
        "execution_log": state.get("execution_log", []) + [f"Short-term trend analysis completed - Trade signal: {trade_signal}"]
    }


async def execute_trade_node(state: AdvancedComprehensiveState) -> Dict[str, Any]:
    """Execute swap trade based on extracted parameters if bullish signal detected."""
    try:
        # Check trade signal - only execute if bullish trend detected
        trade_signal = state.get("trade_signal", "NO_BUY")
        if trade_signal != "BUY":
            return {
                "execution_log": state.get("execution_log", []) + [f"Trade skipped: no bullish signal (signal: {trade_signal})"],
            }

        # Get trade parameters from state (extracted from user query)
        trade_params = state.get("trade_params")
        if not trade_params:
            return {
                "execution_log": state.get("execution_log", []) + ["Trade skipped: no trade parameters found in query"],
            }

        # Extract trading parameters
        from_token = trade_params.get("from_token")
        to_token = trade_params.get("to_token")
        amount = trade_params.get("amount")

        if not all([from_token, to_token, amount]):
            return {
                "execution_log": state.get("execution_log", []) + ["Trade skipped: incomplete trade parameters"],
            }

        # Get RPC URL for trading
        rpc_url = os.getenv("EVM_PROVIDER_URL") or os.getenv("RPC_URL")
        if not rpc_url:
            return {"execution_log": state.get("execution_log", []) + ["Trade skipped: missing EVM_PROVIDER_URL/RPC_URL"]}

        # Execute swap using extracted parameters
        swap_tool = EvmSwapTool(rpc_url=rpc_url)
        res = await swap_tool.execute(
            from_token=from_token,
            to_token=to_token,
            amount=amount,
            signer_type=os.getenv("SIGNER_TYPE", "auto"),
            # signer params are picked up automatically from env if not provided
        )
        if res.error:
            return {
                "trade_result": {"status": "failed", "error": res.error},
                "execution_log": state.get("execution_log", []) + [f"Trade failed: {res.error}"]
            }
        return {
            "trade_result": {"status": "success", "tx": res.output},
            "execution_log": state.get("execution_log", []) + [f"Trade executed: {res.output.get('hash')}"]
        }
    except Exception as e:
        return {
            "trade_result": {"status": "failed", "error": str(e)},
            "execution_log": state.get("execution_log", []) + [f"Trade exception: {str(e)}"]
        }




# ============================================================================
# MACRO TREND ANALYSIS NODES (PARALLEL PROCESSING)
# ============================================================================

fetch_4h_data_node = make_fetch_node(
    "4h",
    MACRO_PRESETS["4h"]["limit"],
    MACRO_PRESETS["4h"]["indicators"],
    "macro_timeframe_data_4h",
)


fetch_daily_data_node = make_fetch_node(
    "1d",
    MACRO_PRESETS["1d"]["limit"],
    MACRO_PRESETS["1d"]["indicators"],
    "macro_timeframe_data_daily",
)


fetch_weekly_data_node = make_fetch_node(
    "1w",
    MACRO_PRESETS["1w"]["limit"],
    MACRO_PRESETS["1w"]["indicators"],
    "macro_timeframe_data_weekly",
)


def _parse_tavily_text_response(text_response: str, symbol: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        import re
        sections = re.split(r'(?=Title:)', text_response.strip())
        sections = [s.strip() for s in sections if s.strip().startswith('Title:')]
        for section in sections[:5]:
            title = None
            url = None
            content = None
            import re as _re
            m = _re.search(r'Title:\s*(.*?)(?=URL:|$)', section, _re.DOTALL)
            if m: title = m.group(1).strip()
            m = _re.search(r'URL:\s*(.*?)(?=Content:|$)', section, _re.DOTALL)
            if m: url = m.group(1).strip()
            m = _re.search(r'Content:\s*(.*?)(?=Title:|$)', section, _re.DOTALL)
            if m: content = m.group(1).strip()
            if title or url or content:
                items.append({
                    "title": title or f"{symbol} News",
                    "url": url or "",
                    "content": (content or "")[:500],
                    "source": "Tavily"
                })
    except Exception:
        pass
    return items


async def _tavily_search(query: str) -> List[Dict[str, Any]]:
    if not tavily_search_tool:
        return []
    result = await tavily_search_tool.execute(query=query, max_results=5, search_depth="basic")
    content = result.output if hasattr(result, "output") and result.output else result
    items: List[Dict[str, Any]] = []
    if isinstance(content, list):
        for it in content[:5]:
            if isinstance(it, dict):
                items.append({
                    "title": it.get("title", ""),
                    "url": it.get("url", ""),
                    "content": (it.get("content", "") or "")[:500],
                    "source": it.get("source", ""),
                })
        return items
    if isinstance(content, str):
        try:
            import json as _json
            parsed = _json.loads(content)
            if isinstance(parsed, list):
                for it in parsed[:5]:
                    if isinstance(it, dict):
                        items.append({
                            "title": it.get("title", ""),
                            "url": it.get("url", ""),
                            "content": (it.get("content", "") or "")[:500],
                            "source": it.get("source", ""),
                        })
            return items
        except Exception:
            pass
        return _parse_tavily_text_response(content, query.split()[0])
    return []


async def search_crypto_news_node(state: AdvancedComprehensiveState) -> Dict[str, Any]:
    """Search crypto news via Tavily MCP"""
    symbol = state.get("symbol", "BTC")
    news_data = await _tavily_search(f"{symbol} cryptocurrency latest news market analysis")
    return {
        "macro_news_data": news_data,
        "execution_log": state.get("execution_log", []) + [f"News search completed for {symbol}"]
    }


async def analyze_macro_trend_node(state: AdvancedComprehensiveState) -> Dict[str, Any]:
    """Analyze macro trend from parallel data"""
    # Combine data from all timeframes
    timeframe_data = {}
    indicators = {}
    news_data = state.get("macro_news_data", [])

    # Collect timeframe data
    if "macro_timeframe_data_4h" in state:
        timeframe_data["4h"] = state["macro_timeframe_data_4h"]
        indicators.update(state["macro_timeframe_data_4h"].get("indicators", {}))

    if "macro_timeframe_data_daily" in state:
        timeframe_data["daily"] = state["macro_timeframe_data_daily"]
        indicators.update(state["macro_timeframe_data_daily"].get("indicators", {}))

    if "macro_timeframe_data_weekly" in state:
        timeframe_data["weekly"] = state["macro_timeframe_data_weekly"]
        indicators.update(state["macro_timeframe_data_weekly"].get("indicators", {}))

    symbol = state.get("symbol", "BTC")

    llm_manager = get_llm_manager()

    analysis_prompt = f"""
    You are a market analyst. Analyze macro trend using raw K-line data from multiple timeframes (4h/daily/weekly) and recent news headlines. Write a concise, human-readable summary WITHOUT any JSON. Include:
    - An overall macro conclusion (bullish/neutral/bearish)
    - 2-4 key price action reasons across timeframes (structure, momentum, breakouts, ranges)
    - Long-term outlook (weeks/months) and key levels if inferable
    - A short risk note

    User Query: {state.get("user_query", "")}
    Timeframe Data: {json.dumps(timeframe_data, indent=2)}
    Technical Indicators: not used
    Recent News (titles+sources): {json.dumps(news_data[:5], indent=2)}
    """

    messages = [Message(role="user", content=analysis_prompt)]
    response = await llm_manager.chat(messages)
    ai_summary = response.content.strip()

    return {
        "macro_trend": {
            "user_query": state.get("user_query", ""),
            "symbol": symbol,
            "timeframe_data": timeframe_data,
            "news_data": news_data,

        },
        "final_output": ai_summary,
        "current_step": "macro_analysis_completed",
        "execution_log": state.get("execution_log", []) + ["Macro trend analysis completed"]
    }


async def deep_research_node(state: AdvancedComprehensiveState) -> Dict[str, Any]:
    """Simplified deep research: LLM-only comprehensive analysis based on the query."""
    query = state.get("user_query", "")
    llm_manager = get_llm_manager()

    prompt = f"""
    Provide a comprehensive, well-structured analysis of the following topic.
    Cover technical/fundamental aspects, historical context, potential risks, and future outlook.
    Use internal knowledge and reasoning only (no external browsing).

    Topic: {query}
    """

    messages = [Message(role="user", content=prompt)]
    response = await llm_manager.chat(messages)
    insights = response.content.strip()

    return {
        "deep_research": {
            "user_query": query,
            "final_insights": insights
        },
        "final_output": insights,
        "current_step": "deep_research_completed",
        "execution_log": state.get("execution_log", []) + ["Deep research completed"]
    }

async def search_deep_research_node(state: AdvancedComprehensiveState) -> Dict[str, Any]:
    """Perform web search for deep research using Tavily MCP with the full user query."""
    query = state.get("user_query", "")
    results = await _tavily_search(query)
    return {
        "deep_research_results": results,
        "execution_log": state.get("execution_log", []) + [f"Deep research web search completed ({len(results)} results)"]
    }

async def synthesize_deep_research_node(state: AdvancedComprehensiveState) -> Dict[str, Any]:
    """Synthesize deep research insights from Tavily search results using LLM."""
    query = state.get("user_query", "")
    results = state.get("deep_research_results", [])
    llm_manager = get_llm_manager()

    synthesis_prompt = f"""
    You are a research analyst. Based on the search results, synthesize a concise, well-structured report.

    Topic: {query}

    Search Results (title, url, content excerpt):
    {json.dumps(results[:8], indent=2)}

    Instructions:
    - Provide key findings, recent developments, and measurable data if present
    - Include brief performance assessment if applicable
    - Reference sources inline using (Title → URL)
    - Keep it factual and avoid speculation
    - No JSON in the output
    """

    messages = [Message(role="user", content=synthesis_prompt)]
    response = await llm_manager.chat(messages)
    insights = response.content.strip()

    return {
        "deep_research": {
            "user_query": query,
            "final_insights": insights
        },
        "final_output": insights,
        "current_step": "deep_research_synthesized",
        "execution_log": state.get("execution_log", []) + ["Deep research synthesized from Tavily results"]
    }


# ============================================================================
# DEPTH RESEARCH NODES (PARALLEL PROCESSING WITH REFLECTION)
# ============================================================================


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def _build_advanced_graph() -> StateGraph:
    """Build the advanced graph with intelligent routing and parallel processing"""
    graph = StateGraph(AdvancedComprehensiveState)

    # Enable monitoring
    graph.enable_monitoring(["execution_time", "success_rate", "routing_performance"])

    # Add core nodes
    graph.add_node("initialize_session", initialize_session_node)
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("analyze_query_intent", analyze_query_intent_node)

    # Add all nodes
    graph.add_node("general_qa", general_qa_node)
    graph.add_node("extract_symbol", extract_symbol_node)

    # Add short-term trend analysis nodes
    graph.add_node("fetch_15m_data", fetch_15m_data_node)
    graph.add_node("fetch_30m_data", fetch_30m_data_node)
    graph.add_node("fetch_1h_data", fetch_1h_data_node)
    graph.add_node("analyze_short_term_trend", analyze_short_term_trend_node)
    graph.add_node("execute_trade", execute_trade_node)

    # Add macro trend analysis nodes
    graph.add_node("fetch_4h_data", fetch_4h_data_node)
    graph.add_node("fetch_daily_data", fetch_daily_data_node)
    graph.add_node("fetch_weekly_data", fetch_weekly_data_node)
    graph.add_node("search_crypto_news", search_crypto_news_node)
    graph.add_node("analyze_macro_trend", analyze_macro_trend_node)

    # Add simplified deep research node
    graph.add_node("deep_research_search", search_deep_research_node)
    graph.add_node("deep_research_synthesize", synthesize_deep_research_node)

    # Disable LLM routing completely to ensure conditional routing works
    graph.llm_router = None
    graph.llm_router_config = {}

    # Set entry point
    graph.set_entry_point("initialize_session")

    # Build routing logic based on query type
    graph.add_edge("initialize_session", "load_memory")
    graph.add_edge("load_memory", "analyze_query_intent")

    # Conditional routing based on query analysis
    def route_based_on_query_type(state):
        """Route based on query analysis result"""
        query_analysis = state.get("query_analysis", {})
        query_type = query_analysis.get("query_type", "general_qa")
        return query_type

    # Clear any existing routing rules that might interfere
    if hasattr(graph, 'routing_rules'):
        graph.routing_rules.clear()

    graph.add_conditional_edges(
        "analyze_query_intent",
        route_based_on_query_type,
        {
            "general_qa": "general_qa",
            "short_term_trend": "extract_symbol",
            "macro_trend": "extract_symbol",
            "deep_research": "deep_research_search"
        }
    )

    # Add fallback edge to ensure graph can complete
    graph.add_edge("analyze_query_intent", "general_qa")

    # Create parallel execution groups
    # Short-term trend parallel group
    graph.add_parallel_group(
        "short_term_data_fetch",
        ["fetch_15m_data", "fetch_30m_data", "fetch_1h_data"],
        {"join_strategy": "all_complete", "error_strategy": "ignore_errors"}
    )

    # Macro trend parallel group
    graph.add_parallel_group(
        "macro_trend_data_fetch",
        ["fetch_4h_data", "fetch_daily_data", "fetch_weekly_data", "search_crypto_news"],
        {"join_strategy": "all_complete", "error_strategy": "ignore_errors"}
    )

    # Define execution flow
    # General Q&A path goes through memory update before END
    graph.add_node("update_memory", update_memory_node)
    graph.add_edge("general_qa", "update_memory")

    # Route based on query type - use different entry points for different analysis types
    def route_to_analysis_type(state):
        """Route to appropriate analysis type based on query classification"""
        query_analysis = state.get("query_analysis", {})
        query_type = query_analysis.get("query_type", "short_term_trend")

        if query_type == "macro_trend":
            return "macro_analysis_entry"
        else:  # short_term_trend or default
            return "short_term_analysis_entry"

    # Create entry point nodes for different analysis types
    async def short_term_analysis_entry_node(state):
        """Entry point for short-term analysis - triggers parallel data fetching"""
        return {"analysis_type": "short_term"}

    async def macro_analysis_entry_node(state):
        """Entry point for macro analysis - triggers parallel data fetching"""
        return {"analysis_type": "macro"}

    graph.add_node("short_term_analysis_entry", short_term_analysis_entry_node)
    graph.add_node("macro_analysis_entry", macro_analysis_entry_node)

    # Conditional routing from extract_symbol to analysis type entry
    graph.add_conditional_edges(
        "extract_symbol",
        route_to_analysis_type,
        {
            "short_term_analysis_entry": "short_term_analysis_entry",
            "macro_analysis_entry": "macro_analysis_entry",
        }
    )

    # Connect analysis entry points to their respective data fetching nodes
    # Short-term analysis triggers parallel data fetching
    graph.add_edge("short_term_analysis_entry", "fetch_15m_data")
    graph.add_edge("short_term_analysis_entry", "fetch_30m_data")
    graph.add_edge("short_term_analysis_entry", "fetch_1h_data")

    # Macro analysis triggers parallel data fetching
    graph.add_edge("macro_analysis_entry", "fetch_4h_data")
    graph.add_edge("macro_analysis_entry", "fetch_daily_data")
    graph.add_edge("macro_analysis_entry", "fetch_weekly_data")
    graph.add_edge("macro_analysis_entry", "search_crypto_news")

    # Short-term trend path entry and joins
    graph.add_edge("fetch_15m_data", "analyze_short_term_trend")
    graph.add_edge("fetch_30m_data", "analyze_short_term_trend")
    graph.add_edge("fetch_1h_data", "analyze_short_term_trend")
    graph.add_edge("analyze_short_term_trend", "execute_trade")
    graph.add_edge("execute_trade", "update_memory")

    # Macro trend joins
    graph.add_edge("fetch_4h_data", "analyze_macro_trend")
    graph.add_edge("fetch_daily_data", "analyze_macro_trend")
    graph.add_edge("fetch_weekly_data", "analyze_macro_trend")
    graph.add_edge("search_crypto_news", "analyze_macro_trend")
    graph.add_edge("analyze_macro_trend", "update_memory")

    # Deep research path
    graph.add_edge("deep_research_search", "deep_research_synthesize")
    graph.add_edge("deep_research_synthesize", "update_memory")

    # Final memory update to END
    graph.add_edge("update_memory", END)

    return graph


# Legacy agent factory removed in simplified version


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

class AdvancedGraphDemo:
    """Advanced demonstration of intelligent routing and parallel processing"""

    def __init__(self):
        self.llm_manager = get_llm_manager()
        self.graph = _build_advanced_graph()

    async def process_query(self, user_query: str, user_name: str = "User") -> Dict[str, Any]:
        """Process a user query through the advanced graph"""

        # Always use real tools - no shutdown interruptions

        # Create initial state
        initial_state = {
            "user_query": user_query,
            "user_name": user_name,
            "session_id": "",
            "query_analysis": None,
            "short_term_trend": None,
            "macro_trend": None,
            "deep_research": None,
            "memory_state": None,
            "execution_log": [],
            "routing_decisions": [],
            "current_step": "started",
            "final_output": "",
            "processing_time": 0.0,
            "parallel_tasks_completed": 0
        }

        # Execute graph with reduced max iterations for faster feedback
        start_time = datetime.now()
        compiled_graph = self.graph.compile()

        # Configure with reduced iterations and real-time logging
        config = {
            "max_iterations": 50,  # Reduced from default 100
            "enable_partial_output": True
        }

        print(f"🔍 Processing query: '{user_query}'")

        # Always use real tools - let exceptions propagate
        result = await compiled_graph.invoke(initial_state, config)

        end_time = datetime.now()

        # Add execution metrics
        result["processing_time"] = (end_time - start_time).total_seconds()
        try:
            metrics = compiled_graph.get_execution_metrics()
            result["execution_metrics"] = metrics
        except:
            result["execution_metrics"] = {}

        return result

    def display_result(self, result: Dict[str, Any]):
        """Display final response with concise, readable output including memory context"""
        user_query = result.get("user_query", "")
        final_output = result.get("final_output", "")
        query_type = result.get("query_analysis", {}).get("query_type", "unknown") if result.get("query_analysis") else "unknown"
        processing_time = result.get("processing_time", 0.0)
        parallel_tasks = result.get("parallel_tasks_completed", 0)
        execution_log = result.get("execution_log", [])[-6:]
        memory_state = result.get("memory_state") or {}

        header = (
            f"\n{'='*80}\n"
            f"🤖 SpoonOS Intelligence Assistant\n"
            f"{'='*80}\n"
            f"📝 Query: {user_query}\n"
            f"🎯 Route: {query_type} | ⚡ {processing_time:.2f}s | 🔄 {parallel_tasks}\n"
            f"{'-'*80}"
        )
        print(header)

        # Short execution trace
        if execution_log:
            print("Steps:")
            for i, log in enumerate(execution_log, 1):
                print(f"  {i}. {log}")

        # Memory teaser
        lp = (memory_state.get("learned_patterns") or {}).get("query_type_counts", {})
        if lp:
            try:
                common = max(lp.items(), key=lambda x: x[1])[0]
                print(f"Memory: frequent route → {common}")
            except Exception:
                pass

        # Final output
        if final_output:
            print("\nResult:")
            print(final_output)
        else:
            print("\nResult: (no output)")

        # Macro extras
        if query_type == "macro_trend":
            macro_section = result.get("macro_trend") or {}
            news_data = macro_section.get("news_data", [])
            if isinstance(news_data, list) and news_data:
                print("\nNews (top 3):")
                for item in news_data[:3]:
                    title = item.get("title", "").strip()
                    url = item.get("url", "").strip()
                    print(f" - {title} -> {url}")

        print(f"{'='*80}\n")


async def main():
    """Run the advanced demonstration with intelligent routing"""
    print("🚀 SpoonOS Advanced Graph Demo - Intelligent Routing & Parallel Processing")
    print("=" * 80)
    print("=" * 80)

    demo = AdvancedGraphDemo()

    # Test queries for different scenarios
    test_queries = [
        # General Q&A
        ("How does blockchain work?", "Bob"),

        # Short-term trend analysis with potential trade
        ("Analyze ETH short-term trend and if bullish, swap 0.1 USDC for ETH on Base", "Diana"),

        # Macro trend analysis
        ("What are the long-term trends for NEO?", "Eve"),

        # Deep research
        ("Research the development of DeFi protocols in the second quarter of 2025, identify any new DeFi projects that have launched, and evaluate their performance.", "Henry"),
    ]

    print(f"🧪 Testing {len(test_queries)} queries across different scenarios...\n")

    for i, (query, user_name) in enumerate(test_queries, 1):
        print(f"Test {i}/{len(test_queries)}")
        result = await demo.process_query(query, user_name)
        demo.display_result(result)
        await asyncio.sleep(1)

if __name__ == "__main__":
    import sys
    asyncio.run(main())
