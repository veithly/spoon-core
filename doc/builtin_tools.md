# Built-in Tools Reference

This document lists all available built-in tools in SpoonOS. These tools are part of the spoon-toolkit and can be used directly in agent configurations without requiring external MCP servers.

**Important**: Built-in tools support flexible configuration through:
1. **Environment Variables** (traditional method)
2. **Tool-level Configuration** (new unified approach)
3. **Hybrid Configuration** (combination of both)

Make sure to set the required environment variables in your `.env` file or configure them directly in tool configurations.

## Environment Variables Setup

Before using built-in tools, configure the required environment variables:

```bash
# Crypto PowerData (for DEX data via OKX API)
OKX_API_KEY=your_okx_api_key
OKX_SECRET_KEY=your_okx_secret_key
OKX_API_PASSPHRASE=your_okx_api_passphrase
OKX_PROJECT_ID=your_okx_project_id

# Chainbase API
CHAINBASE_API_KEY=your_chainbase_api_key

# ThirdWeb API
THIRDWEB_CLIENT_ID=your_thirdweb_client_id

# GoPlusLabs Security API
GO_PLUS_LABS_APP_KEY=your_goplus_api_key
GO_PLUS_LABS_APP_SECRET=your_goplus_secret

# Storage Services
AIOZ_ACCESS_KEY=your_aioz_access_key
AIOZ_SECRET_KEY=your_aioz_secret_key
FOUREVERLAND_ACCESS_KEY=your_foureverland_access_key
FOUREVERLAND_SECRET_KEY=your_foureverland_secret_key
OORT_ACCESS_KEY=your_oort_access_key
OORT_SECRET_KEY=your_oort_secret_key

# Social Media
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
GITHUB_TOKEN=your_github_token

# Blockchain RPC
RPC_URL=your_ethereum_rpc_url
```

## Crypto Tools

### PowerData Tools
Advanced crypto market data from PowerData API:

- **`crypto_powerdata_cex`** - CEX market data and analytics
- **`crypto_powerdata_dex`** - DEX market data and analytics  
- **`crypto_powerdata_indicators`** - Technical indicators and analysis
- **`crypto_powerdata_price`** - Real-time price data

**Configuration:**
```json
{
  "name": "crypto_powerdata_cex",
  "type": "builtin",
  "description": "Crypto PowerData CEX market data",
  "enabled": true,
  "env": {
    "OKX_API_KEY": "your_okx_api_key",
    "OKX_SECRET_KEY": "your_okx_secret_key",
    "OKX_API_PASSPHRASE": "your_okx_api_passphrase",
    "OKX_PROJECT_ID": "your_okx_project_id"
  },
  "config": {
    "timeout": 30,
    "max_retries": 3
  }
}
```

**Environment Variables Required:**
- `OKX_API_KEY`, `OKX_SECRET_KEY`, `OKX_API_PASSPHRASE`, `OKX_PROJECT_ID` (for DEX data)
- Can be set in tool configuration via `env` field or system environment variables

### Basic Crypto Data Tools
Core cryptocurrency data and analysis tools:

- **`get_token_price`** - Get current token prices
- **`get_24h_stats`** - 24-hour trading statistics
- **`get_kline_data`** - Candlestick/K-line chart data
- **`price_threshold_alert`** - Price alert monitoring
- **`lp_range_check`** - Liquidity pool range checking
- **`sudden_price_increase`** - Sudden price movement detection
- **`lending_rate_monitor`** - DeFi lending rate monitoring
- **`crypto_market_monitor`** - General market monitoring
- **`predict_price`** - Price prediction analysis
- **`token_holders`** - Token holder analysis
- **`trading_history`** - Trading history analysis
- **`uniswap_liquidity`** - Uniswap liquidity pool analysis
- **`wallet_analysis`** - Wallet behavior analysis

**Configuration:**
```json
{
  "name": "get_token_price",
  "type": "builtin",
  "description": "Get current token prices",
  "enabled": true,
  "env": {
    "RPC_URL": "https://eth.llamarpc.com",
    "BITQUERY_API_KEY": "your_bitquery_api_key",
    "BITQUERY_CLIENT_ID": "your_bitquery_client_id",
    "BITQUERY_CLIENT_SECRET": "your_bitquery_client_secret"
  },
  "config": {
    "timeout": 30,
    "max_retries": 3
  }
}
```

**Environment Variables Required:**
- `RPC_URL` (for blockchain data)
- `BITQUERY_API_KEY`, `BITQUERY_CLIENT_ID`, `BITQUERY_CLIENT_SECRET` (for some tools)
- Can be set in tool configuration via `env` field or system environment variables

## Data Platform Tools

### Chainbase Tools
Blockchain data from Chainbase API:

- **`get_latest_block_number`** - Latest block information
- **`get_block_by_number`** - Block details by number
- **`get_transaction_by_hash`** - Transaction details by hash
- **`get_account_transactions`** - Account transaction history
- **`contract_call`** - Smart contract function calls
- **`get_account_tokens`** - Account token holdings
- **`get_account_nfts`** - Account NFT holdings
- **`get_account_balance`** - Account balance information
- **`get_token_metadata`** - Token metadata lookup

**Configuration:**
```json
{
  "name": "get_latest_block_number",
  "type": "builtin",
  "description": "Get latest block information",
  "enabled": true,
  "env": {
    "CHAINBASE_API_KEY": "your_chainbase_api_key"
  },
  "config": {
    "timeout": 30,
    "max_retries": 3
  }
}
```

**Environment Variables Required:**
- `CHAINBASE_API_KEY`
- Can be set in tool configuration via `env` field or system environment variables

### ThirdWeb Tools
Web3 development tools via ThirdWeb API:

- **`get_contract_events`** - Contract event logs
- **`get_multichain_transfers`** - Cross-chain transfer data
- **`get_transactions`** - Transaction data
- **`get_contract_transactions`** - Contract-specific transactions
- **`get_contract_transactions_by_signature`** - Transactions by function signature
- **`get_blocks`** - Block information
- **`get_wallet_transactions`** - Wallet transaction history

**Configuration:**
```json
{
  "name": "get_contract_events",
  "type": "builtin",
  "description": "Get contract event logs",
  "enabled": true,
  "env": {
    "THIRDWEB_CLIENT_ID": "your_thirdweb_client_id"
  },
  "config": {
    "timeout": 30,
    "max_retries": 3
  }
}
```

**Environment Variables Required:**
- `THIRDWEB_CLIENT_ID`
- Can be set in tool configuration via `env` field or system environment variables

## Storage Tools

Decentralized storage solutions:

- **`aioz_storage`** - AIOZ decentralized storage
- **`foureverland_storage`** - 4EVERLAND storage platform
- **`oort_storage`** - OORT decentralized storage

**Configuration:**
```json
{
  "name": "aioz_storage",
  "type": "builtin",
  "description": "AIOZ decentralized storage",
  "enabled": true,
  "env": {
    "AIOZ_ACCESS_KEY": "your_aioz_access_key",
    "AIOZ_SECRET_KEY": "your_aioz_secret_key"
  },
  "config": {
    "timeout": 30,
    "max_retries": 3
  }
}
```

**Environment Variables Required:**
- `AIOZ_ACCESS_KEY`, `AIOZ_SECRET_KEY`
- `FOUREVERLAND_ACCESS_KEY`, `FOUREVERLAND_SECRET_KEY`
- `OORT_ACCESS_KEY`, `OORT_SECRET_KEY`
- Can be set in tool configuration via `env` field or system environment variables

## Social Media Tools

Social platform integrations:

- **`discord_tool`** - Discord bot integration
- **`email_tool`** - Email functionality
- **`telegram_tool`** - Telegram bot integration
- **`twitter_tool`** - Twitter/X integration

**Configuration:**
```json
{
  "name": "telegram_tool",
  "type": "builtin",
  "description": "Telegram bot integration",
  "enabled": true,
  "env": {
    "TELEGRAM_BOT_TOKEN": "your_telegram_bot_token",
    "GITHUB_TOKEN": "your_github_token"
  },
  "config": {
    "timeout": 30,
    "max_retries": 3
  }
}
```

**Environment Variables Required:**
- `TELEGRAM_BOT_TOKEN`
- `GITHUB_TOKEN` (for some tools)
- Can be set in tool configuration via `env` field or system environment variables

## Security Tools

Security and risk assessment tools:

- **GoPlusLabs Tools** - Available via MCP integration (not as builtin tools)

## Usage Examples

### Trading Agent with Multiple Tools
```json
{
  "trading_agent": {
    "class": "SpoonReactAI",
    "description": "Comprehensive trading agent",
    "tools": [
      {
        "name": "crypto_powerdata_cex",
        "type": "builtin",
        "description": "CEX market data",
        "enabled": true,
        "env": {
          "OKX_API_KEY": "your_okx_api_key",
          "OKX_SECRET_KEY": "your_okx_secret_key",
          "OKX_API_PASSPHRASE": "your_okx_api_passphrase",
          "OKX_PROJECT_ID": "your_okx_project_id"
        },
        "config": {
          "timeout": 30,
          "max_retries": 3
        }
      },
      {
        "name": "get_token_price",
        "type": "builtin",
        "description": "Basic price lookup",
        "enabled": true,
        "env": {
          "RPC_URL": "https://eth.llamarpc.com",
          "BITQUERY_API_KEY": "your_bitquery_api_key"
        },
        "config": {
          "timeout": 30,
          "max_retries": 3
        }
      },
      {
        "name": "uniswap_liquidity",
        "type": "builtin",
        "description": "Uniswap pool analysis",
        "enabled": true,
        "env": {
          "RPC_URL": "https://eth.llamarpc.com"
        },
        "config": {
          "timeout": 30,
          "max_retries": 3
        }
      }
    ]
  }
}
```

### Research Agent with Data Tools
```json
{
  "research_agent": {
    "class": "SpoonReactAI",
    "description": "Blockchain research agent",
    "tools": [
      {
        "name": "get_latest_block_number",
        "type": "builtin",
        "description": "Latest block data",
        "enabled": true,
        "env": {
          "CHAINBASE_API_KEY": "your_chainbase_api_key"
        },
        "config": {
          "timeout": 30,
          "max_retries": 3
        }
      },
      {
        "name": "get_contract_events",
        "type": "builtin",
        "description": "Contract event analysis",
        "enabled": true,
        "env": {
          "THIRDWEB_CLIENT_ID": "your_thirdweb_client_id"
        },
        "config": {
          "timeout": 30,
          "max_retries": 3
        }
      },
      {
        "name": "wallet_analysis",
        "type": "builtin",
        "description": "Wallet behavior analysis",
        "enabled": true,
        "env": {
          "RPC_URL": "https://eth.llamarpc.com",
          "BITQUERY_API_KEY": "your_bitquery_api_key"
        },
        "config": {
          "timeout": 30,
          "max_retries": 3
        }
      }
    ]
  }
}
```

## Important Notes

1. **Environment Variables**: API keys and credentials can be configured via environment variables or tool-level `env` configuration
2. **Tool-level Configuration**: The new unified configuration system allows setting environment variables directly in tool configurations
3. **Priority Order**: Tool-level environment variables override system environment variables
4. **Dependencies**: Some tools may require additional Python packages to be installed
5. **Rate Limits**: Be aware of API rate limits when configuring multiple tools
6. **Error Handling**: Tools will gracefully handle missing environment variables with clear error messages
7. **Configuration**: Tool `config` sections contain tool-specific settings like timeout, retry attempts, etc.

## Troubleshooting

### Missing API Keys
If you see errors like "Missing CHAINBASE_API_KEY in environment variables!", make sure to:

1. Add the required environment variable to your `.env` file
2. Restart your application to load the new environment variables
3. Check that the variable name matches exactly (case-sensitive)

### Tool Not Working
If a tool isn't working as expected:

1. Check that all required environment variables are set
2. Verify your API keys are valid and have the necessary permissions
3. Check the tool's specific documentation in the spoon-toolkit repository

For more information on tool configuration, see the [Configuration Guide](./configuration.md).