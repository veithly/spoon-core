"""
Neo Toolkit Agent Demo - Comprehensive demonstration using spoon_ai framework

This example demonstrates Neo blockchain tools using spoon_ai agents, showcasing:
- Agent-based tool interaction
- Comprehensive Neo toolkit coverage (60 tools)
- Real-world usage scenarios
- Error handling and validation

Uses testnet for all demonstrations with embedded test data.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import spoon_ai framework
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.tools import ToolManager
from spoon_ai.chat import ChatBot
from spoon_ai.llm.manager import get_llm_manager
from pydantic import Field

# Import Neo tools for agent
from spoon_toolkits.crypto.neo import (
    # Address tools (6)
    GetAddressCountTool,
    GetAddressInfoTool,
    ValidateAddressTool,
    GetActiveAddressesTool,
    GetTotalSentAndReceivedTool,
    GetTransferByAddressTool,
    
    # Asset tools (5)
    GetAssetCountTool,
    GetAssetInfoByNameTool,
    GetAssetInfoByHashTool,
    GetAssetInfosTool,
    GetAssetInfoByAssetAndAddressTool,

    # Block tools (6)
    GetBlockCountTool,
    GetBlockByHeightTool,
    GetBestBlockHashTool,
    GetRecentBlocksInfoTool,
    GetBlockByHashTool,
    GetBlockRewardByHashTool,

    # Contract tools (5)
    GetContractCountTool,
    GetContractByHashTool,
    GetContractListByNameTool,
    GetVerifiedContractByContractHashTool,
    GetVerifiedContractTool,

    # Smart Contract Call tools (3)
    GetScCallByContractHashTool,
    GetScCallByContractHashAddressTool,
    GetScCallByTransactionHashTool,

    # Transaction tools (9)
    GetTransactionCountTool,
    GetTransactionCountByAddressTool,
    GetRawTransactionByBlockHashTool,
    GetRawTransactionByBlockHeightTool,
    GetRawTransactionByTransactionHashTool,
    GetRawTransactionByAddressTool,
    GetTransferByBlockHashTool,
    GetTransferByBlockHeightTool,
    GetTransferEventByTransactionHashTool,

    # NEP tools (11)
    GetNep11BalanceTool,
    GetNep11ByAddressAndHashTool,
    GetNep11TransferByAddressTool,
    GetNep11TransferByBlockHeightTool,
    GetNep11TransferByTransactionHashTool,
    GetNep11TransferCountByAddressTool,
    GetNep17TransferByAddressTool,
    GetNep17TransferByBlockHeightTool,
    GetNep17TransferByContractHashTool,
    GetNep17TransferByTransactionHashTool,
    GetNep17TransferCountByAddressTool,

    # Application Log and State tools (2)
    GetApplicationLogTool,
    GetApplicationStateTool,

    # Governance tools (9)
    GetCandidateCountTool,
    GetCandidateByAddressTool,
    GetCandidateByVoterAddressTool,
    GetScVoteCallByCandidateAddressTool,
    GetScVoteCallByTransactionHashTool,
    GetScVoteCallByVoterAddressTool,
    GetVotersByCandidateAddressTool,
    GetVotesByCandidateAddressTool,
    GetTotalVotesTool,
    
    # Governance tools (1)
    GetCommitteeInfoTool,
)

# Load environment variables
load_dotenv()


class NeoToolkitAgentDemo:
    """Neo Toolkit Agent-based comprehensive demonstration"""

    TEST_DATA = {
        "network": "testnet",
        "basic_test_data": {
            "addresses": [
                "NUTtedVrz5RgKAdCvtKiq3sRkb9pizcewe",
                "NaU3shtZqnR1H6XnDTxghorgkXN687C444",
                "0x661fdb769a0b854427eba1b2bdd73480441ca8c9"
            ],
            
            "transaction_hash": "0xac72c504141743c5ac538ecaf360502b8492208b60d667bd2b3eb445d7ce3c6c",
            "block_height": 10226921,
            "block_hash": "0x9f238c3759e655937b277ef7f2755da2875d8811f415c60b57b8b07d1791de0f",
            "start_time": "2025-10-13T00:00:00Z",
            "end_time": "2025-10-13T23:59:59Z",
            "limit": 5,
            "skip": 0,
            "token_id": "8+wz0DmgIkXEzh8XgoZAhJD6XbJnUE5w1u9WRoETN3U="
        },
        "contract_hashes": {
            "NEO": "0xef4073a0f2b305a38ec4050e4d3d28bc40ea63f5",
            "GAS": "0xd2a4cff31913016155e38e474a2c06d08be276cf",
            "NEO_TOKEN": "0xef4073a0f2b305a38ec4050e4d3d28bc40ea63f5"
        },
        "asset_names": ["fWBTC", "NEO"],
        "governance_data": {
            "active_candidates": ["NS4uUWusVY4sXJ4Fej9uJ2NxEWNoi4Teft"],
        }
    }

    def __init__(self):
        """Initialize the demo with embedded test data"""
        self.load_test_data()
        self.agents = {}

    def load_test_data(self):
        """Load test data from embedded TEST_DATA configuration"""
        try:
            data = self.TEST_DATA
            
            # Load basic configuration
            self.network = data.get("network", "testnet")

            # Load basic test data
            basic_data = data.get("basic_test_data", {})
            self.demo_address = basic_data.get("addresses", [])[0] if basic_data.get("addresses") else "default_address"
            self.demo_addresses = basic_data.get("addresses", [])
            self.test_tx_hash = basic_data.get("transaction_hash", "")
            self.test_block_height = basic_data.get("block_height", 0)
            self.test_block_hash = basic_data.get("block_hash", "")
            self.test_token_id = basic_data.get("token_id", "")
            self.test_limit = basic_data.get("limit", 5)
            self.test_skip = basic_data.get("skip", 0)

            # Load contract and asset data
            contracts = data.get("contract_hashes", {})
            self.demo_contract = contracts.get("NEO", "")
            self.gas_contract = contracts.get("GAS", "")
            self.demo_asset_name = data.get("asset_names", ["NEO"])[0] if data.get("asset_names") else "NEO"

            # Load governance data
            governance_data = data.get("governance_data", {})
            self.active_candidates = governance_data.get("active_candidates", [])
            self.candidate_public_keys = governance_data.get("candidate_public_keys", [])
            print(f"✳️Running the entire example takes around 10 minutes")
            print(f"✅ Loaded test data from embedded configuration")
            print(f"   Network: {self.network}")
            print(f"   Addresses: {len(self.demo_addresses)} available")
            print(f"   Candidates: {len(self.active_candidates)} active")

        except Exception as e:
            print(f"❌ Failed to load test data: {e}")
            # Set minimal defaults
            self.network = "testnet"
            self.demo_address = ""
            self.demo_addresses = []
            self.demo_contract = ""
            self.gas_contract = ""
            self.demo_asset_name = "NEO"
            self.test_tx_hash = ""
            self.test_block_height = 0
            self.test_block_hash = ""
            self.test_token_id = ""
            self.test_limit = 20
            self.test_skip = 0
            self.active_candidates = []
            self.candidate_public_keys = []

    def create_agent(self, name: str, tools: List, description: str) -> ToolCallAgent:
        """Create a specialized agent with specific tools"""
        network = self.network
        test_limit = self.test_limit
        test_skip = self.test_skip

        class NeoSpecializedAgent(ToolCallAgent):
            agent_name: str = name
            agent_description: str = description
            # Increase timeout for Neo RPC calls which can be slow
            _default_timeout: float = 120.0  # 2 minutes per step for RPC-heavy operations
            system_prompt: str = f"""
            You are a Neo blockchain specialist focused on {description}.
            Use the available tools to analyze Neo blockchain data and provide comprehensive insights.
            Always specify network='{network}' when calling tools.

            **Pagination Support:**
            22 tools support optional Skip and Limit parameters for efficient pagination:
            - Skip: the number of items to skip (default: {test_skip})
            - Limit: the number of items to return (default: {test_limit})

            When calling tools that support pagination (especially for list queries), always include:
            - Example: GetAssetInfoByNameTool(asset_name="NEO", Skip={test_skip}, Limit={test_limit}, network="{network}")

            Complete list of tools supporting pagination:
            - GetAssetInfoByNameTool 
            - GetAssetInfosTool 
            - GetRecentBlocksInfoTool 
            - GetContractListByNameTool 
            - GetVerifiedContractTool 
            - GetCommitteeInfoTool 
            - GetApplicationStateTool 
            - GetNep11ByAddressAndHashTool 
            - GetNep11TransferByAddressTool 
            - GetNep11TransferByBlockHeightTool 
            - GetNep11TransferByTransactionHashTool 
            - GetNep17TransferByAddressTool 
            - GetNep17TransferByBlockHeightTool 
            - GetNep17TransferByContractHashTool 
            - GetNep17TransferByTransactionHashTool 
            - GetScCallByContractHashTool 
            - GetScCallByContractHashAddressTool 
            - GetScCallByTransactionHashTool 
            - GetRawTransactionByBlockHeightTool 
            - GetCandidateByVoterAddressTool 
            - GetScVoteCallByCandidateAddressTool 
            - GetScVoteCallByVoterAddressTool
            - GetVotersByCandidateAddressTool
            - GetTransferByAddressTool 

            Provide clear, informative responses based on the tool results.
            
            IMPORTANT: After calling tools and receiving results, you MUST provide a comprehensive summary and analysis. 
            Do not just return the raw tool output. Instead, analyze the data, extract key insights, and present a 
            well-structured response that answers the user's question completely.
            """
            max_steps: int = 10  # Increased to allow for tool calls + summary generation
            available_tools: ToolManager = Field(default_factory=lambda: ToolManager(tools))

        agent = NeoSpecializedAgent(
            llm=ChatBot(
                llm_provider="openrouter",
                model_name="anthropic/claude-3.5-sonnet"
            )
        )
        return agent

    def setup_agents(self):
        """Setup specialized agents for different Neo toolkit categories"""

        # Blockchain Explorer Agent (6 tools)
        blockchain_tools = [
            GetBlockCountTool(),
            GetBlockByHeightTool(),
            GetBestBlockHashTool(),
            GetRecentBlocksInfoTool(),
            GetBlockByHashTool(),
            GetBlockRewardByHashTool()
        ]
        self.agents['blockchain'] = self.create_agent(
            "Blockchain Explorer",
            blockchain_tools,
            "Expert in Neo blockchain exploration, block analysis, and network monitoring"
        )

        # Address Analyst Agent (6 tools)
        address_tools = [
            GetAddressCountTool(),
            GetAddressInfoTool(),
            ValidateAddressTool(),
            GetActiveAddressesTool(),
            GetTotalSentAndReceivedTool(),
            GetTransferByAddressTool(),
        ]
        self.agents['address'] = self.create_agent(
            "Address Analyst",
            address_tools,
            "Specialist in Neo address validation, analysis, and transaction tracking"
        )

        # Asset Manager Agent (5 tools)
        asset_tools = [
            GetAssetCountTool(),
            GetAssetInfoByNameTool(),
            GetAssetInfoByHashTool(),
            GetAssetInfosTool(),
            GetAssetInfoByAssetAndAddressTool()
        ]
        self.agents['asset'] = self.create_agent(
            "Asset Manager",
            asset_tools,
            "Expert in Neo asset management, token information, and portfolio analysis"
        )

        # Transaction Tracker Agent (9 tools)
        transaction_tools = [
            GetTransactionCountTool(),
            GetTransactionCountByAddressTool(),
            GetRawTransactionByBlockHashTool(),
            GetRawTransactionByBlockHeightTool(),
            GetRawTransactionByTransactionHashTool(),
            GetRawTransactionByAddressTool(),
            GetTransferByBlockHashTool(),
            GetTransferByBlockHeightTool(),
            GetTransferEventByTransactionHashTool()
        ]
        self.agents['transaction'] = self.create_agent(
            "Transaction Tracker",
            transaction_tools,
            "Specialist in Neo transaction analysis, tracking, and verification"
        )

        # NEP Token Expert Agent (11 tools)
        nep_tools = [
            GetNep11BalanceTool(),
            GetNep11ByAddressAndHashTool(),
            GetNep11TransferByAddressTool(),
            GetNep11TransferByBlockHeightTool(),
            GetNep11TransferByTransactionHashTool(),
            GetNep11TransferCountByAddressTool(),
            GetNep17TransferByAddressTool(),
            GetNep17TransferByBlockHeightTool(),
            GetNep17TransferByContractHashTool(),
            GetNep17TransferByTransactionHashTool(),
            GetNep17TransferCountByAddressTool()
        ]
        self.agents['nep'] = self.create_agent(
            "NEP Token Expert",
            nep_tools,
            "Expert in NEP-17 fungible tokens and NEP-11 NFTs, transfers, and balance management"
        )

        # Smart Contract Auditor Agent (5 tools)
        contract_tools = [
            GetContractCountTool(),
            GetContractByHashTool(),
            GetContractListByNameTool(),
            GetVerifiedContractByContractHashTool(),
            GetVerifiedContractTool()
        ]
        self.agents['contract'] = self.create_agent(
            "Smart Contract Auditor",
            contract_tools,
            "Specialist in Neo smart contract analysis, verification, state management, and deployment information"
        )

        # Smart Contract Call Analyst Agent (3 tools)
        sc_call_tools = [
            GetScCallByContractHashTool(),
            GetScCallByContractHashAddressTool(),
            GetScCallByTransactionHashTool()
        ]
        self.agents['sc_call'] = self.create_agent(
            "Smart Contract Call Analyst",
            sc_call_tools,
            "Expert in analyzing smart contract call patterns, interactions, and invocation history"
        )

        # Governance Monitor Agent (10 tools)
        governance_tools = [
            GetCandidateCountTool(),
            GetCandidateByAddressTool(),
            GetCandidateByVoterAddressTool(),
            GetScVoteCallByCandidateAddressTool(),
            GetScVoteCallByTransactionHashTool(),
            GetScVoteCallByVoterAddressTool(),
            GetVotersByCandidateAddressTool(),
            GetVotesByCandidateAddressTool(),
            GetTotalVotesTool(),
            GetCommitteeInfoTool()
        ]
        self.agents['governance'] = self.create_agent(
            "Governance Monitor",
            governance_tools,
            "Expert in Neo governance, voting mechanisms, committee operations, and consensus analysis"
        )
        
        # Application Log Analyst Agent (2 tools)
        log_tools = [
            GetApplicationLogTool(),
            GetApplicationStateTool()
        ]
        self.agents['logs'] = self.create_agent(
            "Application Log Analyst",
            log_tools,
            "Specialist in analyzing smart contract execution logs, debugging, and application states"
        )

    def print_section_header(self, title: str):
        """Print formatted section header"""
        print(f"\n{'='*80}")
        print(f" {title}")
        print(f"{'='*80}")

    async def run_agent_scenario(self, agent_name: str, scenario_title: str, user_message: str):
        """Run a specific scenario with an agent"""
        print(f"\n{'-'*60}")
        print(f" Agent: {self.agents[agent_name].agent_name}")
        print(f" Scenario: {scenario_title}")
        print(f" Query: {user_message}")
        print(f"{'-'*60}")

        try:
            # Clear agent state before running
            self.agents[agent_name].clear()

            # Run the agent with the user message
            response = await self.agents[agent_name].run(user_message)

            # Display response with better formatting
            print(f"\n{'='*60}")
            print(f"✅ AI Agent Response:")
            print(f"{'='*60}")
            print(response)
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

    async def demo_blockchain_exploration(self):
        """Demonstrate blockchain exploration capabilities"""
        self.print_section_header("1. Blockchain Exploration with AI Agent")

        # Scenario 1: Current Network Status
        # Uses: GetBlockCountTool, GetBestBlockHashTool
        await self.run_agent_scenario(
            'blockchain',
            "Current Network Status",
            f"What's the current status of the Neo {self.network}? Show me the latest block information and total block count."
        )

        # Scenario 2: Historical Block Analysis by Height
        # Uses: GetBlockByHeightTool
        await self.run_agent_scenario(
            'blockchain',
            "Historical Block Analysis by Height",
            f"Analyze block number {self.test_block_height} on {self.network}. What information can you provide about this block?"
        )

        # Scenario 3: Block by Hash
        # Uses: GetBlockByHashTool
        if self.test_block_hash:
            await self.run_agent_scenario(
                'blockchain',
                "Block Analysis by Hash",
                f"Get detailed information for block hash {self.test_block_hash} on {self.network}. What can you tell me about this block?"
            )

        # Scenario 4: Block Reward Analysis
        # Uses: GetBlockRewardByHashTool
        if self.test_block_hash:
            await self.run_agent_scenario(
                'blockchain',
                "Block Reward Analysis",
                f"Get block reward information for block hash {self.test_block_hash} on {self.network}. What rewards were distributed in this block?"
            )

        # Scenario 5: Recent Blocks Overview (without explicit limit)
        # Uses: GetRecentBlocksInfoTool
        await self.run_agent_scenario(
            'blockchain',
            "Recent Blocks Overview",
            f"Give me an overview of recent blocks on Neo {self.network}. What's the network activity like?"
        )

        # Scenario 6: Recent 50 Blocks (explicit Limit=50)
        # Uses: GetRecentBlocksInfoTool with Limit=50
        await self.run_agent_scenario(
            'blockchain',
            "Recent 50 Blocks",
            f"Use the get_recent_blocks_info tool to get exactly 10 recent blocks from Neo {self.network}. You MUST set the Limit parameter to 50 and Skip parameter to 0. Return the complete list of 50 blocks."
        )

        # Scenario 7: Block by not exit Hash
        # Uses: GetBlockByHashTool
        await self.run_agent_scenario(
            'blockchain',
            "Block Analysis by not exit Hash",
            f"Get detailed information for block hash 99999999999999 on {self.network}. What can you tell me about this block?"
        )

    async def demo_address_analysis(self):
        """Demonstrate address analysis capabilities"""
        self.print_section_header("2. Address Analysis with AI Agent")

        # Scenario 1: Address Count
        # Uses: GetAddressCountTool
        await self.run_agent_scenario(
            'address',
            "Network Address Count",
            f"How many addresses are registered on Neo {self.network}? Get the total address count."
        )

        # Scenario 2: Address Validation
        # Uses: ValidateAddressTool
        if self.demo_address:
            await self.run_agent_scenario(
                'address',
                "Address Validation",
                f"Validate the address format for {self.demo_address}. Is it a valid Neo address? Use the validate_address tool."
            )

        # Scenario 3: Address Validation
        # Uses: ValidateAddressTool
        if self.demo_address:
            await self.run_agent_scenario(
                'address',
                "Address Validation not exit",
                f"Validate the address format for 0x232323. Is it a valid Neo address? Use the validate_address tool."
            )

        # Scenario 4: Address Profile Analysis
        # Uses: GetAddressInfoTool
        await self.run_agent_scenario(
            'address',
            "Address Profile Analysis",
            f"Get detailed information for address {self.demo_address}. What can you tell me about this address's activity, first use time, last use time, and transactions sent?"
        )

        # Scenario 5: Active Addresses Analysis
        # Uses: GetActiveAddressesTool
        await self.run_agent_scenario(
            'address',
            "Active Addresses Analysis",
            f"Get active address counts for the last 7 days on Neo {self.network}. Use days=7. What are the network activity patterns?"
        )

        # Scenario 6: Active Addresses Analysis
        # Uses: GetActiveAddressesTool
        await self.run_agent_scenario(
            'address',
            "Active Addresses Analysis",
            f"Get active address counts for the last 30 days on Neo {self.network}. Use days=30. What are the network activity patterns?"
        )


        # Scenario 7: Total Sent and Received Analysis
        # Uses: GetTotalSentAndReceivedTool
        if self.demo_address and self.demo_contract:
            await self.run_agent_scenario(
                'address',
                "Token Sent/Received Analysis",
                f"Get the total sent and received amounts for address NU5CEku1rJmZBxifmUZBZa1yzEAJpeg3tj and contract {self.demo_contract}. What are the transaction volumes?"
            )

        # Scenario 8: Total Sent and Received Analysis
        # Uses: GetTotalSentAndReceivedTool
        if self.demo_address and self.demo_contract:
            await self.run_agent_scenario(
                'address',
                "Token Sent/Received Analysis",
                f"Get the total sent and received amounts for address {self.demo_address} and contract {self.demo_contract} and network is {self.network}. What are the transaction volumes?"
            )

        # Scenario 9: Transfer Records Analysis
        # Uses: GetTransferByAddressTool
        if self.demo_address:
            await self.run_agent_scenario(
                'address',
                "Transfer Records Analysis",
                f"Use get_transfer_by_address tool to get transfer records for address {self.demo_address} on Neo {self.network} with pagination (Skip={self.test_skip}, Limit={self.test_limit}). What transfer patterns can you observe?"
            )

        # Scenario 10: Address Analysis with Script Hash Format
        # Uses: GetAddressInfoTool, GetTransferByAddressTool
        # Uses the new address in script hash format (0x...)
        if len(self.demo_addresses) >= 3:
            script_hash_address = self.demo_addresses[2]  # 0x661fdb769a0b854427eba1b2bdd73480441ca8c9
            await self.run_agent_scenario(
                'address',
                "Script Hash Address Analysis",
                f"Analyze address {script_hash_address} on Neo {self.network}. Use get_address_info and get_transfer_by_address tools with Skip=0 and Limit={self.test_limit} to get address details and transfer records."
            )

    async def demo_asset_management(self):
        """Demonstrate asset management capabilities"""
        self.print_section_header("3. Asset Management with AI Agent")

        # Scenario 1: Asset Count
        # Uses: GetAssetCountTool
        await self.run_agent_scenario(
            'asset',
            "Asset Count",
            f"How many assets are registered on Neo {self.network}? Get the total asset count."
        )

        # Scenario 2: Asset Discovery by Name
        # Uses: GetAssetInfoByNameTool
        await self.run_agent_scenario(
            'asset',
            "Asset Discovery by Name",
            f"Search for assets named '{self.demo_asset_name}' on Neo {self.network} with pagination (Skip={self.test_skip}, Limit={self.test_limit}). What are the matching assets and their properties?"
        )

        # Scenario 3: Asset Info by Hash
        # Uses: GetAssetInfoByHashTool
        if self.demo_contract:
            await self.run_agent_scenario(
                'asset',
                "Asset Info by Hash",
                f"Get detailed information for asset with hash {self.demo_contract} on Neo {self.network}. What are the asset's properties, symbol, decimals, and total supply?"
            )

        # Scenario 4: Asset Metadata Lookup
        # Uses: GetAssetInfosTool
        if self.demo_contract and self.gas_contract:
            await self.run_agent_scenario(
                'asset',
                "Asset Metadata Lookup",
                (
                    f"Use get_asset_infos tool to query metadata for assets with contract hashes [{self.demo_contract}, {self.gas_contract}] on Neo {self.network} with Skip={self.test_skip} and Limit=10. "
                    f"The addresses parameter should be an array containing these two contract hashes: ['{self.demo_contract}', '{self.gas_contract}']. "
                    f"Summarize symbol, decimals, totalSupply and asset type for each asset."
                )
            )
        # Scenario 5: Specific Asset Balance
        # Uses: GetAssetInfoByAssetAndAddressTool
        if self.demo_address and self.demo_contract:
            await self.run_agent_scenario(
                'asset',
                "Specific Asset Balance",
                f"Get the balance and details for asset {self.demo_contract} held by address {self.demo_address} on Neo {self.network}. What is the balance and asset information?"
            )

        # Scenario 6: Specific Asset Balance
        # Uses: GetAssetInfoByAssetAndAddressTool-different address
        if self.demo_address and self.demo_contract:
            script_hash_address = self.demo_addresses[2]  # 0x661fdb769a0b854427eba1b2bdd73480441ca8c9
            await self.run_agent_scenario(
                'asset',
                "Specific Asset Balance",
                f"Get the balance and details for asset {self.demo_contract} held by address {script_hash_address} on Neo {self.network}. What is the balance and asset information?"
            )

        # # Scenario 7: Comprehensive Asset Analysis
        # Uses: Multiple tools (GetAssetCountTool, GetAssetInfoByNameTool, GetAssetInfosTool)
        if self.demo_address:
            await self.run_agent_scenario(
                'asset',
                "Comprehensive Asset Analysis",
                f"Perform a comprehensive asset analysis for address {self.demo_address} on Neo {self.network}. Include total asset count, address portfolio, and detailed asset information."
            )

    async def demo_transaction_tracking(self):
        """Demonstrate transaction tracking capabilities"""
        self.print_section_header("4. Transaction Tracking with AI Agent")

        # Scenario 1: Network Transaction Overview
        # Uses: GetTransactionCountTool
        await self.run_agent_scenario(
            'transaction',
            "Network Transaction Overview",
            f"Give me an overview of transaction activity on Neo {self.network}. How many transactions have been processed?"
        )

        # Scenario 2: Address Transaction Count
        # Uses: GetTransactionCountByAddressTool
        if self.demo_address:
            await self.run_agent_scenario(
                'transaction',
                "Address Transaction Count",
                f"Get the total number of transactions for address {self.demo_address} on Neo {self.network}. How many transactions has this address been involved in?"
            )

        
        # Scenario 3: Address Transaction Count
        # Uses: GetTransactionCountByAddressTool-different address
        if self.demo_address:
            script_hash_address = self.demo_addresses[2]  # 
            await self.run_agent_scenario(
                'transaction',
                "Address Transaction Count",
                f"Get the total number of transactions for address {script_hash_address} on Neo {self.network}. How many transactions has this address been involved in?"
            )

        # Scenario 4: Address Transaction Count (Different Address)
        # Uses: GetTransactionCountByAddressTool 
        await self.run_agent_scenario(
            'transaction',
            "Address Transaction Count (Different Address)",
            f"Get the total number of transactions for address 0x661fdb769a0b854427eba1b2bdd73480441ca8c9 on Neo {self.network}. How many transactions has this address been involved in?"
        )

        # Scenario 5: Raw Transaction by Hash
        # Uses: GetRawTransactionByTransactionHashTool
        if self.test_tx_hash:
            await self.run_agent_scenario(
                'transaction',
                "Raw Transaction by Hash",
                f"Get the raw transaction data for transaction hash {self.test_tx_hash} on Neo {self.network}. What are the transaction details?"
            )


        # Scenario 6: Raw Transactions by Block Hash
        # Uses: GetRawTransactionByBlockHashTool
        if self.test_block_hash:
            await self.run_agent_scenario(
                'transaction',
                "Raw Transactions by Block Hash",
                f"Get all raw transactions in block with hash {self.test_block_hash} on Neo {self.network}. What transactions are included in this block?"
            )

        # Scenario 7: Raw Transactions by Block Height
        # Uses: GetRawTransactionByBlockHeightTool
        if self.test_block_height:
            await self.run_agent_scenario(
                'transaction',
                "Raw Transactions by Block Height",
                f"Get all raw transactions in block height {self.test_block_height} on Neo {self.network} using pagination (Skip={self.test_skip}, Limit={self.test_limit}). What types of transactions occurred?"
            )

        # Scenario 8: Transfer Records by Block Hash
        # Uses: GetTransferByBlockHashTool
        if self.test_block_hash:
            await self.run_agent_scenario(
                'transaction',
                "Transfer Records by Block Hash",
                f"Get all transfer records in block with hash {self.test_block_hash} on Neo {self.network}. What asset transfers occurred in this block?"
            )

        # Scenario 9: Transfer Records by Block Height
        # Uses: GetTransferByBlockHeightTool
        if self.test_block_height:
            await self.run_agent_scenario(
                'transaction',
                "Transfer Records by Block Height",
                f"Get all transfer records in block height {self.test_block_height} on Neo {self.network}. What asset transfers occurred in this block?"
            )

        # Scenario 10: Transfer Event by Transaction Hash
        # Uses: GetTransferEventByTransactionHashTool
        if self.test_tx_hash:
            await self.run_agent_scenario(
                'transaction',
                "Transfer Event by Transaction Hash",
                f"Get the transfer event details for transaction hash {self.test_tx_hash} on Neo {self.network}. What transfer events occurred in this transaction?"
            )

        # Scenario 11: Raw Transactions by Address
        # Uses: GetRawTransactionByAddressTool
        if self.demo_address:
            await self.run_agent_scenario(
                'transaction',
                "Raw Transactions by Address",
                f"Get all raw transactions for address {self.demo_address} on Neo {self.network} using pagination (Skip={self.test_skip}, Limit={self.test_limit}). What transactions are associated with this address?"
            )
        
        # Scenario 12: Address Transaction Count (Different Address)
        # Uses: GetTransactionCountByAddressTool 
        await self.run_agent_scenario(
            'transaction',
            "Raw Transactions (Different Address)",
           f"Get all raw transactions for address 0x661fdb769a0b854427eba1b2bdd73480441ca8c9 on Neo {self.network} using pagination (Skip={self.test_skip}, Limit=5). What transactions are associated with this address?"
        )

    async def demo_nep_tokens(self):
        """Demonstrate NEP token analysis capabilities"""
        self.print_section_header("5. NEP Token Analysis with AI Agent")

        # Scenario 1: NEP-17 Transfer Analysis by Address
        # Uses: GetNep17TransferByAddressTool
        if self.demo_address:
            await self.run_agent_scenario(
                'nep',
                "NEP-17 Transfer Analysis by Address",
                f"Use get_nep17_transfer_by_address tool to analyze NEP-17 token transfers for address {self.demo_address} on Neo {self.network} using pagination (Skip={self.test_skip}, Limit={self.test_limit}). What token activity can you find?"
            )

        # Scenario 2: NEP-17 Transfer Analysis by Contract Hash
        # Uses: GetNep17TransferByContractHashTool(Sometimes the connection times out)
        if self.demo_contract:
            await self.run_agent_scenario(
                'nep',
                "NEP-17 Transfer Analysis by Contract",
                f"Use get_nep17_transfer_by_contract_hash tool to analyze token activity for contract 0xaedc84b14a9d09cb9f9ae14b54fbc8b8d84ae5eb on Neo {self.network} with pagination Skip={self.test_skip} and Limit=2. What transfer patterns do you see?"
            )

        # Scenario 3: NEP-17 Transfer Analysis by Block Height
        # Uses: GetNep17TransferByBlockHeightTool
        if self.test_block_height:
            await self.run_agent_scenario(
                'nep',
                "NEP-17 Transfer Analysis by Block Height",
                f"Use get_nep17_transfer_by_block_height tool to analyze NEP-17 token transfers in block height {self.test_block_height} on Neo {self.network} using pagination (Skip={self.test_skip}, Limit={self.test_limit}). What token transfers occurred in this block?"
            )

        # Scenario 4: NEP-17 Transfer Analysis by Transaction Hash
        # Uses: GetNep17TransferByTransactionHashTool
        if self.test_tx_hash:
            await self.run_agent_scenario(
                'nep',
                "NEP-17 Transfer Analysis by Transaction Hash",
                f"Use get_nep17_transfer_by_transaction_hash tool to analyze NEP-17 token transfer details for transaction hash {self.test_tx_hash} on Neo {self.network}. What token transfer details can you find?"
            )

        # Scenario 5: NEP-17 Transfer Count by Address
        # Uses: GetNep17TransferCountByAddressTool
        if self.demo_address:
            await self.run_agent_scenario(
                'nep',
                "NEP-17 Transfer Count by Address",
                f"Use get_nep17_transfer_count_by_address tool to get the total number of NEP-17 token transfers for address {self.demo_address} on Neo {self.network}. How many token transfers has this address been involved in?"
            )

        # Scenario 6: NEP-11 Asset Holdings by Address and Contract
        # Uses: GetNep11ByAddressAndHashTool
        if self.demo_address and self.demo_contract:
            await self.run_agent_scenario(
                'nep',
                "NEP-11 Asset Holdings",
                f"Use get_nep11_by_address_and_hash tool to check NEP-11 (NFT) assets for address NVGUQ1qyL4SdSm7sVmGVkXetjEsvw2L3NT and contract 0x250b4e38ac95a83a731af5e532823286dbdae1ff on Neo {self.network} using Skip={self.test_skip} and Limit={self.test_limit}. What NFT information can you find?"
            )

        # Scenario 7: NEP-11 Transfer Analysis by Address
        # Uses: GetNep11TransferByAddressTool
        if self.demo_address:
            await self.run_agent_scenario(
                'nep',
                "NEP-11 Transfer Analysis by Address",
                f"Use get_nep11_transfer_by_address tool to analyze NEP-11 (NFT) token transfers for address {self.demo_address} on Neo {self.network} using pagination (Skip={self.test_skip}, Limit={self.test_limit}). What NFT transfer activity can you find?"
            )

        # Scenario 8: NEP-11 Transfer Analysis by Block Height
        # Uses: GetNep11TransferByBlockHeightTool
        if self.test_block_height:
            await self.run_agent_scenario(
                'nep',
                "NEP-11 Transfer Analysis by Block Height",
                f"Use get_nep11_transfer_by_block_height tool to analyze NEP-11 (NFT) token transfers in block height {self.test_block_height} on Neo {self.network} using pagination (Skip={self.test_skip}, Limit={self.test_limit}). What NFT transfers occurred in this block?"
            )

        # Scenario 9: NEP-11 Transfer Analysis by Transaction Hash
        # Uses: GetNep11TransferByTransactionHashTool
        if self.test_tx_hash:
            await self.run_agent_scenario(
                'nep',
                "NEP-11 Transfer Analysis by Transaction Hash",
                f"Use get_nep11_transfer_by_transaction_hash tool to analyze NEP-11 (NFT) token transfer details for transaction hash 0xfc2207e7d9ac0662cf849ef139f59193a4cd40d4393192948d484f2a164463d8 on Neo {self.network}. What NFT transfer details can you find?"
            )

        # Scenario 10: NEP-11 Transfer Count by Address
        # Uses: GetNep11TransferCountByAddressTool
        if self.demo_address:
            await self.run_agent_scenario(
                'nep',
                "NEP-11 Transfer Count by Address",
                f"Use get_nep11_transfer_count_by_address tool to get the total number of NEP-11 (NFT) token transfers for address NVGUQ1qyL4SdSm7sVmGVkXetjEsvw2L3NT on Neo {self.network}. How many NFT transfers has this address been involved in?"
            )

        # Scenario 11: NEP-11 Transfer Count by Address - different address
        # Uses: GetNep11TransferCountByAddressTool
        if self.demo_address:
            await self.run_agent_scenario(
                'nep',
                "NEP-11 Transfer Count by Address",
                f"Use get_nep11_transfer_count_by_address tool to get the total number of NEP-11 (NFT) token transfers for address 0xfa03cb7b40072c69ca41f0ad3606a548f1d59966 on Neo {self.network}. How many NFT transfers has this address been involved in?"
            )

        # Note: GetNep11BalanceTool requires address, contract_hash, and token_id
        # This tool is more specific and may need a known NFT token ID to test properly
        if self.demo_address and self.demo_contract:
            await self.run_agent_scenario(
                'nep',
                "NEP-11 Balance",
                f"Use get_nep11_balance tool to get the NEP-11 balance for address NVGUQ1qyL4SdSm7sVmGVkXetjEsvw2L3NT, contract 0x250b4e38ac95a83a731af5e532823286dbdae1ff, and a specific token_id 8+wz0DmgIkXEzh8XgoZAhJD6XbJnUE5w1u9WRoETN3U= on Neo {self.network}."
            )

    async def demo_smart_contracts(self):
        """Demonstrate smart contract analysis capabilities"""
        self.print_section_header("6. Smart Contract Analysis with AI Agent")

        # Scenario 1: Contract Count
        # Uses: GetContractCountTool
        await self.run_agent_scenario(
            'contract',
            "Contract Ecosystem Overview",
            f"Use get_contract_count tool to get the total number of smart contracts deployed on Neo {self.network}. How many contracts are there?"
        )

        # Scenario 2: Contract by Hash
        # Uses: GetContractByHashTool 
        if self.demo_contract:
            await self.run_agent_scenario(
                'contract',
                "Contract Information by Hash",
                f"Use get_contract_by_hash tool to get detailed information for contract hash 0x73b148db3e2a58c5370d41592ffec0fb287d3fa4 on Neo {self.network}. What are the contract details?"
            )


        # Scenario 3: Contract List by Name
        # Uses: GetContractListByNameTool
        await self.run_agent_scenario(
            'contract',
            "Contract Search by Name",
            f"Use get_contract_list_by_name tool to search for contracts with name 'NEO' on Neo {self.network} using pagination (Skip={self.test_skip}, Limit={self.test_limit}). What contracts match this name?"
        )

        # Scenario 4: Verified Contracts List
        # Uses: GetVerifiedContractTool
        await self.run_agent_scenario(
            'contract',
            "Verified Contracts Overview",
            f"Use get_verified_contract tool to get all verified smart contracts on Neo {self.network} with pagination (Skip={self.test_skip}, Limit={self.test_limit}). What are the benefits of contract verification?"
        )

        # Scenario 5: Verified Contract by Hash
        # Uses: GetVerifiedContractByContractHashTool
        if self.demo_contract:
            await self.run_agent_scenario(    
                'contract',
                "Contract Verification Check",
                f"Use get_verified_contract_by_contract_hash tool to check if contract 0x73b148db3e2a58c5370d41592ffec0fb287d3fa4 is verified on Neo {self.network}. If yes, show me its verification details."
            )

    async def demo_smart_contract_calls(self):
        """Demonstrate smart contract call analysis capabilities"""
        self.print_section_header("7. Smart Contract Call Analysis with AI Agent")

        # Scenario 1: Contract Call History by Contract Hash
        # Uses: GetScCallByContractHashTool
        if self.demo_contract:
            await self.run_agent_scenario(
                'sc_call',
                "Contract Call History",
                f"Use get_sccall_by_contracthash tool to analyze smart contract call history for contract {self.demo_contract} on Neo {self.network} using pagination (Skip={self.test_skip}, Limit={self.test_limit}). What are the call patterns?"
            )

        # Scenario 2: Address-Contract Interactions
        # Uses: GetScCallByContractHashAddressTool
        if self.demo_contract and self.demo_address:
            await self.run_agent_scenario(
                'sc_call',
                "Address-Contract Interactions",
                f"Use get_sccall_by_contracthash_address tool to analyze contract calls from address {self.demo_address} to contract {self.demo_contract} on Neo {self.network} using pagination (Skip={self.test_skip}, Limit={self.test_limit}). What interactions occurred?"
            )

        # Scenario 3: Transaction Contract Calls
        # Uses: GetScCallByTransactionHashTool
        if self.test_tx_hash:
            await self.run_agent_scenario(
                'sc_call',
                "Transaction Contract Calls",
                f"Use get_sccall_by_transactionhash tool to analyze all contract calls in transaction {self.test_tx_hash} on Neo {self.network} with pagination (Skip={self.test_skip}, Limit={self.test_limit}). What contract methods were invoked and what were the call details?"
            )

    async def demo_governance_monitoring(self):
        """Demonstrate governance monitoring capabilities"""
        self.print_section_header("8. Governance Monitoring with AI Agent")

        # Scenario 1: Candidate Count
        # Uses: GetCandidateCountTool
        await self.run_agent_scenario(
            'governance',
            "Candidate Count Overview",
            f"Use get_candidate_count tool to get the total number of candidates in Neo {self.network}. How many candidates are participating in the consensus?"
        )

        # Scenario 2: Governance Overview
        # Uses: GetCandidateCountTool, GetTotalVotesTool, GetCommitteeInfoTool
        await self.run_agent_scenario(
            'governance',
            "Governance Overview",
            f"Give me an overview of Neo {self.network} governance. How many candidates and what's the voting situation? When retrieving lists, use Skip={self.test_skip} and Limit={self.test_limit}."
        )

        # Scenario 3: Committee Analysis
        # Uses: GetCommitteeInfoTool
        await self.run_agent_scenario(
            'governance',
            "Committee Analysis",
            f"Use get_committee_info tool to analyze the current Neo committee members on {self.network} using pagination (Skip={self.test_skip}, Limit={self.test_limit}). Who are the active committee members?"
        )

        # Scenario 4: Voting Statistics
        # Uses: GetTotalVotesTool
        await self.run_agent_scenario(
            'governance',
            "Voting Statistics",
            f"Use get_total_votes tool to analyze voting statistics on Neo {self.network}. What's the level of community participation?"
        )

        # Scenario 5: Candidate Deep Analysis
        # Uses: GetCandidateByAddressTool (needs address), GetVotersByCandidateAddressTool (needs address), GetVotesByCandidateAddressTool (needs address)
        # Note: All these tools require candidate address parameter
        if self.active_candidates:  
            candidate_address = self.active_candidates[0]
            await self.run_agent_scenario(
                'governance',
                "Candidate Deep Analysis",
                f"Use get_candidate_by_address tool with candidate address {candidate_address} to get candidate information. Then use the same address to call get_voters_by_candidate_address and get_votes_by_candidate_address tools with pagination Skip={self.test_skip} and Limit={self.test_limit}. Show me their detailed information, voters, and vote statistics."
            )

        # Scenario 6: Vote Calls by Candidate
        # Uses: GetScVoteCallByCandidateAddressTool
        if self.active_candidates:
            candidate_address = self.active_candidates[0]
            await self.run_agent_scenario(
                'governance',
                "Vote Calls by Candidate",
                f"Use get_sc_vote_call_by_candidate_address tool to get vote call records for candidate address {candidate_address} on Neo {self.network} with pagination (Skip={self.test_skip}, Limit={self.test_limit}). Show me their voting activity history."
            )

        # Scenario 7: Vote Call by Transaction Hash
        # Uses: GetScVoteCallByTransactionHashTool
        if self.test_tx_hash:
            await self.run_agent_scenario(
                'governance',
                "Vote Call Transaction Analysis",
                f"Use get_sc_vote_call_by_transaction_hash tool to analyze vote call details for transaction 0x4e4c101717e5d6d513168b0cea698100d3f2390010e96817b554f4c57859890b on Neo {self.network}. What are the voting details in this transaction?"
            )

        # Scenario 8: Voter Activity Analysis
        # Uses: GetCandidateByVoterAddressTool, GetScVoteCallByVoterAddressTool
        if self.demo_address:
            candidate_address = self.active_candidates[0]
            await self.run_agent_scenario(
                'governance',
                "Voter Activity Analysis",
                f"Use get_candidate_by_voter_address and get_sc_vote_call_by_voter_address tools to analyze voting activity for address {candidate_address} on Neo {self.network} with pagination (Skip={self.test_skip}, Limit={self.test_limit}). What candidates have they voted for? Show me their voting history."
            )

    async def demo_application_logs(self):
        """Demonstrate application log and state analysis capabilities"""
        self.print_section_header("9. Application Log Analysis with AI Agent")

        # Scenario 1: Transaction Log Analysis
        # Uses: GetApplicationLogTool
        if self.test_tx_hash:
            await self.run_agent_scenario(
                'logs',
                "Transaction Log Analysis",
                f"Use get_application_log tool to analyze the application execution logs for transaction {self.test_tx_hash} on Neo {self.network}. What smart contract operations occurred? Show me the execution details and any errors."
            )

        # Scenario 2: Block State Analysis
        # Uses: GetApplicationStateTool
        if self.test_block_hash:
            await self.run_agent_scenario(
                'logs',
                "Block State Analysis",
                f"Use get_application_state tool to analyze the application state for block {self.test_block_hash} on Neo {self.network} with pagination (Skip={self.test_skip}, Limit={self.test_limit}). What contract executions happened in this block? Show me all application logs."
            )

    async def run_comprehensive_demo(self):
        """Run the complete agent-based demonstration"""
        print("🚀 Neo Blockchain Toolkit - AI Agent Demonstration")
        print("=" * 80)
        print("This demo showcases Neo blockchain tools through specialized AI agents")
        print("Each agent is an expert in specific aspects of the Neo ecosystem")
        print("=" * 80)
        print(f" Network: {self.network}")
        print(f" Demo Address: {self.demo_address}")
        print(f" Demo Contract: {self.demo_contract}")
        print(f"  Test Data Available:")
        print(f"   - Addresses: {len(self.demo_addresses)}")
        print(f"   - Block Height: {self.test_block_height}")
        print(f"   - Transaction Hash: {'✅' if self.test_tx_hash else '❌'}")
        print(f"   - Active Candidates: {len(self.active_candidates)}")

        try:
            # Setup all specialized agents
            print("\n🔧 Setting up specialized agents...")
            self.setup_agents()
            print(f"✅ Created {len(self.agents)} specialized agents")

            # Run comprehensive demonstrations
            await self.demo_blockchain_exploration()
            await self.demo_address_analysis()
            await self.demo_asset_management()
            await self.demo_transaction_tracking()
            await self.demo_nep_tokens()
            await self.demo_smart_contracts()
            await self.demo_smart_contract_calls()
            await self.demo_governance_monitoring()
            await self.demo_application_logs()

            # Final summary
            self.print_section_header("Demo Completed Successfully")
            for agent_name, agent in self.agents.items():
                tool_count = len(agent.available_tools.tools)
                print(f"    {agent.agent_name}: {tool_count} specialized tools")

            total_tools = sum(len(agent.available_tools.tools) for agent in self.agents.values())
            print(f"\n🔧 Total Tools Demonstrated: {total_tools} out of 60 Neo tools")
            print(" All demonstrations powered by AI agents with domain expertise")
            print(" Each agent provides intelligent analysis and insights")
            print("   1. Blockchain Exploration (Block analysis, network status)")
            print("   2. Address Analysis (Profile, transactions, NFT holdings, portfolio, validation)")
            print("   3. Asset Management (Token discovery, portfolio analysis)")
            print("   4. Transaction Tracking (History, patterns, verification)")
            print("   5. NEP Token Analysis (NEP-17 & NEP-11 transfers)")
            print("   6. Smart Contract Analysis (Deployment, verification, state management)")
            print("   7. Smart Contract Call Analysis (Call patterns, interactions, invocation history)")
            print("   8. Governance Monitoring (Candidates, voting, committee)")
            print("   9. Application Logs (Execution logs, debugging)")

        except Exception as e:
            print(f"\n❌ Demo error: {str(e)}")
            print("Please check your environment setup and network connectivity")


async def main():
    """Main demonstration function"""
    print("\n Neo Blockchain Toolkit - AI Agent Demonstration")
    print("=" * 80)
    print("Showcasing comprehensive Neo blockchain analysis through specialized AI agents")
    print("Each agent is equipped with domain-specific tools and expertise")
    print("=" * 80)

    demo = NeoToolkitAgentDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
