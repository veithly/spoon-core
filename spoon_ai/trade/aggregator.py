import logging
import os
import time
from typing import Any, Dict, Optional

import requests
from web3 import Web3, HTTPProvider
from web3.middleware import ExtraDataToPOAMiddleware

from .abi import ERC20_ABI

logger = logging.getLogger(__name__)

class Aggregator:
    def __init__(self, network: str = "ethereum", rpc_url: str = None, scan_url: str = None, chain_id: int = 1):
        self.network = network
        self.rpc_url = rpc_url
        if not self.rpc_url:
            raise ValueError("rpc_url is required")
        self.scan_url = scan_url
        if not chain_id:
            chain_id = 1
        self.chain_id = chain_id
        for i in range(3):
            try:
                self._web3 = Web3(HTTPProvider(self.rpc_url))
                self._web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                if not self._web3.is_connected():
                    raise Exception("Failed to connect to RPC")
                chain_id = self._web3.eth.chain_id
                if str(chain_id) != str(self.chain_id):
                    raise Exception(f"Chain ID mismatch: {chain_id} != {self.chain_id}")
                return
            except Exception as e:
                logger.error(f"Failed to connect to {self.network} RPC: {e}")
                if i == 2:
                    raise e
                time.sleep(1)

    def _get_explorer_link(self, tx_hash: str) -> str:
        """Generate block explorer link for transaction"""
        return f"{self.scan_url}/tx/{tx_hash}"

    def get_native_token_address(self)->str:
        return "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"

    def get_token_info_by_address(self, token_address: str) -> Dict[str, Any]:
        """
        Get token information from contract address using CoinGecko API
        """
        try:
            if token_address.lower() == self.get_native_token_address().lower():
                # Handle native token
                native_id_map = {
                    "ethereum": "ethereum",
                    "bsc": "binancecoin",
                    "polygon": "matic-network",
                    "avalanche": "avalanche-2",
                    "fantom": "fantom"
                }
                
                coin_id = native_id_map.get(self.network)
                if coin_id:
                    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                    response = requests.get(url)
                    response.raise_for_status()
                    data = response.json()
                    
                    return {
                        "address": token_address,
                        "name": data.get("name", "Native Token"),
                        "symbol": data.get("symbol", "").upper(),
                        "decimals": 18,
                        "totalSupply": 0,
                        "network": self.network,
                        "chainId": self.chain_id,
                        "price_usd": data.get("market_data", {}).get("current_price", {}).get("usd"),
                        "image": data.get("image", {}).get("small")
                    }
                else:
                    return {
                        "address": token_address,
                        "name": "Native Token",
                        "symbol": self.network.upper(),
                        "decimals": 18,
                        "totalSupply": 0,
                        "network": self.network,
                        "chainId": self.chain_id
                    }
            
            # For ERC20 tokens
            # Convert network name to CoinGecko platform ID
            platform_map = {
                "ethereum": "ethereum",
                "bsc": "binance-smart-chain",
                "polygon": "polygon-pos",
                "avalanche": "avalanche",
                "fantom": "fantom"
            }
            
            platform = platform_map.get(self.network, self.network)
            
            # Get token info from CoinGecko
            url = f"https://api.coingecko.com/api/v3/coins/{platform}/contract/{token_address.lower()}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Get token decimals from contract if not provided by CoinGecko
            decimals = 18  # Default
            try:
                token_address_checksum = Web3.to_checksum_address(token_address)
                contract = self._web3.eth.contract(address=token_address_checksum, abi=ERC20_ABI)
                decimals = contract.functions.decimals().call()
            except Exception as e:
                logger.warning(f"Could not get decimals from contract: {e}")
            
            return {
                "address": token_address,
                "name": data.get("name", ""),
                "symbol": data.get("symbol", "").upper(),
                "decimals": decimals,
                "totalSupply": data.get("market_data", {}).get("total_supply"),
                "network": self.network,
                "chainId": self.chain_id,
                "price_usd": data.get("market_data", {}).get("current_price", {}).get("usd"),
                "market_cap": data.get("market_data", {}).get("market_cap", {}).get("usd"),
                "image": data.get("image", {}).get("small")
            }
        except Exception as e:
            logger.error(f"Error getting token info for {token_address}: {e}")
            return None
    
    def get_token_info_by_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Get token information from symbol using CoinGecko API
        """
        try:
            # Check if it's a native token
            native_symbols = {
                "ethereum": "ETH",
                "bsc": "BNB",
                "polygon": "MATIC",
                "avalanche": "AVAX",
                "fantom": "FTM"
            }
            
            if symbol.upper() == native_symbols.get(self.network, ""):
                return self.get_token_info_by_address(self.get_native_token_address())
            
            if not hasattr(self, "coins"):
                # Search for the token in CoinGecko
                url = "https://api.coingecko.com/api/v3/coins/list"
                response = requests.get(url)
                response.raise_for_status()
                self.coins = response.json()
            
            # Filter coins by symbol
            matching_coins = [coin for coin in self.coins if coin.get("symbol", "").lower() == symbol.lower()]
            
            if not matching_coins:
                logger.warning(f"No tokens found with symbol {symbol}")
                return None
            
            # Get contract addresses for each matching coin
            platform_map = {
                "ethereum": "ethereum",
                "bsc": "binance-smart-chain",
                "polygon": "polygon-pos",
                "avalanche": "avalanche",
                "fantom": "fantom"
            }
            
            
            platform = platform_map.get(self.network, self.network)
            
            for coin in matching_coins:
                coin_id = coin.get("id")
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                response = requests.get(url)
                
                if response.status_code != 200:
                    continue
                
                data = response.json()
                platforms = data.get("platforms", {})
                
                if platform in platforms and platforms[platform]:
                    token_address = platforms[platform]
                    return self.get_token_info_by_address(token_address)
            
            # If we couldn't find a token with matching symbol on the current network
            logger.warning(f"No token with symbol {symbol} found on {self.network}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting token info for symbol {symbol}: {e}")
            return None

    def get_balance(self, token_address: str = None) -> float:
        try:
            private_key = os.getenv("PRIVATE_KEY")
            if not private_key:
                raise ValueError("PRIVATE_KEY is not set")
            account = self._web3.eth.account.from_key(private_key)
            
            if token_address is None:
                # get balance of native token
                raw_balance = self._web3.eth.get_balance(account.address)
                return self._web3.from_wei(raw_balance, 'ether')
            
            contract = self._web3.eth.contract(address=Web3.to_checksum_address(token_address), abi=ERC20_ABI)
            decimals = contract.functions.decimals().call()
            raw_balance = contract.functions.balanceOf(account.address).call()
            return raw_balance / (10 ** decimals)
        except Exception as e:
            logger.error(f"Error getting balance for {token_address}: {e}")
            return 0.0
                

    def _prepare_transfer_tx(self, to_address: str, amount: float, token_address: Optional[str] = None)->Dict[str, Any]:
        private_key = os.getenv("PRIVATE_KEY")
        if not private_key:
            raise ValueError("PRIVATE_KEY is not set")
        account = self._web3.eth.account.from_key(private_key)
        
        nonce = self._web3.eth.get_transaction_count(account.address)
        gas_price = self._web3.eth.gas_price
        
        if token_address and token_address.lower() != self.get_native_token_address().lower():
            contract = self._web3.eth.contract(address=Web3.to_checksum_address(token_address), abi=ERC20_ABI)
            decimals = contract.functions.decimals().call()
            amount_raw = int(amount * (10 ** decimals))
            tx = contract.functions.transfer(Web3.to_checksum_address(to_address), amount_raw).build_transaction({
                'from': account.address,
                'nonce': nonce,
                'gas': 100000,
                'gasPrice': gas_price,
                'chainId': self.chain_id
            })
            tx['value'] = amount_raw
        else:
            tx = {
                'nonce': nonce,
                'to': Web3.to_checksum_address(to_address),
                'value': self._web3.to_wei(amount, 'ether'),
                'gas': 21000,  # Standard ETH transfer gas
                'gasPrice': gas_price,
                'chainId': self.chain_id
            }
        
        return tx

    def transfer(self, to_address: str, amount: float, token_address: Optional[str] = None)->str:
        try:
            current_balance = self.get_balance(token_address)
            if current_balance < amount:
                raise ValueError(f"Insufficient balance: {current_balance} {token_address}")
            
            tx = self._prepare_transfer_tx(to_address, amount, token_address)
            private_key = os.getenv("PRIVATE_KEY")
            account = self._web3.eth.account.from_key(private_key)
            signed_tx = self._web3.eth.account.sign_transaction(tx, account.key)
            tx_hash = self._web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            self._web3.eth.wait_for_transaction_receipt(tx_hash)
            return tx_hash.hex()
        except Exception as e:
            logger.error(f"Error transferring {amount} {token_address} to {to_address}: {e}")
            return None
    
    def _get_swap_route(self, token_in: str, token_out: str, amount: float, sender: str)->Optional[Dict[str, Any]]:
        url = f"https://aggregator-api.kyberswap.com/{self.network}/api/v1/routes"
        if token_in.lower() == self.get_native_token_address().lower():
            amount_raw = self._web3.to_wei(amount, 'ether')
        else:
            contract = self._web3.eth.contract(address=Web3.to_checksum_address(token_in), abi=ERC20_ABI)
            decimals = contract.functions.decimals().call()
            amount_raw = int(amount * (10 ** decimals))
        
        params = {
            "tokenIn": token_in,
            "tokenOut": token_out,
            "amountIn": str(amount_raw),
            "to": sender,
            "gasInclude": "true"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            raise Exception(f"Error getting swap route: {data.get('message')}")
        
        return data["data"]

    def _build_swap_tx(
        self,
        token_in: str,
        token_out: str,
        amount: float,
        slippage: float,
        route_data: Dict
    ) -> Dict[str, Any]:
        """Build swap transaction using route data"""
        try:
            private_key = os.getenv('PRIVATE_KEY')
            account = self._web3.eth.account.from_key(private_key)
            
            url = f"https://aggregator-api.kyberswap.com/{self.network}/api/v1/route/build"
            
            payload = {
                "routeSummary": route_data["routeSummary"],
                "sender": account.address,
                "recipient": account.address,
                "slippageTolerance": int(slippage * 100),  # Convert to bps
                "deadline": int(time.time() + 1200),  # 20 minutes
                "source": "zerepy"
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            if data.get("code") != 0:
                raise ValueError(f"API error: {data.get('message')}")
                
            # Prepare transaction parameters
            tx = {
                'from': account.address,
                'to': Web3.to_checksum_address(route_data["routerAddress"]),
                'data': data["data"]["data"],
                'value': self._web3.to_wei(amount, 'ether') if token_in.lower() == self.get_native_token_address().lower() else 0,
                'nonce': self._web3.eth.get_transaction_count(account.address),
                'gasPrice': self._web3.eth.gas_price,
                'chainId': self.chain_id
            }
            
            # Estimate gas
            try:
                gas_estimate = self._web3.eth.estimate_gas(tx)
                tx['gas'] = int(gas_estimate * 1.2)  # Add 20% buffer
            except Exception as e:
                logger.warning(f"Gas estimation failed: {e}, using default gas limit")
                tx['gas'] = 500000  # Default gas limit for swaps
                
            return tx
            
        except Exception as e:
            logger.error(f"Failed to build swap transaction: {str(e)}")
            raise

    def _handle_token_approval(
            self,
            token_address: str,
            spender_address: str,
            amount: int
        ) -> Optional[str]:
            """Handle token approval for spender, returns tx hash if approval needed"""
            try:
                private_key = os.getenv('PRIVATE_KEY')
                account = self._web3.eth.account.from_key(private_key)
                
                token_contract = self._web3.eth.contract(
                    address=Web3.to_checksum_address(token_address),
                    abi=ERC20_ABI
                )
                
                # Check current allowance
                current_allowance = token_contract.functions.allowance(
                    account.address,
                    spender_address
                ).call()
                
                if current_allowance < amount:
                    # Prepare approval transaction
                    approve_tx = token_contract.functions.approve(
                        spender_address,
                        amount
                    ).build_transaction({
                        'from': account.address,
                        'nonce': self._web3.eth.get_transaction_count(account.address),
                        'gasPrice': self._web3.eth.gas_price,
                        'chainId': self.chain_id
                    })
                    
                    # Estimate gas for approval
                    try:
                        gas_estimate = self._web3.eth.estimate_gas(approve_tx)
                        approve_tx['gas'] = int(gas_estimate * 1.1)  # Add 10% buffer
                    except Exception as e:
                        logger.warning(f"Approval gas estimation failed: {e}, using default")
                        approve_tx['gas'] = 100000  # Default gas for approvals
                    
                    # Sign and send approval transaction
                    signed_approve = account.sign_transaction(approve_tx)
                    tx_hash = self._web3.eth.send_raw_transaction(signed_approve.rawTransaction)
                    
                    # Wait for approval to be mined
                    receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash)
                    if receipt['status'] != 1:
                        raise ValueError("Token approval failed")
                    
                    return tx_hash.hex()
                    
                return None

            except Exception as e:
                logger.error(f"Token approval failed: {str(e)}")
                raise

    def swap(
        self,
        token_in: str,
        token_out: str,
        amount: float,
        slippage: float = 0.5
    ) -> str:
        """Execute token swap using Kyberswap aggregator"""
        try:
            private_key = os.getenv('PRIVATE_KEY')
            account = self._web3.eth.account.from_key(private_key)

            # Validate balance
            current_balance = self.get_balance(
                token_address=None if token_in.lower() == self.get_native_token_address().lower() else token_in
            )
            if current_balance < amount:
                raise ValueError(f"Insufficient balance. Required: {amount}, Available: {current_balance}")
            
            # Get optimal swap route
            route_data = self._get_swap_route(
                token_in,
                token_out,
                amount,
                account.address
            )
            
            # Handle token approval if needed
            if token_in.lower() != self.get_native_token_address().lower():
                router_address = route_data["routerAddress"]
                
                if token_in.lower() == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2".lower():  # WETH
                    amount_raw = self._web3.to_wei(amount, 'ether')
                else:
                    token_contract = self._web3.eth.contract(
                        address=Web3.to_checksum_address(token_in),
                        abi=ERC20_ABI
                    )
                    decimals = token_contract.functions.decimals().call()
                    amount_raw = int(amount * (10 ** decimals))
                    
                approval_hash = self._handle_token_approval(token_in, router_address, amount_raw)
                if approval_hash:
                    logger.info(f"Token approval transaction: {self._get_explorer_link(approval_hash)}")
            
            # Build and send swap transaction
            swap_tx = self._build_swap_tx(token_in, token_out, amount, slippage, route_data)
            signed_tx = account.sign_transaction(swap_tx)
            tx_hash = self._web3.eth.send_raw_transaction(signed_tx.rawTransaction)

            tx_url = self._get_explorer_link(tx_hash.hex())
            
            return (f"Swap transaction sent!(allow time for scanner to populate it):\n"
                    f"Transaction: {tx_url}")
                
        except Exception as e:
            return f"Swap failed: {str(e)}"
            