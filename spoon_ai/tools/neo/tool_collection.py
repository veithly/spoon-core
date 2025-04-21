from typing import Annotated, Callable, Any, Dict, List, TypeVar, get_type_hints
import functools

# Custom tool decorator to replace langchain's tool
def tool(func):
    """Simple tool decorator to replace langchain's tool decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    # Store original function metadata
    wrapper.__tool_name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = func.__annotations__
    
    return wrapper

import requests
import json
import os
import getpass


from decimal import *
from datetime import datetime

# from neo-mamba import 
import neo3.wallet.utils

# Define the api URL, should be put into a config file later
mainnet_url = "https://explorer.onegate.space/api"
testnet_url = "https://testmagnet.explorer.onegate.space/api"

url = testnet_url

if not os.environ.get("NCHAT_NETWORK"):
    os.environ["NCHAT_NETWORK"] = getpass.getpass("Choose network for nchat (mainnet or testnet): ")

if os.environ["NCHAT_NETWORK"] == "mainnet":
    url = mainnet_url

class AddressInfo:
    address: str
    firstusetime: str
    lastusetime: str
    transactionssent: int

    def __init__(self, address: str, firstusetime: int, lastusetime: int, transactionssent: int):
        """
        Custom initializer for the AddressInfo class.
        Args:
            address (str): The address string.
            firstusetime (int): Unix timestamp in milliseconds for the first use.
            lastusetime (int): Unix timestamp in milliseconds for the last use.
            transactionssent (int): The number of transactions sent.
        """
        if firstusetime > lastusetime:
            raise ValueError("firstusetime cannot be greater than lastusetime")
        # Perform initialization
        self.address = address
        self.firstusetime = datetime.fromtimestamp(firstusetime / 1000).strftime('%Y-%m-%d %H:%M:%S')
        self.lastusetime = datetime.fromtimestamp(lastusetime / 1000).strftime('%Y-%m-%d %H:%M:%S')
        self.transactionssent = transactionssent

def to_json(obj):
    return json.dumps(obj, default=lambda obj: obj.__dict__)

def convert_address_to_script_hash(address: str) -> str:
    if neo3.wallet.utils.is_valid_address(address):
        return "0x" + neo3.wallet.utils.address_to_script_hash(address=address).__str__()
    else:
        return address

# define custom tools
@tool
def getActiveAddresses(
    days: Annotated[int, "The number of the days in the past to get active addresses."]
) -> list[int]:
    """Gets the count of active addresses in the specified past days."""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetActiveAddresses",
        "params": {"Days": days},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data and isinstance(json_data["result"], list):
        active_addresses = [item["ActiveAddresses"] for item in json_data["result"] if "ActiveAddresses" in item]
        return active_addresses
    else:
        return (f"Error from server: {json_data['error']}")
    
@tool
def getAddressCount() -> int:
    """Gets the count of all addresses"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetAddressCount",
        "params": {},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        return json_data["result"]["total counts"]
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getAddressInfoByAddress(
    address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
) -> str:
    """
Gets the details of the given address. 
Returns the details in a JSON string.
The following fields of the address info are important:
- address: the address in script hash format
- firstusetime: when the address is first time used
- lastusetime: when the address is last time used
- transactionssent: the number of transactions this address has sent
"""
    scriptHash = convert_address_to_script_hash(address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetAddressInfoByAddress",
        "params": {"address": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        info = AddressInfo(
            address=result_data["address"],
            firstusetime=result_data["firstusetime"],
            lastusetime=result_data["lastusetime"],
            transactionssent=result_data["transactionssent"]
        )
        return to_json(info)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getApplicationLogByTransactionHash(
    tx_hash: Annotated[int, "The transaction hash string."]
) -> str:
    """Gets the application log by the given transaction hash."""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetApplicationLogByTransactionHash",
        "params": {"TransactionHash": tx_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()

    if "result" in json_data:
        result_data = json_data["result"] # todo
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getAssetCount() -> int:
    """Gets the count of all assets in Neo N3 blockchain system."""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetAssetCount",
        "params": {},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        return json_data["result"]["total counts"]
    else:
        return (f"Error from server: {json_data['error']}")
    
@tool
def getAssetInfoByHash(
    asset_hash: Annotated[str, "The asset contract script hash string."]
) -> str:
    """
Gets the asset information by the contact script hash.
Returns the result in a JSON string. 
The following fields of the result are important:
- "decimals": how many decimals should be used when representing the asset amount
- "firsttransfertime": timestamp when the asset is first time transfered
- "hash": the asset script hash
- "holders": the count of asset holders
- "ispopular": indicates if this asset is popular or not
- "symbol": the symbol of the asset
- "tokenname": the name of the asset
- "totalsupply": the total supplied amount of the asset
- "type": the type of the asset, "NEP17" for fungible token and "NEP11" for non-fungible token
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetAssetInfoByContractHash",
        "params": {"ContractHash": asset_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()

    if "result" in json_data:
        result_data = json_data["result"] # todo
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")
    
@tool
def getAssetInfoByName(
    asset_name: Annotated[str, "The contract script hash string."],
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
    Gets the asset information by the asset name (fuzzy search supported).
    Returns the result in a JSON string. 
    The following fields of the result are important:
    - "decimals": how many decimals should be used when representing the asset amount
    - "firsttransfertime": timestamp when the asset is first time transfered
    - "hash": the asset script hash
    - "holders": the count of asset holders
    - "ispopular": indicates if this asset is popular or not
    - "symbol": the symbol of the asset
    - "tokenname": the name of the asset
    - "totalsupply": the total supplied amount of the asset
    - "type": the type of the asset, "NEP17" for fungible token and "NEP11" for non-fungible token
    """
    payload = {
        "jsonrpc": "2.0",
        "method": "GetAssetInfosByName",
        "params": {"Name": asset_name},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()

    if "result" in json_data:
        result_data = json_data["result"] # todo
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getAssetsInfoByUserAddress(
    user_address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
) -> str:
    """
Gets the information of assets owned by the user's address.
Returns the result in a JSON string. 
The following fields for each asset are important:
- "asset": the asset script hash
- "balance": The balance owned by the user address. The balance is represented in a big integer which is the result of the true balance number multiplied by 10 to the "asset's decimal"th power.
"""
    scriptHash = convert_address_to_script_hash(user_address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetAssetsHeldByAddress",
        "params": {"Address": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"] # todo
        # for item in result_data["result"]:
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")
    
@tool
def getAssetInfoByAssetAndAddress(
    asset_hash: Annotated[str, "The asset contract script hash string."],
    user_address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
) -> str:
    """
Gets the information of one specific asset owned by the user's address.
Returns the result in a JSON string. 
The following fields for the asset are important:
- "asset": the asset script hash
- "balance": The balance owned by the user address. The balance is represented in a big integer which is the result of the true balance number multiplied by 10 to the "asset's decimal"th power.
"""
    # - "balance": The balance owned by the user address. The balance is represented in a big integer which is the result of the true balance number multiplied by 10 to the "asset's decimal"th power.
    
    scriptHash = convert_address_to_script_hash(user_address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetAssetsHeldByContractHashAddress",
        "params": {
            "ContractHash": asset_hash,
            "Address": scriptHash
        },
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"] # todo
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getBestBlockHash(
) -> str:
    """Gets the latest (best) block hash from Neo N3 blockchain system."""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetBestBlockHash",
        "params": {},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()

    if "result" in json_data :
        return json_data["result"]["hash"]
    else:
        return (f"Error from server: {json_data['error']}")
    
@tool
def getBlockByHash(
    block_hash: Annotated[str, "The block hash, for example: 0x7688cf2521bbb5274c22363350539f402e4614a015d9e62b63694c049dec89d6"]
) -> int:
    """
Gets the details of the block by its hash.
Returns the details in a JSON string.
The following fields of the block info are important:
- "version": block version, current is 0
- "prevhash": the previous block's hash
- "merkleroot": the merkle tree root of the block's transactions
- "timestamp": the timestamp when this block is generated
- "nonce": the random number of the block
- "index": the block height, and the Genesis Block's index is 0
- "primary": the index of the proposal validator in the current round
- "nextconsensus": the script hash of more than two-thirds of validator's signatures for the next round
- "witnesses": the verification script of the block, it contains InvocationScript and VerificationScript. The InvocationScript provides the parameters for the VerificationScript to execute.
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetBlockByBlockHash",
        "params": {"BlockHash": block_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")
    
@tool
def getBlockByHeight(
    block_height: Annotated[int, "The block height (index), for example: 3823"]
) -> int:
    """
Gets the details of the block by its height (index).
Returns the details in a JSON string.
The following fields of the block info are important:
- "version": block version, current is 0
- "prevhash": the previous block's hash
- "merkleroot": the merkle tree root of the block's transactions
- "timestamp": the timestamp when this block is generated
- "nonce": the random number of the block
- "index": the block height, and the Genesis Block's index is 0
- "primary": the index of the proposal validator in the current round
- "nextconsensus": the script hash of more than two-thirds of validator's signatures for the next round
- "witnesses": the verification script of the block, it contains InvocationScript and VerificationScript. The InvocationScript provides the parameters for the VerificationScript to execute.
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetBlockByBlockHeight",
        "params": {"BlockHeight": block_height},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getBlockCount() -> int:
    """Gets the count of all blocks in Neo N3 blockchain system."""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetBlockCount",
        "params": {},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        return json_data["result"]["index"]
    else:
        return (f"Error from server: {json_data['error']}")
    
@tool
def getRecentBlocksInfo(
    # limit: Annotated[int, "Optional. The number of blocks to return."],
    # skip: Annotated[int, "Optional. The number of blocks to skip."]
) -> str:
    """
Gets the block information of the recent blocks.
Returns the result in a JSON string. 
The following fields of the result are important:
- "hash": the block hash
- "index": the block index 
- "size": the size of the block in bytes
- "timestamp": the timestamp when this block is generated
- "transactioncount": the count of transactions contained in this block
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetBlockInfoList",
        "params": {"Limit":10},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getBlockRewardByHash(
    block_hash: Annotated[str, "The block hash, for example: 0x7688cf2521bbb5274c22363350539f402e4614a015d9e62b63694c049dec89d6"]
) -> str:
    """
Gets the block reward details by the block hash.
The following fields of the result are important:
- "hash": the block hash
- "index": the block index 
- "size": the size of the block in bytes
- "timestamp": the timestamp when this block is generated
- "transactioncount": the count of transactions contained in this block
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetBlockRewardByBlockHash",
        "params": {"BlockHash": block_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getCandidateByAddress(
    address: Annotated[str, "Neo N3 address of the candidate, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
) -> str:
    """
Gets the candidate information by the candidate address.
Returns the details in a JSON string.
The following fields of the block info are important:
- "candidate": the script hash of the candidate
- "isCommittee": indicates if this candidate is selected into the committee
- "state": indicates the current state of the candidate
- "votesOfCandidate": the vote count of this committee
"""
    scriptHash = convert_address_to_script_hash(address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetCandidateByAddress",
        "params": {"Address": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data :
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getCandidateByVoterAddress(
    voter_address: Annotated[str, "Neo N3 address of the voter, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
) -> str:
    """
Gets the information of the candidate voted by the voter address.
Returns the details in a JSON string.
The following fields of the block info are important:
- "balanceOfVoter": the vote count voted to the candidate by the voter
- "blockNumber": the height of the block which the vote transaction is in
- "candidate": the candidate address (script hash)
- "candidatePubKey": the public key of the candidate
- "lastVoteTxid": the last vote transaction hash of the voter
- "voter": the voter address (script hash)
"""
    scriptHash = convert_address_to_script_hash(voter_address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetCandidateByVoterAddress",
        "params": {"VoterAddress": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data :
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getCandidateCount() -> int:
    """Gets the count of all addresses"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetCandidateCount",
        "params": {},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        return json_data["result"]["total counts"]
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getCommittee() -> str:
    """
Gets all the current committee members in the Neo N3 blockchain system.
Returns the details in a JSON string.
The following fields of the block info are important:
- "candidate": the address (script hash) of the committee member
- "isCommittee": indicates if this candidate is selected into the committee
- "state": indicates the current state of the candidate
- "votesOfCandidate": the vote count of this committee
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetCommittee",
        "params": {},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data :
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getContractByHash(
    contract_hash: Annotated[str, "Neo N3 contract script hash, for example: 0xcc5e4edd9f5f8dba8bb65734541df7a1c081c67b"]
) -> str:
    """
Gets the contract information by the contract script hash.
Returns the details in a JSON string.
The following fields of the contract info are important:
- "createTxid": the hash of the transaction that creates this contract
- "createtime": the timestamp when the contract is created
- "manifest": the content of the manifest file which explicitly declares the contract functions and permissions
- "name": the contract name
- "nef": the content of the NEF (Neo Executable Format) file
- "sender": the sender address who sends the transaction to create the contract
- "totalsccall": total count of calls to this contract
- "updatecounter": total count of updates to this contract
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetContractByContractHash",
        "params": {"ContractHash": contract_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data :
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getContractCount() -> int:
    """Gets the count of all contracts deployed in the Neo N3 blockchain system."""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetAddressCount",
        "params": {},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        return json_data["result"]["total counts"]
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getContractListByName(
    contract_name: Annotated[str, "The contract name."],
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets the contract list by the given name (fuzzy search supported)
Returns the result in a JSON string. 
The following fields of the result are important:
- "sender": the sender address who sends the transaction to create the contract
- "createtime": the timestamp when the contract is created
- "hash": the contract script hash
- "id": the contract id number
- "name": the contract name
- "updatecounter": total count of updates to this contract
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetContractListByName",
        "params": {"Name": contract_name},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

# @tool
# def getCumulativeFeeBurn()

@tool
def getDailyTransactions(
    days: Annotated[int, "The number of the days in the past to get transactions."]
) -> list[int]:
    """Gets the count of transactions in the specified past days."""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetDailyTransactions",
        "params": {"Days": days},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data and isinstance(json_data["result"], list):
        dayly_txs = [item["DailyTransactions"] for item in json_data["result"] if "DailyTransactions" in item]
        return dayly_txs
    else:
        return (f"Error from server: {json_data['error']}")

# def getExecutionByBlockHash
# def getExecutionByTransactionHash
# def getExecutionByTrigger
# def getExtraTransferByBlockHash

@tool
def getHourlyTransactions(
    hours: Annotated[int, "The count of the hours in the past to get transactions."]
) -> list[int]:
    """Gets the count of transactions in the specified past hours."""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetHourlyTransactions",
        "params": {"Hours": hours},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data and isinstance(json_data["result"], list):
        hourly_txs = [item["HourlyTransactions"] for item in json_data["result"] if "HourlyTransactions" in item]
        return hourly_txs
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getNep11Balance(
    contract_hash: Annotated[str, "Neo N3 contract script hash, for example: 0xcc5e4edd9f5f8dba8bb65734541df7a1c081c67b"],
    address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"],
    token_id: Annotated[str, "The token id of the NFT, represented in base64 format, for example: QmxpbmQgQm94IDIxNQ=="]
) -> list[int]:
    """
NEP11 is the NFT (Non-fungible Token) standard of Neo N3 blockchain system.
Gets the NFT balance by the contract script hash, user's address (script hash), and token Id.
Returns the details in a JSON string.
The following fields of the result are important:
- "asset": the NFT asset contract script hash
- "balance": the balance of this NFT in the address
"""
    scriptHash = convert_address_to_script_hash(address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetNep11BalanceByContractHashAddressTokenId",
        "params": {
            "ContractHash": contract_hash,
            "Address": scriptHash,
            "tokenId": token_id
        },
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getNep11OwnedByAddress(
    address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"],
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
NEP11 is the NFT (Non-fungible Token) standard of Neo N3 blockchain system.
Gets all the NFT owned by the user's address.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockhash": the hash of the block where the user gets the NFT
- "contract": the NFT contract script hash
- "from": the address where this NFT is transfered from, if it's null, the NFT is newly minted
- "frombalance": the remaining balance of this NFT in the "from" address after this transfer
- "timestamp": the timestamp when this NFT is owned by this address
- "to": the address where this NFT is transfered to
- "tobalance": the remaining balance of this NFT in the "to" address after this transfer
- "tokenId": the token id of the NFT, represented in base64 format
- "txid": the hash of the transaction that transfers this NFT
- "value": the count transfered of this NFT
"""
    scriptHash = convert_address_to_script_hash(address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetNep11OwnedByAddress",
        "params": {"Address": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"] # todo
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getNep11ByAddressAndHash(
    address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"],
    contract_hash: Annotated[str, "Neo N3 contract script hash, for example: 0xcc5e4edd9f5f8dba8bb65734541df7a1c081c67b"]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
NEP11 is the NFT (Non-fungible Token) standard of Neo N3 blockchain system.
Gets all the NFT owned by the user's address and the NFT contract script hash.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockhash": the hash of the block where the user gets the NFT
- "contract": the NFT contract script hash
- "from": the address where this NFT is transfered from, if it's null, the NFT is newly minted
- "frombalance": the remaining balance of this NFT in the "from" address after this transfer
- "timestamp": the timestamp when this NFT is owned by this address
- "to": the address where this NFT is transfered to
- "tobalance": the remaining balance of this NFT in the "to" address after this transfer
- "tokenId": the token id of the NFT, represented in base64 format
- "txid": the hash of the transaction that transfers this NFT
- "value": the count transfered of this NFT
"""
    scriptHash = convert_address_to_script_hash(address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetNep11OwnedByContractHashAddress",
        "params": {
            "Address": scriptHash,
            "ContractHash": contract_hash
        },
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getNep11TransferByAddress(
    address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"],
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets all the NFT transfer record by the user's address.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockhash": the hash of the block where the user gets the NFT
- "contract": the NFT contract script hash
- "from": the address where this NFT is transfered from, if it's null, the NFT is newly minted
- "frombalance": the remaining balance of this NFT in the "from" address after this transfer
- "netfee": the network fee of this transaction
- "sysfee": the system fee of this transaction
- "timestamp": the timestamp when this NFT is owned by this address
- "to": the address where this NFT is transfered to
- "tobalance": the remaining balance of this NFT in the "to" address after this transfer
- "tokenId": the token id of the NFT, represented in base64 format
- "txid": the hash of the transaction that transfers this NFT
- "value": the count transfered of this NFT
- "vmstate": the state of the Neo N3 blockchain virtual machine
"""
    scriptHash = convert_address_to_script_hash(address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetNep11TransferByAddress",
        "params": {
            "Address": scriptHash
        },
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getNep11TransferByBlockHeight(
    block_height: Annotated[int, "The block height (index), for example: 3823"]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets all the NFT transfer information in the specified block height.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockhash": the hash of the block where the user gets the NFT
- "contract": the NFT contract script hash
- "from": the address where this NFT is transfered from, if it's null, the NFT is newly minted
- "frombalance": the remaining balance of this NFT in the "from" address after this transfer
- "netfee": the network fee of this transaction
- "sysfee": the system fee of this transaction
- "timestamp": the timestamp when this NFT is owned by this address
- "to": the address where this NFT is transfered to
- "tobalance": the remaining balance of this NFT in the "to" address after this transfer
- "tokenId": the token id of the NFT, represented in base64 format
- "txid": the hash of the transaction that transfers this NFT
- "value": the count transfered of this NFT
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetNep11TransferByBlockHeight",
        "params": {
            "BlockHeight": block_height
        },
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getNep11TransferByTransactionHash(
    tx_hash: Annotated[int, "The transaction hash string."]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets all the NFT transfer information by the specified transaction hash.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockhash": the hash of the block where the user gets the NFT
- "contract": the NFT contract script hash
- "from": the address where this NFT is transfered from, if it's null, the NFT is newly minted
- "frombalance": the remaining balance of this NFT in the "from" address after this transfer
- "netfee": the network fee of this transaction
- "sysfee": the system fee of this transaction
- "timestamp": the timestamp when this NFT is owned by this address
- "to": the address where this NFT is transfered to
- "tobalance": the remaining balance of this NFT in the "to" address after this transfer
- "tokenId": the token id of the NFT, represented in base64 format
- "txid": the hash of the transaction that transfers this NFT
- "value": the count transfered of this NFT
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetNep11TransferByTransactionHash",
        "params": {
            "TransactionHash": tx_hash
        },
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getNep11TransferCountByAddress(
    address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"],
) -> int:
    """
NEP11 is the NFT (Non-fungible Token) standard of Neo N3 blockchain system.
Gets the count of all NFT transfers of this address.
"""
    scriptHash = convert_address_to_script_hash(address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetNep11TransferCountByAddress",
        "params": {"Address": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        return json_data["result"]
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getNep17TransferByAddress(
    address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"],
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
NEP17 is the fungible token standard of Neo N3 blockchain system.
Gets all the token transfer information by the user's address.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockhash": the hash of the block where the user gets the token
- "contract": the token contract script hash
- "from": the address where this token is transfered from, if it's null, the token is newly minted
- "frombalance": the remaining balance of this token in the "from" address after this transfer
- "netfee": the network fee of this transaction
- "sysfee": the system fee of this transaction
- "timestamp": the timestamp when this token is owned by this address
- "to": the address where this token is transfered to
- "tobalance": the remaining balance of this token in the "to" address after this transfer
- "tokenId": the token id of the token, represented in base64 format
- "txid": the hash of the transaction that transfers this token
- "value": the count transfered of this token
"""
    scriptHash = convert_address_to_script_hash(address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetNep11OwnedByAddress",
        "params": {"Address": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"] # todo
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getNep17TransferByBlockHeight(
    block_height: Annotated[int, "The block height (index), for example: 3823"]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
NEP17 is the fungible token standard of Neo N3 blockchain system.
Gets all the token transfer information by the specified block height.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockhash": the hash of the block where the user gets the token
- "contract": the token contract script hash
- "from": the address where this token is transfered from, if it's null, the token is newly minted
- "frombalance": the remaining balance of this token in the "from" address after this transfer
- "timestamp": the timestamp when this token is owned by this address
- "to": the address where this token is transfered to
- "tobalance": the remaining balance of this token in the "to" address after this transfer
- "tokenId": the token id of the token, represented in base64 format
- "txid": the hash of the transaction that transfers this token
- "value": the count transfered of this token
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetNep17TransferByBlockHeight",
        "params": {"BlockHeight": block_height},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"] # todo
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getNep17TransferByContractHash(
    contract_hash: Annotated[str, "Neo N3 contract script hash, for example: 0xcc5e4edd9f5f8dba8bb65734541df7a1c081c67b"]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets all the token transfer information by the specified contract hash.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockhash": the hash of the block where the user gets the token
- "contract": the token contract script hash
- "from": the address where this token is transfered from, if it's null, the token is newly minted
- "frombalance": the remaining balance of this token in the "from" address after this transfer
- "timestamp": the timestamp when this token is owned by this address
- "to": the address where this token is transfered to
- "tobalance": the remaining balance of this token in the "to" address after this transfer
- "tokenId": the token id of the token, represented in base64 format
- "txid": the hash of the transaction that transfers this token
- "value": the count transfered of this token
- "vmstate": the state of the Neo N3 blockchain virtual machine after this transaction
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetNep17TransferByContractHash",
        "params": {"ContractHash": contract_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"] # todo
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getNep17TransferByTransactionHash(
    tx_hash: Annotated[int, "The transaction hash string."]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets all the token transfer information by the specified transaction hash.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockhash": the hash of the block where the transaction is in
- "contract": the token contract script hash
- "decimals": how many decimals should be used when representing the asset amount
- "from": the address where this token is transfered from, the token is newly minted if null
- "symbol": the symbol of the asset
- "frombalance": the remaining balance of this token in the "from" address after this transfer
- "timestamp": the timestamp when this token is owned by this address
- "to": the address where this token is transfered to
- "tobalance": the remaining balance of this token in the "to" address after this transfer
- "tokenname": the name of the asset
- "txid": the hash of the transaction that transfers this token
- "value": the count transfered of this token
- "vmstate": the state of the Neo N3 blockchain virtual machine
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetNep17TransferByTransactionHash",
        "params": {"TransactionHash": tx_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"] # todo
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getNep17TransferCountByAddress(
    address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"],
) -> int:
    """
NEP17 is the fungible token standard of Neo N3 blockchain system.
Gets the count of all token transfers of this address.
"""
    scriptHash = convert_address_to_script_hash(address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetNep17TransferCountByAddress",
        "params": {"Address": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        return json_data["result"]
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getNetFeeRange() -> str:
    """
Gets the range of network fee for Neo N3 blockchain system.
Returns the result in a JSON string. 
The following fields of the result are important:
- "fast": the network fee cost when the transaction is handled fast
- "fastest": the network fee cost when the transaction is handled fastest
- "slow": the network fee cost when the transaction is handled slow
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetNetFeeRange",
        "params": {},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        return json_data["result"]
    else:
        return (f"Error from server: {json_data['error']}")


# getNewAddresses
# getNotificationByContractHash
# getNotificationByEvent

@tool
def getPopularToken(
    standard: Annotated[str, "The token standard of the asset: NEP11 or NEP17"]
) -> str:
    """
Gets the current popular tokens by the given standard.
Returns the result in a JSON string. 
The following fields of the result are important:
- "decimals": how many decimals should be used when representing the asset amount
- "hash": the asset script hash
- "symbol": the symbol of the asset
- "tokenname": the name of the asset
- "totalsupply": the total supplied amount of the asset
- "type": the standard, NEP11 or NEP17
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetPopularToken",
        "params": {"Standard": standard},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        return json_data["result"]
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getRawMempool() -> list[str]:
    """Gets the hashes of transactions in the memory pool."""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetRawMemPool",
        "params": {},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data and isinstance(json_data["result"], list):
        return json_data["result"]
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getRawTransactionByAddress(
    address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets the raw transactions by user's address.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockIndex": the index of the block which the transaction is in
- "blockhash": the hash of the block which the transaction is in
- "blocktime": the timestamp when this block is created
- "hash": the hash of the transaction
- "netfee": the network fee of this transaction
- "nonce": the nonce of the transaction
- "script": the script (in base64 format) of the transaction
- "sender": the address which sends the transaction
- "signers": the signers of the transaction
- "size": the size in bytes of the transaction
- "sysfee": the system fee of this transaction
- "validUntilBlock": the index of the block which the transaction is valid until
- "version": the transaction version
- "vmstate": the state of the virtual machine after running this transaction script
- "witnesses": the verification script of the transaction
"""
    scriptHash = convert_address_to_script_hash(address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetRawTransactionByAddress",
        "params": {"Address": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

    
@tool
def getRawTransactionByBlockHash(
    block_hash: Annotated[str, "The block hash, for example: 0x7688cf2521bbb5274c22363350539f402e4614a015d9e62b63694c049dec89d6"]
) -> str:
    """
Gets the raw transactions by the specific block hash.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockIndex": the index of the block which the transaction is in
- "blockhash": the hash of the block which the transaction is in
- "blocktime": the timestamp when this block is created
- "hash": the hash of the transaction
- "netfee": the network fee of this transaction
- "nonce": the nonce of the transaction
- "script": the script (in base64 format) of the transaction
- "sender": the address which sends the transaction
- "signers": the signers of the transaction
- "size": the size in bytes of the transaction
- "sysfee": the system fee of this transaction
- "validUntilBlock": the index of the block which the transaction is valid until
- "version": the transaction version
- "vmstate": the state of the virtual machine after running this transaction script
- "witnesses": the verification script of the transaction
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetRawTransactionByBlockHash",
        "params": {"BlockHash": block_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getRawTransactionByBlockHeight(
    block_height: Annotated[int, "The block height (index), for example: 3823"]
) -> str:
    """
Gets the raw transactions by the specific block height.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockIndex": the index of the block which the transaction is in
- "blockhash": the hash of the block which the transaction is in
- "blocktime": the timestamp when this block is created
- "hash": the hash of the transaction
- "netfee": the network fee of this transaction
- "nonce": the nonce of the transaction
- "script": the script (in base64 format) of the transaction
- "sender": the address which sends the transaction
- "signers": the signers of the transaction
- "size": the size in bytes of the transaction
- "sysfee": the system fee of this transaction
- "validUntilBlock": the index of the block which the transaction is valid until
- "version": the transaction version
- "vmstate": the state of the virtual machine after running this transaction script
- "witnesses": the verification script of the transaction
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetRawTransactionByBlockHeight",
        "params": {"BlockHash": block_height},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")
    
@tool
def getRawTransactionByTransactionHash(
    tx_hash: Annotated[str, "The transaction hash, for example: 0x85b55479fc43668077821234f547824d3111343aec21988f8c0aa1ff9b2ee287"]
) -> str:
    """
Gets the raw transactions by the specific transaction hash.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockIndex": the index of the block which the transaction is in
- "blockhash": the hash of the block which the transaction is in
- "blocktime": the timestamp when this block is created
- "hash": the hash of the transaction
- "netfee": the network fee of this transaction
- "nonce": the nonce of the transaction
- "script": the script (in base64 format) of the transaction
- "sender": the address which sends the transaction
- "signers": the signers of the transaction
- "size": the size in bytes of the transaction
- "sysfee": the system fee of this transaction
- "validUntilBlock": the index of the block which the transaction is valid until
- "version": the transaction version
- "vmstate": the state of the virtual machine after running this transaction script
- "witnesses": the verification script of the transaction
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetRawTransactionByTransactionHash",
        "params": {"TransactionHash": tx_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()

    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getScCallByContractHash(
    contract_hash: Annotated[str, "Neo N3 contract script hash, for example: 0xcc5e4edd9f5f8dba8bb65734541df7a1c081c67b"]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets the smart contract call information by the contract hash.
Returns the result in a JSON string. 
The following fields of the result are important:
- "callFlags": an enum defines special behaviors allowed when invoking smart contracts, such as chain calls, sending notifications, modifying states, etc.
- "contractHash": the hash of the invoked contract
- "hexStringParams": the params passed when the contract is invoked
- "method": the specific method called of the contract
- "originSender": the address which sends the transaction to call this contract
- "stack": the state of the virtual machine after calling this contract
- "txid": the hash of the transaction
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetScCallByContractHash",
        "params": {"ContractHash": contract_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getScCallByContractHashAddress(
    contract_hash: Annotated[str, "Neo N3 contract script hash, for example: 0xcc5e4edd9f5f8dba8bb65734541df7a1c081c67b"],
    user_address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets the smart contract call information by the contract hash and the user's address.
Returns the result in a JSON string. 
The following fields of the result are important:
- "callFlags": an enum defines special behaviors allowed when invoking smart contracts, such as chain calls, sending notifications, modifying states, etc.
- "contractHash": the hash of the invoked contract
- "hexStringParams": the params passed when the contract is invoked
- "method": the specific method called of the contract
- "originSender": the address which sends the transaction to call this contract
- "stack": the state of the virtual machine after calling this contract
- "txid": the hash of the transaction
"""
    scriptHash = convert_address_to_script_hash(user_address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetScCallByContractHashAddress",
        "params": {
            "ContractHash": contract_hash,
            "Address": scriptHash
        },
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")
    
@tool
def getScCallByTransactionHash(
    tx_hash: Annotated[str, "The transaction hash, for example: 0x85b55479fc43668077821234f547824d3111343aec21988f8c0aa1ff9b2ee287"]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets the smart contract call information by the transaction hash.
Returns the result in a JSON string. 
The following fields of the result are important:
- "callFlags": an enum defines special behaviors allowed when invoking smart contracts, such as chain calls, sending notifications, modifying states, etc.
- "contractHash": the hash of the invoked contract
- "hexStringParams": the params passed when the contract is invoked
- "method": the specific method called of the contract
- "originSender": the address which sends the transaction to call this contract
- "stack": the state of the virtual machine after calling this contract
- "txid": the hash of the transaction
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetScCallByTransactionHash",
        "params": {"TransactionHash": tx_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getScVoteCallByCandidateAddress(
    candidate_address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets the call information to the voting smart contract by the candidate's address.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockNumber": the index of the block which the voting transaction is in
- "candidate": the candidate's address (script hash)
- "candidatePubKey": the public key of the candidate
- "txid": the hash of the transaction
- "voter": the voter's address (script hash)
"""
    scriptHash = convert_address_to_script_hash(candidate_address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetScVoteCallByCandidateAddress",
        "params": {"CandidateAddress": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getScVoteCallByTransactionHash(
    tx_hash: Annotated[str, "The transaction hash, for example: 0x85b55479fc43668077821234f547824d3111343aec21988f8c0aa1ff9b2ee287"]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets the call information to the voting smart contract by the transaction hash.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockNumber": the index of the block which the voting transaction is in
- "candidate": the candidate's address (script hash)
- "candidatePubKey": the public key of the candidate
- "txid": the hash of the transaction
- "voter": the voter's address (script hash)
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetScVoteCallByTransactionHash",
        "params": {"TransactionHash": tx_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getScVoteCallByVoterAddress(
    voter_address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets the call information to the voting smart contract by the voter's address.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockNumber": the index of the block which the voting transaction is in
- "candidate": the candidate's address (script hash)
- "candidatePubKey": the public key of the candidate
- "txid": the hash of the transaction
- "voter": the voter's address (script hash)
"""
    scriptHash = convert_address_to_script_hash(voter_address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetScVoteCallByVoterAddress",
        "params": {"VoterAddress": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getSourceCodeByContractHash(
    contract_hash: Annotated[str, "Neo N3 contract script hash, for example: 0xcc5e4edd9f5f8dba8bb65734541df7a1c081c67b"]
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets the smart contract source code information by the contract hash.
Returns the result in a JSON string. 
The following fields of the result are important:
- "code": the source code of the contract
- "filename": the name of the contract file
- "hash": the hash of the contract
- "updatecounter": total count of updates to this contract
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetSourceCodeByContractHash",
        "params": {"ContractHash": contract_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getTagByAddresses(
    addresses: Annotated[list[str], "A list of Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
) -> str:
    """
Gets the tag information by the address in batches.
Returns the result in a JSON string.
The following fields of the tag info are important:
- "address": the user address in script hash format
- "bneoSum": the total bNEO amount of this address
- "ft_tag": the fungible token tag of this user's address
- "ft_total": the fungible token total amount of this address
- "neoSum": the total NEO amount of this address
- "nft_tag": the non-fungible token tag of this user's address
- "ntf_total": the non-fungible token total amount of this address
"""
    for i in range(0, len(addresses)): 
        addresses[i] = convert_address_to_script_hash(address=addresses[i])
    payload = {
        "jsonrpc": "2.0",
        "method": "GetTagByAddresses",
        "params": {"Address": addresses},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getTotalSentAndReceivedByContractHashAddress(
    contract_hash: Annotated[str, "Neo N3 contract script hash, for example: 0xcc5e4edd9f5f8dba8bb65734541df7a1c081c67b"],
    user_address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
) -> str:
    """
Gets the total sent and received amount by the token contract hash and the user's address.
Returns the result in a JSON string. 
The following fields of the result are important:
- "Address": the user address in script hash format
- "ContractHash": the hash of the invoked contract
- "received": the received token amount of this address
- "sent": the sent token amount of this address
"""
    scriptHash = convert_address_to_script_hash(user_address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetTotalSentAndReceivedByContractHashAddress",
        "params": {
            "ContractHash": contract_hash,
            "Address": scriptHash
        },
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getTotalVotes() -> int:
    """Gets the total votes of all candidates."""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetTotalVotes",
        "params": {},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        return json_data["result"]["totalvotes"]
    else:
        return (f"Error from server: {json_data['error']}")
    
@tool
def getTransactionCount() -> int:
    """Gets the number of all transactions executed in the Neo N3 blockchain."""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetTransactionCount",
        "params": {},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        return json_data["result"]["total counts"]
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getTransactionCountByAddress(
    user_address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
) -> int:
    """Gets the transactions count by the given user's address."""
    scriptHash = convert_address_to_script_hash(user_address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetTransactionCount",
        "params": {"Address": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        return json_data["result"]["total counts"]
    else:
        return (f"Error from server: {json_data['error']}")


# getTransactionList

@tool
def getTransferByAddress(
    address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"],
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets all the token transfer record by the user's address.
Returns the result in a JSON string. 
The following fields of the result are important:
- "blockhash": the hash of the block where the user gets the token
- "contract": the token contract script hash
- "from": the address where this token is transfered from, if it's null, the token is newly minted
- "frombalance": the remaining balance of this token in the "from" address after this transfer
- "timestamp": the timestamp when this token is owned by this address
- "to": the address where this token is transfered to
- "tobalance": the remaining balance of this token in the "to" address after this transfer
- "tokenId": the token id of the token, represented in base64 format
- "txid": the hash of the transaction that transfers this token
- "value": the count transfered of this token
"""
    scriptHash = convert_address_to_script_hash(address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetTransferByAddress",
        "params": {"Address": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getTransferByBlockHash(
    block_hash: Annotated[str, "The block hash, for example: 0x7688cf2521bbb5274c22363350539f402e4614a015d9e62b63694c049dec89d6"]
) -> str:
    """
Gets all the token transfer record by the specific block hash.
Returns the result in a JSON string.
The following fields of the result are important:
- "blockhash": the hash of the block where the user gets the token
- "contract": the token contract script hash
- "from": the address where this token is transfered from, if it's null, the token is newly minted
- "frombalance": the remaining balance of this token in the "from" address after this transfer
- "timestamp": the timestamp when this token is owned by this address
- "to": the address where this token is transfered to
- "tobalance": the remaining balance of this token in the "to" address after this transfer
- "tokenId": the token id of the token, represented in base64 format
- "txid": the hash of the transaction that transfers this token
- "value": the count transfered of this token
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetTransferByBlockHash",
        "params": {"BlockHash": block_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getTransferByBlockHeight(
    block_height: Annotated[int, "The block height (index), for example: 3823"]
) -> str:
    """
Gets all the token transfer record by the specific block height.
Returns the result in a JSON string.
The following fields of the result are important:
- "blockhash": the hash of the block where the user gets the token
- "contract": the token contract script hash
- "from": the address where this token is transfered from, if it's null, the token is newly minted
- "frombalance": the remaining balance of this token in the "from" address after this transfer
- "timestamp": the timestamp when this token is owned by this address
- "to": the address where this token is transfered to
- "tobalance": the remaining balance of this token in the "to" address after this transfer
- "tokenId": the token id of the token, represented in base64 format
- "txid": the hash of the transaction that transfers this token
- "value": the count transfered of this token
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetTransferByBlockHeight",
        "params": {"BlockHeight": block_height},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getTransferEventByTransactionHash(
    tx_hash: Annotated[str, "The transaction hash, for example: 0x85b55479fc43668077821234f547824d3111343aec21988f8c0aa1ff9b2ee287"]
) -> str:
    """
Gets the transfer event by the specific transaction hash.
Returns the result in a JSON string.
The following fields of the result are important:
- "callFlags": an enum defines special behaviors allowed when invoking smart contracts, such as chain calls, sending notifications, modifying states, etc.
- "contractHash": the hash of the invoked contract
- "hexStringParams": the params passed when the contract is invoked
- "method": the specific method called of the contract
- "originSender": the address which sends the transaction to call this contract
- "stack": the state of the virtual machine after calling this contract
- "txid": the hash of the transaction
- "vmstate": the state of the virtual machine after calling this contract
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetTransferEventByTransactionHash",
        "params": {"TransactionHash": tx_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getVerifiedContractByContractHash(
    contract_hash: Annotated[str, "Neo N3 contract script hash, for example: 0xcc5e4edd9f5f8dba8bb65734541df7a1c081c67b"]
) -> str:
    """
Gets the verified contract by the contract hash.
Returns the result in a JSON string.
The following fields of the result are important:
- "hash": the contract script hash
- "id": the contract id of this contract in the Neo N3 blockchain system, for example: native contract
- "updatecounter": total count of updates to this contract
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetVerifiedContractByContractHash",
        "params": {"ContractHash": contract_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getVerifiedContract(
    # limit: Annotated[int, "Optional. The number of items to return."],
    # skip: Annotated[int, "Optional. The number of items to skip."]
) -> str:
    """
Gets the information of all the verified contracts.
Returns the result in a JSON string.
The following fields of the result are important:
- "hash": the contract script hash
- "id": the contract id of this contract in the Neo N3 blockchain system, for example: native contract
- "updatecounter": total count of updates to this contract
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetVerifiedContract",
        "params": {},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getVmStateByTransactionHash(
    tx_hash: Annotated[str, "The transaction hash, for example: 0x85b55479fc43668077821234f547824d3111343aec21988f8c0aa1ff9b2ee287"]
) -> str:
    """
Gets the vm state by the transaction hash.
Returns the result in a JSON string.
The following fields of the result are important:
- "vmstate": the contract script hash
"""
    payload = {
        "jsonrpc": "2.0",
        "method": "GetVmStateByTransactionHash",
        "params": {"TransactionHash": tx_hash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def getVotersByCandidateAddress(
    candidate_address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
) -> str:
    """
Gets the information of all the voters of a candidate by the candidate address.
Returns the result in a JSON string.
The following fields of the result are important:
- "balanceOfVoter": the vote count voted to the candidate by the voter
- "blockNumber": the height of the block which the vote transaction is in
- "candidate": the candidate address (script hash)
- "candidatePubKey": the public key of the candidate
- "lastTransferTxid": the last vote transaction hash of the voter
- "lastVoteTxid": the last vote transaction hash of the voter
- "voter": the voter address (script hash)
"""
    scriptHash = convert_address_to_script_hash(candidate_address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetVotersByCandidateAddress",
        "params": {"CandidateAddress": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")

@tool
def getVotesByCandidateAddress(
    candidate_address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
) -> str:
    """
Gets the information of all the votes of a candidate by the candidate address.
Returns the result in a JSON string.
The following fields of the result are important:
- "candidate": the script hash of the candidate
- "isCommittee": indicates if this candidate is selected into the committee
- "state": indicates the current state of the candidate
- "votesOfCandidate": the vote count of this committee
"""
    scriptHash = convert_address_to_script_hash(candidate_address)
    payload = {
        "jsonrpc": "2.0",
        "method": "GetVotersByCandidateAddress",
        "params": {"CandidateAddress": scriptHash},
        "id": 1
    }
    response = requests.post(url, json=payload)
    json_data = response.json()
    if "result" in json_data:
        result_data = json_data["result"]
        return to_json(result_data)
    else:
        return (f"Error from server: {json_data['error']}")


@tool
def convertAssetAmountString(
    amount_string:  Annotated[str, "The asset amount string, such as: totalsupply, balance, and etc."],
    decimals:  Annotated[int, "The asset decimals script hash string."],
) -> Decimal:
    """
Converts an asset amount string to its real value by dividing the amount by 10 to the "asset's decimal"th power.
For example, amount_string = "10000000000000000", decimals = 6, then the result would be 10000000000.000000
"""
    return Decimal(amount_string) / Decimal(10 ** decimals)

tools = [
    getActiveAddresses,
    getAddressCount,
    getAddressInfoByAddress,
    getApplicationLogByTransactionHash,
    getAssetCount,
    getAssetInfoByHash,
    getAssetInfoByName,
    getAssetsInfoByUserAddress,
    getAssetInfoByAssetAndAddress,

    getBestBlockHash, # index 10
    getBlockByHash,
    getBlockByHeight,
    getBlockCount,
    getRecentBlocksInfo,
    getBlockRewardByHash,

    getCandidateByAddress,
    getCandidateByVoterAddress,
    getCandidateCount,
    getCommittee,
    getContractByHash, # index 20
    getContractCount,
    getContractListByName,

    getDailyTransactions,
    getHourlyTransactions,

    getNep11Balance,
    getNep11OwnedByAddress,
    getNep11ByAddressAndHash,
    getNep11TransferByAddress,
    getNep11TransferByBlockHeight,
    getNep11TransferByTransactionHash, # index 30
    getNep11TransferCountByAddress,
    getNep17TransferByAddress,
    getNep17TransferByBlockHeight,
    getNep17TransferByTransactionHash,
    getNep17TransferByContractHash,
    getNep17TransferCountByAddress,

    getNetFeeRange,
    getPopularToken,

    getRawMempool,
    getRawTransactionByAddress, # index 40
    getRawTransactionByBlockHash,
    getRawTransactionByBlockHeight,
    getRawTransactionByTransactionHash,

    getScCallByContractHash,
    getScCallByContractHashAddress,
    getScCallByTransactionHash,
    getScVoteCallByCandidateAddress,
    getScVoteCallByTransactionHash,
    getScVoteCallByVoterAddress,
    getSourceCodeByContractHash, # index 50

    getTagByAddresses,
    getTotalSentAndReceivedByContractHashAddress,
    getTotalVotes,
    getTransactionCount,
    getTransactionCountByAddress,
    getTransferByAddress,
    getTransferByBlockHash,
    getTransferByBlockHeight,
    getTransferEventByTransactionHash,

    getVerifiedContractByContractHash, # index 60
    getVerifiedContract,
    getVmStateByTransactionHash,
    getVotersByCandidateAddress,
    getVotesByCandidateAddress,

    convertAssetAmountString
]

# tools = []

# @tool
# def getAddressDetailByAddress(
#     address: Annotated[str, "Neo N3 address, both the standard format and the script hash format are OK, for example: NiEtVMWVYgpXrWkRTMwRaMJtJ41gD3912N, 0xaad8073e6df9caaf6abc0749250eb0b800c0e6f4"]
# ) -> str:
#     """
#     Gets the details of the given address. 
#     For now, the details only contain the timestamp of when the address is first time used.
#     So the tool only returns a human-readable time string.
#     """
#     scriptHash = ""
#     if neo3.wallet.utils.is_valid_address(address):
#         scriptHash = "0x" + neo3.wallet.utils.address_to_script_hash(address=address).__str__()
#     else:
#         scriptHash = address

#     # Create a JSON-RPC request
#     payload = {
#         "jsonrpc": "2.0",
#         "method": "GetAddressByAddress",
#         "params": {"address": scriptHash},
#         "id": 1
#     }

#     # Send the request
#     response = requests.post(url, json=payload)

#     # Parse the result
#     json_data = response.json()

#     if "result" in json_data:
#         unix_timestamp_ms = json_data["result"]["firstusetime"]
#         # Convert milliseconds to seconds
#         unix_timestamp_s = unix_timestamp_ms / 1000.0
#         # Convert to datetime object
#         human_readable_time = datetime.fromtimestamp(unix_timestamp_s)
#         # Format the datetime as a string
#         return human_readable_time.strftime('%Y-%m-%d %H:%M:%S')
#     else:
#         return (f"Error from server: {json_data['error']}")
    
# @tool
# def getAddressList(
#     limit: Annotated[int, "The number of items to return"]
#     skip: Annotated[int, "The number of items to skip"]
# )

# @tool
# def convertTimestamp(
#     unix_timestamp_ms: Annotated[int, "Unix timestamp in milliseconds"]
# ) -> str:
#     """Converts a Unix timestamp in milliseconds to a human-readable time string"""
#     # Convert milliseconds to seconds
#     unix_timestamp_s = unix_timestamp_ms / 1000.0
    
#     # Convert to datetime object
#     human_readable_time = datetime.fromtimestamp(unix_timestamp_s)
    
#     # Format the datetime as a string
#     return human_readable_time.strftime('%Y-%m-%d %H:%M:%S')

