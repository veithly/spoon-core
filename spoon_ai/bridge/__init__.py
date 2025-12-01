"""
Cross-chain bridge module for DID synchronization
Ethereum ← → NeoFS/IPFS event indexing
"""

from .eth_neofs_indexer import EthereumNeoFSIndexer
from .neo_state_sync import NeoStateSyncStub

__all__ = ["EthereumNeoFSIndexer", "NeoStateSyncStub"]


