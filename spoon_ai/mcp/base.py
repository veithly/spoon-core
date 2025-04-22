import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class MCPClient(ABC):
    """
    Base class for MCP (Message Connectivity Protocol) client.
    Defines the core functionality for message-based communication.
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the MCP server/network
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the MCP server/network
        
        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def send_message(self, recipient: str, message: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Send a message to a specific recipient
        
        Args:
            recipient: The recipient identifier
            message: The message content as string or dict
            
        Returns:
            Dict[str, Any]: Response data including message ID and status
        """
        pass
    
    @abstractmethod
    async def receive_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent messages
        
        Args:
            count: Maximum number of messages to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of received messages with metadata
        """
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, callback: callable) -> bool:
        """
        Subscribe to a specific topic/channel
        
        Args:
            topic: The topic or channel to subscribe to
            callback: The callback function to be called when a message is received
            
        Returns:
            bool: True if subscription is successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self, topic: str) -> bool:
        """
        Unsubscribe from a specific topic/channel
        
        Args:
            topic: The topic or channel to unsubscribe from
            
        Returns:
            bool: True if unsubscription is successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the MCP client
        
        Returns:
            Dict[str, Any]: Status information
        """
        pass
