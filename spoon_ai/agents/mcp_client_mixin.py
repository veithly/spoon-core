from typing import Union, Dict, Any, Optional, List, AsyncIterator, AsyncContextManager
import asyncio
import uuid
from contextlib import asynccontextmanager

from fastmcp.client.transports import (FastMCPTransport, PythonStdioTransport,
                                       SSETransport, WSTransport, NpxStdioTransport,
                                       FastMCPStdioTransport, UvxStdioTransport)
from fastmcp.client import Client as MCPClient
import logging

logger = logging.getLogger(__name__)

class MCPClientMixin:
    def __init__(self, mcp_transport):
        self._client = MCPClient(mcp_transport)
        self._last_sender = None
        self._last_topic = None
        self._last_message_id = None

        # Session management is now simplified - no caching needed

    @asynccontextmanager
    async def get_session(self):
        """Create a new session for each use to avoid connection issues."""
        session = None
        try:
            # Always create a fresh session to avoid connection reuse issues
            session = await self._client.__aenter__()
            yield session
        except asyncio.CancelledError:
            logger.warning("MCP session was cancelled")
            raise
        except Exception as e:
            # Convert generic TaskGroup errors to more specific error messages
            error_msg = str(e)
            if "unhandled errors in a TaskGroup" in error_msg:
                # Try to extract more specific error information
                if hasattr(e, '__cause__') and e.__cause__:
                    specific_error = str(e.__cause__)
                    logger.error(f"MCP session error (TaskGroup): {specific_error}")
                    raise ConnectionError(f"MCP connection failed: {specific_error}") from e
                else:
                    logger.error(f"MCP session error (TaskGroup): {error_msg}")
                    raise ConnectionError("MCP connection failed: Task execution error") from e
            else:
                logger.error(f"MCP session error: {e}")
                raise
        finally:
            if session:
                try:
                    await self._client.__aexit__(None, None, None)
                except asyncio.CancelledError:
                    logger.warning("MCP session close was cancelled")
                except Exception as e:
                    logger.error(f"Error closing MCP session: {e}")

    async def list_mcp_tools(self):
        """Get the list of available tools from the MCP server"""
        async with self.get_session() as session:
            return await session.list_tools()

    async def call_mcp_tool(self, tool_name: str, **kwargs):
        """Call a tool on the MCP server"""
        try:
            async with self.get_session() as session:
                res = await session.call_tool(tool_name, arguments=kwargs)
                if not res:
                    return ""

                # Handle different types of MCP responses
                for item in res:
                    # If it's a text response, check if it's a coroutine object string
                    if hasattr(item, 'text') and item.text is not None:
                        text = item.text
                        # Check if the text indicates a coroutine object (FastMCP async tool issue)
                        if "<coroutine object" in text and "at 0x" in text:
                            # This indicates the MCP server returned a coroutine object instead of executing it
                            # Return an error message indicating the issue
                            return f"Error: MCP tool '{tool_name}' returned a coroutine object instead of executing it. This suggests the tool is async but not properly handled by the MCP server."
                        return text
                    # If it's a JSON response, return the JSON content
                    elif hasattr(item, 'json') and item.json is not None:
                        import json
                        return json.dumps(item.json, ensure_ascii=False, indent=2)

                # Fallback to string representation
                if res:
                    result_str = str(res[0])
                    # Check for coroutine object in fallback as well
                    if "<coroutine object" in result_str and "at 0x" in result_str:
                        return f"Error: MCP tool '{tool_name}' returned a coroutine object instead of executing it. This suggests the tool is async but not properly handled by the MCP server."
                    return result_str

                return ""
        except asyncio.CancelledError:
            logger.warning(f"MCP tool call '{tool_name}' was cancelled")
            raise
        except Exception as e:
            logger.error(f"MCP tool '{tool_name}' call failed: {e}")
            return f"MCP tool '{tool_name}' execution failed: {str(e)}"

    async def send_mcp_message(self, recipient: str, message: Union[str, Dict[str, Any]],
                              topic: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a message to the MCP system

        Args:
            recipient: Recipient ID
            message: Message content (string or dictionary)
            topic: Message topic
            metadata: Additional metadata

        Returns:
            bool: Whether the message was sent successfully
        """
        if isinstance(message, str):
            content = {
                "text": message,
                "source": "agent",
            }
            if metadata:
                content["metadata"] = metadata
        else:
            content = message

        try:
            async with self.get_session() as session:
                await session.send_message(
                    recipient=recipient,
                    message=content,
                    topic=topic or "general"
                )
            return True
        except Exception as e:
            logger.error(f"Failed to send MCP message: {str(e)}")
            return False

    async def reply_to_mcp(self, message: str, topic: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Reply to a previously received MCP message

        Args:
            message: Reply content
            topic: Reply topic, defaults to the topic of the previous message
            metadata: Additional metadata

        Returns:
            bool: Whether the reply was successful
        """
        if not hasattr(self, "_last_sender") or not self._last_sender:
            logger.warning(f"No previous sender to reply to")
            return False

        recipient = self._last_sender
        reply_topic = topic or self._last_topic or "general"

        reply_metadata = {
            "reply_to": self._last_message_id
        }

        if metadata:
            reply_metadata.update(metadata)

        return await self.send_mcp_message(
            recipient=recipient,
            message=message,
            topic=reply_topic,
            metadata=reply_metadata
        )

    async def process_mcp_message(self, content: Any, sender: str, message: Dict[str, Any]):
        """
        Process a message received from the MCP system (should be overridden by subclasses)

        Args:
            content: Message content
            sender: Sender ID
            message: Complete message
        """
        if isinstance(content, dict) and "text" in content:
            text_content = content["text"]
        elif isinstance(content, str):
            text_content = content
        else:
            text_content = str(content)

        self._last_sender = sender
        self._last_topic = message.get("topic", "general")
        self._last_message_id = message.get("id")

        # This method should be overridden by subclasses to handle messages
        logger.info(f"Received MCP message from {sender}: {text_content[:50]}{'...' if len(text_content) > 50 else ''}")

    async def connect(self):
        """
        Establish a connection to the MCP server (this just creates a session to verify connection)
        """
        try:
            async with self.get_session() as session:
                # Just verify we can connect by doing a simple operation
                await session.ping()
                logger.info("Successfully connected to MCP server")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            raise

    async def cleanup(self):
        """Clean up MCP client resources."""
        logger.info("MCP client resources cleaned up")
