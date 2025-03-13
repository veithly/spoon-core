import asyncio
import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from typing import (Any, AsyncGenerator, Awaitable, Callable, Dict, Generator,
                    List, Optional)

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from spoon_ai.agents.rag import RetrievalMixin
from utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)



class SpoonAI(RetrievalMixin):
    def __init__(self, name: str):
        agent_config_path = Path('agents') / f'{name}.json'
        if not agent_config_path.exists():
            raise FileNotFoundError(f"Agent config file not found: {agent_config_path}")
        
        with open(agent_config_path, 'r') as f:
            config_dict = json.load(f)
            
        self.name = config_dict['name']
        self.description = config_dict['description']
        self.llm_config = config_dict['llm_config']
        self.tools = config_dict['tools']
        self.prompt_template = config_dict['prompt_template']
        # response to user or other agents
        self.output_type = config_dict['output_type']
        
        # Load config manager
        self.config_manager = ConfigManager()
        
        # Initialize chat history
        self.chat_history = []
        
        # Set up config directory for retrieval
        self.config_dir = Path.home() / ".config" / "spoonai"
        self.config_dir.mkdir(exist_ok=True)
        # retrieval_client will be initialized by RetrievalMixin when needed
        
        self.chatbot = self._generate_chatbot()
    
    def _generate_chatbot(self):
        llm_provider = self.llm_config['llm_provider']
        
        chatbot_config = {
            "model" : self.llm_config['model'],
            "temperature" : self.llm_config.get('temperature', 0.5),
            "max_tokens" : self.llm_config.get('max_tokens', 300),
            "streaming" : True,  
            "verbose" : False
        }
        
        if llm_provider == 'openai':
            api_key = self.config_manager.get_api_key('openai')
            if api_key:
                chatbot_config["api_key"] = api_key
            return ChatOpenAI(**chatbot_config)
        elif llm_provider == 'deepseek':
            api_key = self.config_manager.get_api_key('deepseek')
            if api_key:
                chatbot_config["api_key"] = api_key
            return ChatDeepSeek(**chatbot_config)
        elif llm_provider == 'anthropic':
            api_key = self.config_manager.get_api_key('anthropic')
            if api_key:
                chatbot_config["api_key"] = api_key
            return ChatAnthropic(**chatbot_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        
        
    def perform_action(self, action_name: str, action_args: List[str], callback: Optional[Callable[[str], None]] = None):
        """Perform an action with the given name and arguments"""
        if action_name == "chat":
            assert len(action_args) == 1, 'Usage: action chat "<message>"'
            return self._generate_response(action_args[0])
        else:
            raise ValueError(f"Unsupported action: {action_name}")
        
    # RAG methods have been moved to RetrievalMixin
            
    async def astream_chat_response(self, message: str, callback: Optional[Callable[[str], Awaitable[None]]] = None) -> AsyncGenerator[str, None]:
        debug_log(f"Starting streaming response for message: {message[:30]}...")
        
        try:
            system_message = SystemMessage(content=self.prompt_template)
            messages = [system_message]
            
            # Add chat history
            history_limit = 10
            history_pairs = []
            
            # Get message list
            chat_messages = []
            if isinstance(self.chat_history, dict) and 'messages' in self.chat_history:
                chat_messages = self.chat_history.get('messages', [])
            elif isinstance(self.chat_history, list):
                chat_messages = self.chat_history
            
            # Retrieve relevant documents using the mixin method
            context_str, relevant_docs = self.get_context_from_query(message)
            
            for i in range(0, len(chat_messages) - 1, 2):
                if i + 1 < len(chat_messages) and len(history_pairs) < history_limit:
                    user_msg = chat_messages[i]
                    assistant_msg = chat_messages[i + 1]
                    if user_msg['role'] == 'user' and assistant_msg['role'] == 'assistant':
                        history_pairs.append((user_msg, assistant_msg))
            
            debug_log(f"Added {len(history_pairs)} history pairs to context")
            
            for user_msg, assistant_msg in history_pairs:
                messages.append(HumanMessage(content=user_msg['content']))
                messages.append(SystemMessage(content=f"Assistant: {assistant_msg['content']}"))
            
            # Add context from retrieved documents if available
            if context_str:
                message_with_context = f"{message}\n{context_str}"
                messages.append(HumanMessage(content=message_with_context))
            else:
                messages.append(HumanMessage(content=message))
            
            debug_log("Creating callback handler for streaming")
            callback_handler = AsyncIteratorCallbackHandler()
            
            debug_log("Creating streaming LLM")
            streaming_llm = self.chatbot.with_config(
                callbacks=[callback_handler],
                streaming=True
            )
            
            debug_log("Starting LLM task")
            task = asyncio.create_task(streaming_llm.ainvoke(messages))
            
            chunk_count = 0
            has_yielded = False
            
            try:
                debug_log("Starting to iterate through chunks")
                async for chunk in callback_handler.aiter():
                    if chunk:
                        chunk_count += 1
                        debug_log(f"Received chunk #{chunk_count}: {chunk[:20]}...")
                        has_yielded = True
                        yield chunk
                        if callback:
                            await callback(chunk)
                
                debug_log(f"Finished streaming, received {chunk_count} chunks")
                
                if not has_yielded:
                    debug_log("No chunks yielded, generating fallback response")
                    fallback_response = self._generate_response(message)
                    debug_log(f"Generated fallback response of length {len(fallback_response)}")
                    
                    # Check if fallback response starts with agent name and remove it
                    agent_name_prefix = f"{self.name}: "
                    if fallback_response.startswith(agent_name_prefix):
                        fallback_response = fallback_response[len(agent_name_prefix):]
                        debug_log("Removed agent name prefix from fallback response")
                    
                    debug_log("Simulating streaming with fallback response")
                    words = fallback_response.split()
                    for i in range(0, len(words), 3):
                        chunk = " ".join(words[i:i+3]) + " "
                        debug_log(f"Yielding simulated chunk: {chunk}")
                        yield chunk
                        if callback:
                            await callback(chunk)
                        await asyncio.sleep(0.1)
            finally:
                debug_log("Cleaning up streaming task")
                if not task.done():
                    await task
        
        except Exception as e:
            debug_log(f"Error in astream_chat_response: {e}")
            error_message = f"Error during streaming: {e}. Please try again."
            yield error_message
            if callback:
                await callback(error_message)
        
    def _generate_response(self, message: str) -> str:
        system_message = SystemMessage(content=self.prompt_template)
        messages = [system_message]
        
        history_limit = 10
        history_pairs = []
        
        chat_messages = []
        if isinstance(self.chat_history, dict) and 'messages' in self.chat_history:
            chat_messages = self.chat_history.get('messages', [])
        elif isinstance(self.chat_history, list):
            chat_messages = self.chat_history
        for i in range(0, len(chat_messages) - 1, 2):
            if i + 1 < len(chat_messages) and len(history_pairs) < history_limit:
                user_msg = chat_messages[i]
                assistant_msg = chat_messages[i + 1]
                if user_msg['role'] == 'user' and assistant_msg['role'] == 'assistant':
                    history_pairs.append((user_msg, assistant_msg))
        
        for user_msg, assistant_msg in history_pairs:
            messages.append(HumanMessage(content=user_msg['content']))
            messages.append(SystemMessage(content=f"Assistant: {assistant_msg['content']}"))
        
        # Add current message
        messages.append(HumanMessage(content=message))
        
        res = self.chatbot.invoke(messages)
        return res.content
        
    def load_chat_history(self):
        history_file = Path('chat_logs') / f'{self.name}_history.json'
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, dict) and 'messages' in data:
                    self.chat_history = data
                    message_count = len(data.get('messages', []))
                elif isinstance(data, list):
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.chat_history = {
                        'metadata': {
                            'agent_name': self.name,
                            'created_at': now,
                            'updated_at': now
                        },
                        'messages': data
                    }
                    message_count = len(data)
                else:
                    debug_log(f"Unknown chat history format, initializing empty history")
                    self.chat_history = {
                        'metadata': {
                            'agent_name': self.name,
                            'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'updated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        },
                        'messages': []
                    }
                    message_count = 0
                
                debug_log(f"Loaded {message_count} messages from chat history")
            except Exception as e:
                debug_log(f"Error loading chat history: {e}")
                self.chat_history = {
                    'metadata': {
                        'agent_name': self.name,
                        'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'updated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'messages': []
                }
        else:
            self.chat_history = {
                'metadata': {
                    'agent_name': self.name,
                    'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'updated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'messages': []
            }
        
    def clear_chat_history(self):
        """Clear chat history"""
        old_length = 0
        if isinstance(self.chat_history, dict) and 'messages' in self.chat_history:
            old_length = len(self.chat_history.get('messages', []))
            metadata = self.chat_history.get('metadata', {})
            metadata['updated_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.chat_history = {
                'metadata': metadata,
                'messages': []
            }
        elif isinstance(self.chat_history, list):
            old_length = len(self.chat_history)
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.chat_history = {
                'metadata': {
                    'agent_name': self.name,
                    'created_at': now,
                    'updated_at': now
                },
                'messages': []
            }
        else:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.chat_history = {
                'metadata': {
                    'agent_name': self.name,
                    'created_at': now,
                    'updated_at': now
                },
                'messages': []
            }
        
        debug_log(f"Cleared {old_length} messages from chat history")
        
        history_file = Path('chat_logs') / f'{self.name}_history.json'
        if history_file.exists():
            try:
                history_file.unlink()
                debug_log(f"Deleted chat history file: {history_file}")
            except Exception as e:
                debug_log(f"Error deleting chat history file: {e}")
    
    def save_chat_history(self):
        history_dir = Path('chat_logs')
        history_dir.mkdir(exist_ok=True)
        
        history_file = history_dir / f'{self.name}_history.json'
        
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if isinstance(self.chat_history, list):
            save_data = {
                'metadata': {
                    'agent_name': self.name,
                    'created_at': now,
                    'updated_at': now
                },
                'messages': self.chat_history
            }
        elif isinstance(self.chat_history, dict) and 'metadata' in self.chat_history:
            save_data = self.chat_history
            save_data['metadata']['updated_at'] = now
        else:
            save_data = {
                'metadata': {
                    'agent_name': self.name,
                    'created_at': now,
                    'updated_at': now
                },
                'messages': []
            }
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            debug_log(f"Saved chat history with {len(save_data.get('messages', []))} messages")
        except Exception as e:
            debug_log(f"Error saving chat history: {e}")
    
    def reload_config(self):
        self.config_manager = ConfigManager()
        self.chatbot = self._generate_chatbot()
        debug_log(f"Reloaded configuration for agent: {self.name}")
        
                