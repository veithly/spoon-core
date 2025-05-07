import asyncio
import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from typing import (Any, AsyncGenerator, Awaitable, Callable, Dict, Generator,
                    List, Optional)

import anthropic
import openai
from openai import AsyncOpenAI, OpenAI

from spoon_ai.agents.rag import RetrievalMixin
from spoon_ai.utils.config_manager import ConfigManager
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.schema import Role, Message


logger = logging.getLogger(__name__)

DEBUG = False
def debug_log(msg: str):
    if DEBUG:
        logger.info(f"DEBUG: {msg}\n")

class SpoonChatAI(RetrievalMixin):
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
        self.config_dir.mkdir(parents=True, exist_ok=True)
        # retrieval_client will be initialized by RetrievalMixin when needed
        
        # Initialize native SDK client
        self.client = self._generate_client()
    
    def _generate_client(self):
        llm_provider = self.llm_config['llm_provider']
        
        if llm_provider == 'openai':
            api_key = self.config_manager.get_api_key('openai')
            if api_key:
                return OpenAI(api_key=api_key)
            return OpenAI()
        elif llm_provider == 'anthropic':
            api_key = self.config_manager.get_api_key('anthropic')
            if api_key:
                return anthropic.Anthropic(api_key=api_key)
            return anthropic.Anthropic()
        elif llm_provider == 'deepseek':
            # Note: May need to import appropriate DeepSeek SDK
            api_key = self.config_manager.get_api_key('deepseek')
            # Assuming DeepSeek has similar SDK structure
            # May need to adjust based on actual DeepSeek API interface
            return None  # Temporarily return None, to be replaced with appropriate DeepSeek client
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
    def perform_action(self, action_name: str, action_args: List[str], callback: Optional[Callable[[str], None]] = None):
        """Perform an action with the given name and arguments"""
        if action_name == "chat":
            assert len(action_args) == 1, 'Usage: action chat "<message>"'
            return self._generate_response(action_args[0])
        elif action_name == "act":
            pass
        else:
            raise ValueError(f"Unsupported action: {action_name}")
    
    async def astream_chat_response(self, message: str, callback: Optional[Callable[[str], Awaitable[None]]] = None) -> AsyncGenerator[str, None]:
        debug_log(f"Starting streaming response for message: {message[:30]}...")
        
        try:
            # Prepare message history
            history_messages = []
            chat_messages = []
            if isinstance(self.chat_history, dict) and 'messages' in self.chat_history:
                chat_messages = self.chat_history.get('messages', [])
            elif isinstance(self.chat_history, list):
                chat_messages = self.chat_history
            
            # Get relevant documents as context
            context_str, relevant_docs = self.get_context_from_query(message)
            
            # Prepare message format
            llm_provider = self.llm_config['llm_provider']
            
            if llm_provider == 'openai':
                # Build OpenAI format messages
                messages = [{"role": "system", "content": self.prompt_template}]
                
                # Add history messages
                for msg in chat_messages:
                    if msg['role'] in ['user', 'assistant', 'system']:
                        messages.append({"role": msg['role'], "content": msg['content']})
                
                # Add current message, possibly with context
                user_content = message
                if context_str:
                    user_content = f"{message}\n{context_str}"
                messages.append({"role": "user", "content": user_content})
                
                # Create async client
                async_client = AsyncOpenAI(api_key=self.config_manager.get_api_key('openai'))
                
                # Stream API call
                stream = await async_client.chat.completions.create(
                    model=self.llm_config['model'],
                    messages=messages,
                    temperature=self.llm_config.get('temperature', 0.5),
                    max_tokens=self.llm_config.get('max_tokens', 300),
                    stream=True
                )
                
                # Handle streaming response
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield content
                        if callback:
                            await callback(content)
                
            elif llm_provider == 'anthropic':
                # Build Anthropic format messages
                system_prompt = self.prompt_template
                messages = []
                
                # Add history messages
                for msg in chat_messages:
                    if msg['role'] == 'user':
                        messages.append({"role": "user", "content": msg['content']})
                    elif msg['role'] == 'assistant':
                        messages.append({"role": "assistant", "content": msg['content']})
                
                # Add current message, possibly with context
                user_content = message
                if context_str:
                    user_content = f"{message}\n{context_str}"
                messages.append({"role": "user", "content": user_content})
                
                # Call Anthropic API
                stream = await self.client.messages.create(
                    model=self.llm_config['model'],
                    system=system_prompt,
                    messages=messages,
                    temperature=self.llm_config.get('temperature', 0.5),
                    max_tokens=self.llm_config.get('max_tokens', 300),
                    stream=True
                )
                
                # Handle streaming response
                async for chunk in stream:
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                        content = chunk.delta.text
                        if content:
                            yield content
                            if callback:
                                await callback(content)
            
            # DeepSeek and other providers can be added similarly
            else:
                # Use fallback non-streaming response for unsupported providers
                debug_log(f"Streaming not implemented for provider {llm_provider}, using fallback")
                fallback_response = self._generate_response(message)
                
                # Simulate streaming output
                words = fallback_response.split()
                for i in range(0, len(words), 3):
                    chunk = " ".join(words[i:i+3]) + " "
                    yield chunk
                    if callback:
                        await callback(chunk)
                    await asyncio.sleep(0.1)
                
        except Exception as e:
            debug_log(f"Error in astream_chat_response: {e}")
            error_message = f"Error during streaming: {e}. Please try again."
            yield error_message
            if callback:
                await callback(error_message)
        
    def _generate_response(self, message: str) -> str:
        """Generate non-streaming response using native SDK"""
        llm_provider = self.llm_config['llm_provider']
        
        # Prepare history messages
        chat_messages = []
        if isinstance(self.chat_history, dict) and 'messages' in self.chat_history:
            chat_messages = self.chat_history.get('messages', [])
        elif isinstance(self.chat_history, list):
            chat_messages = self.chat_history
            
        # Get context
        context_str, _ = self.get_context_from_query(message)
        user_content = message
        if context_str:
            user_content = f"{message}\n{context_str}"
            
        if llm_provider == 'openai':
            # Build OpenAI format messages
            messages = [{"role": "system", "content": self.prompt_template}]
            
            # Add history messages
            for msg in chat_messages:
                if msg['role'] in ['user', 'assistant', 'system']:
                    messages.append({"role": msg['role'], "content": msg['content']})
            
            # Add current message
            messages.append({"role": "user", "content": user_content})
            
            # Call API
            response = self.client.chat.completions.create(
                model=self.llm_config['model'],
                messages=messages,
                temperature=self.llm_config.get('temperature', 0.5),
                max_tokens=self.llm_config.get('max_tokens', 300)
            )
            
            return response.choices[0].message.content
            
        elif llm_provider == 'anthropic':
            # Build Anthropic format messages
            system_prompt = self.prompt_template
            messages = []
            
            # Add history messages
            for msg in chat_messages:
                if msg['role'] == 'user':
                    messages.append({"role": "user", "content": msg['content']})
                elif msg['role'] == 'assistant':
                    messages.append({"role": "assistant", "content": msg['content']})
            
            # Add current message
            messages.append({"role": "user", "content": user_content})
            
            # Call API
            response = self.client.messages.create(
                model=self.llm_config['model'],
                system=system_prompt,
                messages=messages,
                temperature=self.llm_config.get('temperature', 0.5),
                max_tokens=self.llm_config.get('max_tokens', 300)
            )
            
            return response.content[0].text
        
        # Can add implementation for other providers as needed
        else:
            raise ValueError(f"Response generation not implemented for provider: {llm_provider}")
        
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
        self.client = self._generate_client()
        debug_log(f"Reloaded configuration for agent: {self.name}")
        
                