import logging
import shlex
import sys
from pathlib import Path
from typing import Callable, Dict, List
import json
import datetime
import os
import asyncio

logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_openai").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML as PromptHTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit import print_formatted_text
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from spoon_ai.agent import Agent, debug_log
from utils.config_manager import ConfigManager

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("cli")

DEBUG = False
def cli_debug_log(message):
    if DEBUG:
        logger.info(f"CLI DEBUG: {message}\n")

class SpoonCommand:
    name: str
    description: str
    handler: Callable
    aliases: List[str] = []
    def __init__(self, name: str, description: str, handler: Callable, aliases: List[str] = []):
        self.name = name
        self.description = description
        self.handler = handler
        self.aliases = aliases

class SpoonAICLI:
    def __init__(self):
        self.agents = {}
        self.current_agent = None
        self.config_dir = Path.home() / ".config" / "spoonai"
        self.config_dir.mkdir(exist_ok=True)
        self.commands: Dict[str, SpoonCommand] = {}
        self.config_manager = ConfigManager()
        self._init_commands()
        self._set_prompt_toolkit()
        
    
    def _init_commands(self):
        
        # Help Command
        self.add_command(SpoonCommand(
            name="help",
            description="Show help information",
            handler=self._help,
            aliases=["h", "?"]
        ))
        
        # Exit Command
        self.add_command(SpoonCommand(
            name="exit",
            description="Exit the CLI",
            handler=self._exit,
            aliases=["quit", "q"]
        ))
        
        # Load Agent Command
        self.add_command(SpoonCommand(
            name="load-agent",
            description="Load an agent by name",
            handler=self._handle_load_agent,
            aliases=["load"]
        ))
        
        # List Agents Command
        self.add_command(SpoonCommand(
            name="list-agents",
            description="List all available agents",
            handler=self._handle_list_agents,
            aliases=["agents"]
        ))
        
        # Config Command
        self.add_command(SpoonCommand(
            name="config",
            description="Configure settings like API keys",
            handler=self._handle_config,
            aliases=["cfg", "settings"]
        ))
        
        # Reload Config Command
        self.add_command(SpoonCommand(
            name="reload-config",
            description="Reload configuration for current agent",
            handler=self._handle_reload_config,
            aliases=["reload"]
        ))
        
        # Action Command
        self.add_command(SpoonCommand(
            name="action",
            description="Perform an action with the current agent",
            handler=self._handle_action,
            aliases=["a"]
        ))
        
        # Chat History Commands
        self.add_command(SpoonCommand(
            name="new-chat",
            description="Start a new chat (clear history)",
            handler=self._handle_new_chat,
            aliases=["new"]
        ))
        
        self.add_command(SpoonCommand(
            name="list-chats",
            description="List available chat histories",
            handler=self._handle_list_chats,
            aliases=["chats"]
        ))
        
        self.add_command(SpoonCommand(
            name="load-chat",
            description="Load a specific chat history",
            handler=self._handle_load_chat
        ))
    
    def add_command(self, command: SpoonCommand):
        self.commands[command.name] = command
    
    def _help(self, input_list: List[str]):
        if len(input_list) <= 1:
            # show all available commands
            logger.info("Available commands:")
            for command in self.commands.values():
                logger.info(f"  {command.name}: {command.description}")
        else:
            # show help for a specific command
            command_name = input_list[1]
            if command_name in self.commands:
                logger.info(f"Help for {command_name}:")
                logger.info(self.commands[command_name].description)
            else:
                logger.error(f"Command {command_name} not found")

    def _get_prompt(self):
        agent_part = f"({self.current_agent.name})" if self.current_agent else "(no agent)"
        return f"Spoon AI {agent_part} > "
    
    def _handle_load_agent(self, input_list: List[str]):
        if len(input_list) != 1:
            logger.error("Usage: load-agent <agent_name>")
            return
        name = input_list[0]
        self._load_agent(name)
    
    def  _load_agent(self, name: str):
        self.agents[name] = Agent(name)
        self.current_agent = self.agents[name]
        logger.info(f"Loaded agent: {self.current_agent.name}")
    
    def _handle_list_agents(self, input_list: List[str]):
        logger.info("Available agents:")
        for agent in self.agents.values():
            logger.info(f"  {agent.name}: {agent.description}")

    def _load_default_agent(self):
        self._load_agent("default")
    
    def _set_prompt_toolkit(self):
        self.style = Style.from_dict({
            'prompt': 'ansicyan bold',
            'command': 'ansigreen',
            'error': 'ansired bold',
            'success': 'ansigreen bold',
            'warning': 'ansiyellow',
        })
        
        self.completer = WordCompleter(
            list(self.commands.keys()),
            ignore_case=True,
        )
        history_file = self.config_dir / "history.txt"
        history_file.touch(exist_ok=True)
        self.session = PromptSession(
            style=self.style,
            completer=self.completer,
            history=FileHistory(history_file),
        )
        
    def _handle_input(self, input_text: str):
        try:
            input_list = shlex.split(input_text)
            command_name = input_list[0]
            command = self.commands.get(command_name)
            if command:
                command.handler(input_list[1:] if len(input_list) > 1 else [])
            else:
                logger.error(f"Command {command_name} not found")
        except Exception as e:
            logger.error(f"Error: {e}")
        
    def _handle_action(self, input_list: List[str]):
        if not self.current_agent:
            logger.error("No agent loaded")
            return
        
        if len(input_list) < 1:
            logger.error("Usage: action <action_name> [action_args]")
            return
        
        action_name = input_list[0]
        action_args = input_list[1:] if len(input_list) > 1 else []
        
        if action_name == "chat":
            try:
                if action_args:
                    # If arguments provided, use the old behavior
                    res = self.current_agent.perform_action(action_name, action_args)
                    logger.info(res)
                else:
                    # Start interactive chat mode
                    self._start_interactive_chat()
            except Exception as e:
                logger.error(f"Error during action: {e}")
        elif action_name == "new":
            self._handle_new_chat([])
        elif action_name == "list":
            self._handle_list_chats([])
        elif action_name == "load":
            if len(action_args) != 1:
                logger.error("Usage: action load <agent_name>")
                return
            self._handle_load_chat(action_args)
        else:
            self.current_agent.perform_action(action_name, action_args)

    def _start_interactive_chat(self):
        """Start an interactive chat session with the current agent."""
        # Initialize chat history if not exists
        if not hasattr(self.current_agent, 'chat_history'):
            self.current_agent.chat_history = {
                'metadata': {
                    'agent_name': self.current_agent.name,
                    'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'updated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'messages': []
            }
        
        # Create a new prompt session for chat
        chat_style = Style.from_dict({
            'agent': 'ansicyan bold',
            'user': 'ansigreen',
            'system': 'ansigray',
            'header': 'ansiyellow bold',
            'thinking': 'ansiyellow',
            'info': 'ansiblue',
        })
        
        # Create a chat log file
        chat_log_dir = Path('chat_logs')
        chat_log_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        chat_log_file = chat_log_dir / f"chat_{self.current_agent.name}_{timestamp}.txt"
        
        # Display welcome message
        logger.info("="*80)
        logger.info(f"Starting chat with {self.current_agent.name}")
        logger.info("📝 Type your message and press Enter to send.")
        logger.info("🔄 Press Ctrl+C or Ctrl+D to exit chat mode and return to main CLI.")
        logger.info(f"📋 Chat log will be saved to: {chat_log_file}")
        logger.info("="*80 + "\n")
        
        # Function to save chat to log file
        def save_chat_to_log():
            with open(chat_log_file, 'w') as f:
                f.write(f"Chat session with {self.current_agent.name}\n")
                f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 获取消息列表
                chat_messages = []
                if isinstance(self.current_agent.chat_history, dict) and 'messages' in self.current_agent.chat_history:
                    chat_messages = self.current_agent.chat_history.get('messages', [])
                elif isinstance(self.current_agent.chat_history, list):
                    chat_messages = self.current_agent.chat_history
                
                for entry in chat_messages:
                    if entry['role'] == 'user':
                        f.write(f"You: {entry['content']}\n\n")
                    else:
                        f.write(f"{self.current_agent.name}: {entry['content']}\n\n")
                
                f.write(f"\nChat ended at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            self.current_agent.save_chat_history()
        
        # Display chat history
        chat_messages = []
        if isinstance(self.current_agent.chat_history, dict) and 'messages' in self.current_agent.chat_history:
            chat_messages = self.current_agent.chat_history.get('messages', [])
        elif isinstance(self.current_agent.chat_history, list):
            chat_messages = self.current_agent.chat_history
            
        if chat_messages:
            print_formatted_text(PromptHTML("<header>Chat History:</header>"), style=chat_style)
            for entry in chat_messages:
                if entry['role'] == 'user':
                    print_formatted_text(PromptHTML(f"<user>You:</user> {entry['content']}"), style=chat_style)
                else:
                    print_formatted_text(PromptHTML(f"<agent>{self.current_agent.name}:</agent> {entry['content']}"), style=chat_style)
            print("\n" + "-"*50 + "\n")
        
        # Start chat loop
        try:
            while True:
                try:
                    # Get user input
                    user_message = self.session.prompt(
                        PromptHTML("<user>You</user> > "),
                        style=self.style,
                    ).strip()
                    
                    if not user_message:
                        continue
                    
                    # Add to history
                    if isinstance(self.current_agent.chat_history, dict) and 'messages' in self.current_agent.chat_history:
                        self.current_agent.chat_history['messages'].append({
                            'role': 'user',
                            'content': user_message
                        })
                    else:
                        self.current_agent.chat_history.append({
                            'role': 'user',
                            'content': user_message
                        })
                    
                    # Get response from agent
                    print_formatted_text(PromptHTML(f"<thinking>{self.current_agent.name} is thinking...</thinking>"), style=chat_style)
                    
                    # Use streaming response if available
                    try:
                        # Define a simpler streaming function
                        async def stream_response():
                            cli_debug_log("Starting stream_response function")
                            # Clear the "thinking" line and show agent name
                            # print("\033[1A\033[K", end="")
                            
                            # Display agent name
                            print_formatted_text(
                                PromptHTML(f"<agent>{self.current_agent.name}:</agent>"),
                                style=chat_style,
                                end=" "
                            )
                            
                            # Collect the full response
                            full_response = ""
                            chunk_count = 0
                            agent_name_prefix = f"{self.current_agent.name}: "
                            agent_name_prefix_lower = agent_name_prefix.lower()
                            
                            # Stream the response
                            cli_debug_log("Starting to stream response")
                            try:
                                async for chunk in self.current_agent.astream_chat_response(user_message):
                                    if chunk:
                                        chunk_count += 1
                                        cli_debug_log(f"Received chunk #{chunk_count}: {chunk[:20]}...")
                                        
                                        # Check if first chunk starts with agent name and remove it
                                        if chunk_count == 1:
                                            # Check for exact match
                                            if chunk.startswith(agent_name_prefix):
                                                chunk = chunk[len(agent_name_prefix):]
                                                cli_debug_log(f"Removed agent name prefix from chunk")
                                            # Check for case-insensitive match
                                            elif chunk.lower().startswith(agent_name_prefix_lower):
                                                chunk = chunk[len(agent_name_prefix):]
                                                cli_debug_log(f"Removed case-insensitive agent name prefix from chunk")
                                        
                                        full_response += chunk
                                        print(chunk, end="", flush=True)
                                
                                cli_debug_log(f"Finished streaming, received {chunk_count} chunks")
                                
                                # Ensure a new line after the response
                                if chunk_count > 0:
                                    print()
                            except Exception as e:
                                cli_debug_log(f"Error during streaming iteration: {e}")
                                print(f"\nError during streaming: {e}")
                                if not full_response:
                                    cli_debug_log("No response received, using non-streaming")
                                    print("Using non-streaming response...")
                                    full_response = self.current_agent._generate_response(user_message)
                                    print(full_response)
                            
                            return full_response
                        
                        # Run the streaming function
                        cli_debug_log("Running stream_response function")
                        response = asyncio.run(stream_response())
                        cli_debug_log(f"Stream response completed, got response of length {len(response)}")
                        
                        # Add to history if we got a response
                        if response:
                            cli_debug_log("Adding streaming response to chat history")
                            if isinstance(self.current_agent.chat_history, dict) and 'messages' in self.current_agent.chat_history:
                                self.current_agent.chat_history['messages'].append({
                                    'role': 'assistant',
                                    'content': response
                                })
                            else:
                                self.current_agent.chat_history.append({
                                    'role': 'assistant',
                                    'content': response
                                })
                        else:
                            # Fallback if streaming returned empty
                            cli_debug_log("Streaming returned empty response, falling back to non-streaming")
                            logger.info("Streaming returned empty response, falling back to non-streaming...")
                            response = self.current_agent._generate_response(user_message)
                            cli_debug_log(f"Got non-streaming response of length {len(response)}")
                            
                            # Check if response starts with agent name and remove it
                            agent_name_prefix = f"{self.current_agent.name}: "
                            if response.startswith(agent_name_prefix):
                                response = response[len(agent_name_prefix):]
                                cli_debug_log(f"Removed agent name prefix from non-streaming response")
                            
                            # Add to history
                            if isinstance(self.current_agent.chat_history, dict) and 'messages' in self.current_agent.chat_history:
                                self.current_agent.chat_history['messages'].append({
                                    'role': 'assistant',
                                    'content': response
                                })
                            else:
                                self.current_agent.chat_history.append({
                                    'role': 'assistant',
                                    'content': response
                                })
                            
                            # Display response
                            print_formatted_text(PromptHTML(f"<agent>{self.current_agent.name}:</agent> {response}"), style=chat_style)
                        
                    except Exception as e:
                        # Fallback to non-streaming if streaming not available or failed
                        cli_debug_log(f"Streaming failed with error: {e}")
                        logger.info(f"Streaming failed: {e}. Using non-streaming response...")
                        response = self.current_agent._generate_response(user_message)
                        cli_debug_log(f"Got non-streaming response of length {len(response)}")
                        
                        # Check if response starts with agent name and remove it
                        agent_name_prefix = f"{self.current_agent.name}: "
                        if response.startswith(agent_name_prefix):
                            response = response[len(agent_name_prefix):]
                            cli_debug_log(f"Removed agent name prefix from non-streaming response")
                        
                        # Add to history
                        if isinstance(self.current_agent.chat_history, dict) and 'messages' in self.current_agent.chat_history:
                            self.current_agent.chat_history['messages'].append({
                                'role': 'assistant',
                                'content': response
                            })
                        else:
                            self.current_agent.chat_history.append({
                                'role': 'assistant',
                                'content': response
                            })
                        
                        # Display response
                        print_formatted_text(PromptHTML(f"<agent>{self.current_agent.name}:</agent> {response}"), style=chat_style)
                    
                except (KeyboardInterrupt, EOFError):
                    logger.info("\nExiting chat mode...")
                    break
        finally:
            # Save chat log when exiting
            save_chat_to_log()
            print_formatted_text(
                PromptHTML(f"<info>Chat log saved to: {chat_log_file}</info>"),
                style=chat_style
            )
            print("="*50 + "\n")
        
    def run(self):
        self._load_default_agent()
        
        while True:
            try:
                input_text = self.session.prompt(
                    self._get_prompt(),
                    style=self.style,
                ).strip()
                
                if not input_text:
                    continue
                
                self._handle_input(input_text)
            except KeyboardInterrupt:
                continue
            except EOFError:
                self._exit([])
            
    def _exit(self, input_list: List[str]):
        logger.info("Exiting Spoon AI")
        sys.exit(0)

    def _handle_new_chat(self, input_list: List[str]):
        if not self.current_agent:
            logger.error("No agent loaded")
            return
            
        self.current_agent.clear_chat_history()
        logger.info(f"Started new chat with {self.current_agent.name}")
    
    def _handle_list_chats(self, input_list: List[str]):
        chat_logs_dir = Path('chat_logs')
        if not chat_logs_dir.exists():
            logger.info("No chat histories found")
            return
            
        chat_files = list(chat_logs_dir.glob('*_history.json'))
        if not chat_files:
            logger.info("No chat histories found")
            return
            
        logger.info("Available chat histories:")
        for chat_file in chat_files:
            agent_name = chat_file.stem.replace('_history', '')
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    msg_count = len(history)
                    if msg_count > 0:
                        first_date = "Unknown"
                        last_date = "Unknown"
                        
                        if 'metadata' in history and 'created_at' in history['metadata']:
                            first_date = history['metadata']['created_at']
                            last_date = history['metadata'].get('updated_at', first_date)
                        else:
                            file_time = datetime.datetime.fromtimestamp(chat_file.stat().st_mtime)
                            last_date = file_time.strftime('%Y-%m-%d')
                            
                        logger.info(f"  {agent_name}: {msg_count} messages ({first_date} - {last_date})")
                    else:
                        logger.info(f"  {agent_name}: Empty chat history")
            except Exception as e:
                logger.info(f"  {agent_name}: Error reading history - {e}")
    
    def _handle_load_chat(self, input_list: List[str]):
        if not self.current_agent:
            logger.error("No agent loaded")
            return
            
        if len(input_list) != 1:
            logger.error("Usage: load-chat <agent_name>")
            return
            
        agent_name = input_list[0]
        chat_file = Path('chat_logs') / f'{agent_name}_history.json'
        
        if not chat_file.exists():
            logger.error(f"Chat history for {agent_name} not found")
            return
            
        try:
            with open(chat_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                
            self.current_agent.save_chat_history()
            
            self.current_agent.chat_history = history
            logger.info(f"Loaded chat history from {agent_name} ({len(history)} messages)")
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")

    def _handle_config(self, input_list: List[str]):
        """Handle configuration command"""
        if not input_list:
            self._show_config()
            return
        
        if input_list[0] == "help":
            self._show_config_help()
            return
        
        if len(input_list) == 1:
            key = input_list[0]
            value = self.config_manager.get(key)
            if value is not None:
                logger.info(f"{key}: {value}")
            else:
                logger.info(f"Configuration item '{key}' does not exist")
            return
        
        if len(input_list) >= 2:
            key = input_list[0]
            value = " ".join(input_list[1:])
            
            if key.startswith("api_keys.") or key == "api_key":
                provider = key.split(".")[-1] if "." in key else input_list[1]
                if key == "api_key":
                    if len(input_list) < 3:
                        logger.info("Usage: config api_key <provider> <key>")
                        return
                    provider = input_list[1]
                    value = " ".join(input_list[2:])
                self.config_manager.set_api_key(provider, value)
                logger.info(f"Set {provider} API key")
            else:
                self.config_manager.set(key, value)
                logger.info(f"Set {key} = {value}")
    
    def _show_config(self):
        """Show all configuration"""
        config = self.config_manager.list_config()
        logger.info("Current configuration:")
        
        # Handle API keys, don't show full keys
        if "api_keys" in config:
            logger.info("API Keys:")
            for provider, key in config["api_keys"].items():
                masked_key = "Not set" if not key else f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "Set"
                logger.info(f"  {provider}: {masked_key}")
        
        # Show other configuration
        for key, value in config.items():
            if key != "api_keys":
                logger.info(f"{key}: {value}")
    
    def _show_config_help(self):
        """Show configuration command help"""
        logger.info("Configuration command usage:")
        logger.info("  config                 - Show all configuration")
        logger.info("  config <key>           - Show specific configuration item")
        logger.info("  config <key> <value>   - Set configuration item")
        logger.info("  config api_key <provider> <key> - Set API key")
        logger.info("  config help            - Show this help")
        logger.info("")
        logger.info("Examples:")
        logger.info("  config api_key openai sk-xxxx")
        logger.info("  config api_key anthropic sk-ant-xxxx")
        logger.info("  config default_agent my_agent")

    def _handle_list_agents(self, input_list: List[str]):
        logger.info("Available agents:")
        for agent in self.agents.values():
            logger.info(f"  {agent.name}: {agent.description}")

    def _handle_reload_config(self, input_list: List[str]):
        """Reload configuration"""
        if not self.current_agent:
            logger.info("No agent loaded, please load an agent first")
            return
        
        self.current_agent.reload_config()
        logger.info(f"Reloaded configuration for agent '{self.current_agent.name}'")