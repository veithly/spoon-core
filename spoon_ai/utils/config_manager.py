import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Configuration management class for user settings like API keys"""
    
    def __init__(self):
        """Initialize the configuration manager"""
        self.config_dir = Path.home()/ ".config" / "spoonai"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        if not self.config_file.exists():
            # Create default configuration
            default_config = {
                "api_keys": {
                    "openai": os.environ.get("OPENAI_API_KEY", ""),
                    "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
                    "deepseek": os.environ.get("DEEPSEEK_API_KEY", "")
                },
                "default_agent": "default"
            }
            self._save_config(default_config)
            return default_config
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            # Return empty configuration
            return {
                "api_keys": {},
                "default_agent": "default"
            }
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration item"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration item"""
        keys = key.split('.')
        config = self.config
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self._save_config(self.config)
    
    def list_config(self) -> Dict[str, Any]:
        """List all configuration items"""
        return self.config
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specified provider"""
        return self.get(f"api_keys.{provider}")
    
    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for specified provider"""
        self.set(f"api_keys.{provider}", api_key)
        # Also set environment variable
        os.environ[f"{provider.upper()}_API_KEY"] = api_key 