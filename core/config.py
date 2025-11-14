"""
Configuration Management
Handles loading and accessing application configuration.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class Config:
    """Configuration manager for Story Architect."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            logger.debug(f"Loaded configuration from {self.config_path}")
            return config

        except Exception as e:
            raise Exception(f"Failed to load configuration: {str(e)}")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get configuration value by key path.

        Args:
            key: Dot-separated key path (e.g., 'agents.agent_1.temperature')
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            config.get('gemini.model')  # Returns 'gemini-2.5-pro'
            config.get('agents.agent_1.temperature')  # Returns 0.8
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent.

        Args:
            agent_name: Agent name (e.g., 'agent_1')

        Returns:
            Agent configuration dictionary

        Raises:
            KeyError: If agent configuration not found
        """
        agent_config = self.get(f'agents.{agent_name}')

        if agent_config is None:
            raise KeyError(f"Configuration not found for agent: {agent_name}")

        return agent_config

    def get_gemini_config(self) -> Dict[str, Any]:
        """Get Gemini API configuration."""
        return self.get('gemini', {})

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.get('output', {})

    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration."""
        return self.get('app', {})

    def get_vertex_ai_config(self) -> Dict[str, Any]:
        """Get Vertex AI configuration."""
        return self.get('gemini.vertex_ai', {})

    def is_vertex_ai_enabled(self) -> bool:
        """Check if Vertex AI mode is enabled."""
        return self.get('gemini.use_vertex_ai', False)

    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()
        logger.info("Configuration reloaded")

    @property
    def all(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return self._config


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get global configuration instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Config instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config(config_path)

    return _config_instance


def reload_config() -> None:
    """Reload global configuration."""
    global _config_instance

    if _config_instance is not None:
        _config_instance.reload()
    else:
        _config_instance = Config()
