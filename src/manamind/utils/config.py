"""Configuration management for ManaMind."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""

    state_encoder: Dict[str, Any] = Field(default_factory=dict)
    policy_value_network: Dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Training configuration."""

    self_play: Dict[str, Any] = Field(default_factory=dict)
    mcts: Dict[str, Any] = Field(default_factory=dict)
    neural_training: Dict[str, Any] = Field(default_factory=dict)
    training_loop: Dict[str, Any] = Field(default_factory=dict)
    optimizer: Dict[str, Any] = Field(default_factory=dict)
    lr_scheduler: Dict[str, Any] = Field(default_factory=dict)


class ForgeConfig(BaseModel):
    """Forge integration configuration."""

    installation_path: Optional[str] = None
    java_opts: list = Field(default_factory=lambda: ["-Xmx4G", "-server"])
    port: int = 25333
    timeout: float = 30.0
    use_py4j: bool = True
    default_decks: Dict[str, str] = Field(default_factory=dict)


class Config(BaseModel):
    """Main configuration class for ManaMind."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    forge: ForgeConfig = Field(default_factory=ForgeConfig)
    data: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)
    hardware: Dict[str, Any] = Field(default_factory=dict)
    phases: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config instance
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Load configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Config instance
        """
        return cls(**data)

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration.

        Returns:
            Config instance with default values
        """
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return self.model_dump()

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save configuration file
        """
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., 'training.mcts.simulations')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        try:
            keys = key.split(".")
            value = self.to_dict()

            for k in keys:
                value = value[k]

            return value
        except (KeyError, TypeError):
            return default

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.

        Args:
            updates: Dictionary of updates to apply
        """

        def update_nested_dict(d: dict, u: dict) -> dict:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        current_dict = self.to_dict()
        updated_dict = update_nested_dict(current_dict, updates)

        # Recreate the config with updated values
        new_config = self.__class__(**updated_dict)

        # Update current instance
        for field_name, field_value in new_config:
            setattr(self, field_name, field_value)

    def override_from_env(self, prefix: str = "MANAMIND_") -> None:
        """Override configuration values from environment variables.

        Args:
            prefix: Environment variable prefix
        """
        env_updates: Dict[str, Any] = {}

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert MANAMIND_TRAINING_MCTS_SIMULATIONS -> training.mcts
                config_key = key[len(prefix) :].lower().replace("_", ".")

                # Try to convert to appropriate type
                converted_value: Any = value
                try:
                    if value.lower() in ("true", "false"):
                        converted_value = value.lower() == "true"
                    elif value.isdigit():
                        converted_value = int(value)
                    elif "." in value and value.replace(".", "").isdigit():
                        converted_value = float(value)
                except ValueError:
                    pass  # Keep as string

                # Set nested dictionary value
                keys = config_key.split(".")
                d = env_updates
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = converted_value

        if env_updates:
            self.update(env_updates)


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    use_env: bool = True,
) -> Config:
    """Load configuration with various sources.

    Args:
        config_path: Path to configuration file
        overrides: Dictionary of configuration overrides
        use_env: Whether to use environment variable overrides

    Returns:
        Loaded configuration
    """
    # Start with default configuration
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        config = Config.default()

    # Apply environment overrides
    if use_env:
        config.override_from_env()

    # Apply explicit overrides
    if overrides:
        config.update(overrides)

    return config
