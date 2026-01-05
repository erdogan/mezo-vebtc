"""Configuration loader and validator for veBTC dashboard."""
import json
import os
from typing import Dict, Any


class Config:
    """Configuration manager for the veBTC dashboard."""

    def __init__(self, config_path: str = "config.json"):
        """Load configuration from file.

        Args:
            config_path: Path to config.json file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _validate_config(self) -> None:
        """Validate required configuration fields."""
        required_sections = ["network", "contracts", "api", "constants"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

        # Validate contracts section
        required_contracts = ["veBTC", "voter"]
        for contract in required_contracts:
            if contract not in self.config["contracts"]:
                raise ValueError(f"Missing required contract: {contract}")
            if "address" not in self.config["contracts"][contract]:
                raise ValueError(f"Missing address for contract: {contract}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports nested keys with dots, e.g., 'network.rpc_url')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @property
    def rpc_url(self) -> str:
        """Get RPC URL."""
        return self.get('network.rpc_url')

    @property
    def chain_id(self) -> int:
        """Get chain ID."""
        return self.get('network.chain_id')

    @property
    def vebtc_address(self) -> str:
        """Get veBTC contract address."""
        return self.get('contracts.veBTC.address')

    @property
    def voter_address(self) -> str:
        """Get Voter contract address."""
        return self.get('contracts.voter.address')

    @property
    def lock_token(self) -> str:
        """Get lock token address."""
        return self.get('contracts.veBTC.token')

    @property
    def lock_url(self) -> str:
        """Get lock API URL."""
        return self.get('api.lock_url')

    @property
    def vote_url(self) -> str:
        """Get vote API URL."""
        return self.get('api.vote_url')

    @property
    def voted_topic_0(self) -> str:
        """Get voted event topic 0."""
        return self.get('api.voted_topic_0')

    @property
    def default_decimals(self) -> int:
        """Get default decimals."""
        return self.get('constants.default_decimals', 18)

    @property
    def week_seconds(self) -> int:
        """Get seconds in a week."""
        return self.get('constants.week_seconds', 604800)


def load_config(config_path: str = "config.json") -> Config:
    """Load configuration from file.

    Args:
        config_path: Path to config.json

    Returns:
        Config object
    """
    return Config(config_path)
