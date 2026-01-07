"""Mezo username resolution utilities.

Resolves wallet addresses to username.mezo identities and vice versa
using the Mezo API (https://api.mezo.org/accounts/{address_or_username}).
"""
import json
import logging
import os
from typing import Dict, List, Optional
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# API configuration
MEZO_API_BASE = "https://api.mezo.org"
MEZO_ACCOUNTS_ENDPOINT = f"{MEZO_API_BASE}/accounts"
REQUEST_TIMEOUT = 10  # seconds

# Cache file path
CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
CACHE_FILE = CACHE_DIR / "mezo_usernames.json"


class MezoUsernameResolver:
    """Resolves Mezo usernames from wallet addresses and vice versa."""

    def __init__(self, cache_file: Optional[Path] = None):
        """Initialize resolver with optional custom cache file.

        Args:
            cache_file: Path to cache file. Defaults to cache/mezo_usernames.json
        """
        self.cache_file = cache_file or CACHE_FILE
        self._cache: Dict[str, str] = {}  # address -> mezoId
        self._reverse_cache: Dict[str, str] = {}  # mezoId -> address
        self._load_cache()

    def _load_cache(self) -> None:
        """Load username cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self._cache = data.get('address_to_username', {})
                    self._reverse_cache = data.get('username_to_address', {})
                    logger.debug(f"Loaded {len(self._cache)} cached usernames")
        except Exception as e:
            logger.warning(f"Failed to load username cache: {e}")
            self._cache = {}
            self._reverse_cache = {}

    def _save_cache(self) -> None:
        """Save username cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'address_to_username': self._cache,
                    'username_to_address': self._reverse_cache
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save username cache: {e}")

    def resolve_username(self, address: str) -> Optional[str]:
        """Resolve a wallet address to its Mezo username.

        Args:
            address: Ethereum wallet address (0x...)

        Returns:
            Mezo username (e.g., "username.mezo") or None if not found
        """
        if not address:
            return None

        address_lower = address.lower()

        # Check cache first
        if address_lower in self._cache:
            return self._cache[address_lower]

        # Query API
        try:
            response = requests.get(
                f"{MEZO_ACCOUNTS_ENDPOINT}/{address}",
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                mezo_id = data.get('mezoId')
                if mezo_id:
                    # Update caches
                    self._cache[address_lower] = mezo_id
                    self._reverse_cache[mezo_id.lower()] = address_lower
                    self._save_cache()
                    return mezo_id

            elif response.status_code == 404:
                # Cache negative result to avoid repeated lookups
                # We don't cache None, just skip
                pass
            else:
                logger.warning(f"Unexpected API response: {response.status_code}")

        except requests.RequestException as e:
            logger.warning(f"Failed to resolve username for {address[:10]}...: {e}")

        return None

    def resolve_address(self, username: str) -> Optional[str]:
        """Resolve a Mezo username to its wallet address.

        Args:
            username: Mezo username (e.g., "satoshi" or "satoshi.mezo")

        Returns:
            Wallet address (lowercase) or None if not found
        """
        if not username:
            return None

        # Normalize username (remove .mezo suffix if present for cache lookup)
        username_lower = username.lower()
        if not username_lower.endswith('.mezo'):
            username_with_suffix = f"{username_lower}.mezo"
        else:
            username_with_suffix = username_lower

        # Check cache first
        if username_with_suffix in self._reverse_cache:
            return self._reverse_cache[username_with_suffix]

        # Query API
        try:
            response = requests.get(
                f"{MEZO_ACCOUNTS_ENDPOINT}/{username}",
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                mezo_id = data.get('mezoId')
                linked_accounts = data.get('linkedAccounts', [])

                # Find wallet address from linked accounts
                for account in linked_accounts:
                    if account.get('type') == 'wallet':
                        address = account.get('evmAddress', '').lower()
                        if address:
                            # Update caches
                            if mezo_id:
                                self._cache[address] = mezo_id
                                self._reverse_cache[mezo_id.lower()] = address
                                self._save_cache()
                            return address

            elif response.status_code == 404:
                logger.debug(f"Username not found: {username}")
            else:
                logger.warning(f"Unexpected API response: {response.status_code}")

        except requests.RequestException as e:
            logger.warning(f"Failed to resolve address for {username}: {e}")

        return None

    def batch_resolve_usernames(self, addresses: List[str]) -> Dict[str, str]:
        """Resolve multiple addresses to usernames.

        Args:
            addresses: List of wallet addresses

        Returns:
            Dictionary mapping address (lowercase) to mezoId
        """
        result = {}

        for address in addresses:
            if not address:
                continue

            address_lower = address.lower()

            # Check cache first
            if address_lower in self._cache:
                result[address_lower] = self._cache[address_lower]
            else:
                # Query API (with rate limiting consideration)
                username = self.resolve_username(address)
                if username:
                    result[address_lower] = username

        return result

    def get_display_name(self, address: str) -> str:
        """Get display name for an address (username or truncated address).

        Args:
            address: Wallet address

        Returns:
            Username if available, otherwise truncated address (0x1234...abcd)
        """
        username = self.resolve_username(address)
        if username:
            return username
        return f"{address[:6]}...{address[-4:]}" if address else "Unknown"

    def clear_cache(self) -> None:
        """Clear the username cache."""
        self._cache = {}
        self._reverse_cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()


# Global resolver instance
_resolver: Optional[MezoUsernameResolver] = None


def get_resolver() -> MezoUsernameResolver:
    """Get the global resolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = MezoUsernameResolver()
    return _resolver


def resolve_username(address: str) -> Optional[str]:
    """Convenience function to resolve address to username."""
    return get_resolver().resolve_username(address)


def resolve_address(username: str) -> Optional[str]:
    """Convenience function to resolve username to address."""
    return get_resolver().resolve_address(username)


def get_display_name(address: str) -> str:
    """Convenience function to get display name for address."""
    return get_resolver().get_display_name(address)


def batch_resolve_usernames(addresses: List[str]) -> Dict[str, str]:
    """Convenience function for batch username resolution."""
    return get_resolver().batch_resolve_usernames(addresses)
