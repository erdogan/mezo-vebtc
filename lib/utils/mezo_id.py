"""Mezo ID resolution utilities."""

import logging
import requests
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MEZO_API_BASE = "https://api.mezo.org"


def is_mezo_id(value: str) -> bool:
    """Check if a value looks like a Mezo ID.

    Mezo IDs end with .mezo (e.g., playtest.mezo)

    Args:
        value: String to check

    Returns:
        True if it looks like a Mezo ID
    """
    if not value:
        return False
    return value.lower().endswith('.mezo')


def resolve_mezo_id(mezo_id: str, timeout: int = 10) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a Mezo ID to a wallet address.

    Args:
        mezo_id: The Mezo ID to resolve (e.g., playtest.mezo)
        timeout: Request timeout in seconds

    Returns:
        Tuple of (wallet_address, error_message)
        - On success: (address, None)
        - On failure: (None, error_message)
    """
    try:
        url = f"{MEZO_API_BASE}/accounts/{mezo_id}"

        response = requests.get(url, timeout=timeout)

        if response.status_code == 404:
            return None, f"Mezo ID '{mezo_id}' not found"

        if response.status_code != 200:
            logger.error(f"Mezo API error: {response.status_code} for {mezo_id}")
            return None, "Failed to lookup Mezo ID. Please try again."

        data = response.json()

        # Extract wallet address from linkedAccounts
        linked_accounts = data.get('linkedAccounts', [])

        for account in linked_accounts:
            if account.get('type') == 'wallet':
                evm_address = account.get('evmAddress')
                if evm_address:
                    logger.info(f"Resolved Mezo ID {mezo_id} -> {evm_address}")
                    return evm_address, None

        return None, f"Mezo ID '{mezo_id}' has no linked wallet"

    except requests.exceptions.Timeout:
        logger.error(f"Timeout resolving Mezo ID: {mezo_id}")
        return None, "Request timed out. Please try again."

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error resolving Mezo ID {mezo_id}: {e}")
        return None, "Network error. Please try again."

    except Exception as e:
        logger.error(f"Unexpected error resolving Mezo ID {mezo_id}: {e}")
        return None, "An unexpected error occurred. Please try again."
