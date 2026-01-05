"""API fetcher for Mezo explorer endpoints."""
import time
import requests
from typing import List, Dict, Any, Set


def get_unique_id(item: Dict[str, Any]) -> str:
    """Generate a unique ID for deduplication.

    Args:
        item: API response item

    Returns:
        Unique identifier string
    """
    # Locks use 'tx_hash' + 'log_index'
    # Logs use 'transaction_hash' + 'index'
    uid = item.get("tx_hash") or item.get("transaction_hash") or item.get("hash")
    idx = str(item.get("index", item.get("log_index", "0")))
    return f"{uid}_{idx}"


def fetch_incremental(url: str,
                      params: Dict[str, Any],
                      existing_items: List[Dict[str, Any]],
                      type_label: str = "items") -> List[Dict[str, Any]]:
    """Fetch only NEW items until a known item is found.

    Args:
        url: API endpoint URL
        params: Query parameters
        existing_items: Previously fetched items
        type_label: Label for progress messages

    Returns:
        List of new items
    """
    existing_ids: Set[str] = set([get_unique_id(i) for i in existing_items])
    new_items: List[Dict[str, Any]] = []

    print(f"Fetching new {type_label}...")

    while True:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])

            if not items:
                break

            page_new_count = 0
            stop_fetching = False

            for item in items:
                uid = get_unique_id(item)

                if uid in existing_ids:
                    # We hit an item we already have.
                    # Since APIs return newest first, we can stop fetching history.
                    stop_fetching = True
                    continue

                # Double check we haven't already added it in this session
                if uid not in existing_ids:
                    new_items.append(item)
                    existing_ids.add(uid)  # Add to set to prevent dups in same run
                    page_new_count += 1

            print(f"  Fetched page... ({page_new_count} new)")

            if stop_fetching:
                print("  Caught up to existing data. Stopping fetch.")
                break

            if data.get("next_page_params"):
                params.update(data.get("next_page_params"))
                time.sleep(0.1)
            else:
                break

        except Exception as e:
            print(f"Error fetching: {e}")
            break

    return new_items


def fetch_current_balance(address_details_url: str) -> str:
    """Fetch the current coin balance of a contract.

    Args:
        address_details_url: API URL for address details

    Returns:
        Formatted balance string
    """
    print("Fetching current balance...")
    try:
        response = requests.get(address_details_url)
        response.raise_for_status()
        data = response.json()
        raw_balance = data.get("coin_balance", "0")
        # remove 18 digits, 2 decimals
        balance_val = float(raw_balance) / (10 ** 18)
        return f"{balance_val:,.2f}"
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return "N/A"
