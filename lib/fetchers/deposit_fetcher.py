"""Fetcher for Deposit event logs from veBTC contract."""
from typing import List, Dict, Any
import time

# Deposit event signature hash
DEPOSIT_TOPIC_0 = "0x8835c22a0c751188de86681e15904223c054bedd5c68ec8858945b7831290273"


def fetch_deposit_logs(vebtc_address: str,
                       existing_deposits: List[Dict[str, Any]],
                       explorer_api_base: str) -> List[Dict[str, Any]]:
    """Fetch Deposit event logs from Explorer API.

    Args:
        vebtc_address: veBTC contract address
        existing_deposits: Previously fetched deposits
        explorer_api_base: Explorer API base URL

    Returns:
        List of new deposit logs
    """
    import requests

    url = f"{explorer_api_base}/addresses/{vebtc_address}/logs"

    # Build set of existing transaction hashes + log index for deduplication
    existing_keys = set()
    for deposit in existing_deposits:
        tx_hash = deposit.get('transaction_hash') or deposit.get('tx_hash')
        log_index = deposit.get('index')
        if tx_hash and log_index is not None:
            existing_keys.add(f"{tx_hash}:{log_index}")

    new_deposits = []
    next_page_params = {}
    page_count = 0

    print(f"Fetching deposit logs...")

    while True:
        try:
            # Make API request
            params = {"limit": 50, **next_page_params}
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            items = data.get("items", [])
            if not items:
                print(f"  No more items. Total pages: {page_count}")
                break

            # Filter for Deposit events
            new_count = 0
            for item in items:
                # Check if it's a Deposit event
                topics = item.get('topics', [])
                if not topics or topics[0] != DEPOSIT_TOPIC_0:
                    continue

                # Check for Deposit in decoded method_call as backup
                decoded = item.get('decoded', {})
                method_call = decoded.get('method_call', '')
                if 'Deposit' not in method_call:
                    continue

                # Check if we already have this deposit
                tx_hash = item.get('transaction_hash') or item.get('tx_hash')
                log_index = item.get('index')
                key = f"{tx_hash}:{log_index}"

                if key in existing_keys:
                    # We've reached items we already have
                    print(f"  Caught up to existing data after {page_count} pages. Total new deposits: {len(new_deposits)}")
                    return new_deposits

                new_deposits.append(item)
                new_count += 1

            page_count += 1
            print(f"  Page {page_count}: {new_count} new deposits found")

            # Check for next page
            next_page_url = data.get("next_page_params")
            if not next_page_url:
                print(f"  No more pages. Total pages: {page_count}")
                break

            next_page_params = next_page_url

            # Rate limiting
            time.sleep(0.1)

        except requests.exceptions.RequestException as e:
            print(f"  Error fetching deposits: {e}")
            break

    print(f"Fetched {len(new_deposits)} new deposit events")
    return new_deposits
