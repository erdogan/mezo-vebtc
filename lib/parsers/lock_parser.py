"""Parser for lock events."""
from datetime import datetime
from typing import List, Dict, Any


def parse_locks(locks: List[Dict[str, Any]],
                contract_address: str,
                default_decimals: int = 18) -> List[Dict[str, Any]]:
    """Parse lock transaction data.

    Args:
        locks: Raw lock transactions from API
        contract_address: veBTC contract address (to filter self-transfers)
        default_decimals: Token decimal places

    Returns:
        List of parsed lock records
    """
    lock_list = []

    print(f"Processing {len(locks)} lock records...")

    for tx in locks:
        try:
            ts = tx.get("timestamp")
            if not ts:
                continue

            dt = datetime.strptime(ts.split('.')[0].replace('Z', ''), "%Y-%m-%dT%H:%M:%S")

            raw_val = tx.get("total")
            if isinstance(raw_val, dict):
                raw_val = raw_val.get("value")
            amount = float(raw_val or 0) / (10 ** default_decimals)

            from_obj = tx.get("from", {})
            sender = from_obj.get("hash", str(from_obj)) if isinstance(from_obj, dict) else str(from_obj)

            # Filter out self-transfers (Contract -> Contract)
            if sender.lower() == contract_address.lower():
                continue

            # Categorize by amount
            if amount < 0.001:
                cat, order = "< 0.001", 1
            elif amount < 0.01:
                cat, order = "0.001 - 0.01", 2
            elif amount < 0.1:
                cat, order = "0.01 - 0.1", 3
            elif amount < 1:
                cat, order = "0.1 - 1", 4
            else:
                cat, order = "1 and above", 5

            lock_list.append({
                "date": dt.date(),
                "ts": dt,
                "amount": amount,
                "type": "lock",
                "sender": sender,
                "cat": cat,
                "order": order
            })
        except Exception as e:
            # Optionally log verbose errors
            continue

    return lock_list
