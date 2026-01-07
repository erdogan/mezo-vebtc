"""Parser for Deposit events from veBTC contract."""
from datetime import datetime
from typing import List, Dict, Any, Optional

# DepositType enum from Velodrome VotingEscrow
DEPOSIT_TYPE_CREATE = 0
DEPOSIT_TYPE_DEPOSIT_FOR = 1
DEPOSIT_TYPE_INCREASE_AMOUNT = 2
DEPOSIT_TYPE_INCREASE_UNLOCK_TIME = 3

# Max lock duration range (tied to epoch length of 4 weeks = 28 days)
# Depending on when you lock within the epoch, max lock is 22-28 days
MAX_LOCK_MIN_DURATION = 1900800  # 22 days in seconds
MAX_LOCK_MAX_DURATION = 2419200  # 28 days in seconds


def get_deposit_type_name(deposit_type: int) -> str:
    """Get human-readable deposit type name."""
    names = {
        0: "CREATE",
        1: "DEPOSIT_FOR",
        2: "INCREASE_AMOUNT",
        3: "INCREASE_UNLOCK_TIME"
    }
    return names.get(deposit_type, "UNKNOWN")


def parse_deposit_event(log: Dict[str, Any], default_decimals: int = 18) -> Optional[Dict[str, Any]]:
    """Parse single Deposit event log.

    Args:
        log: Raw log from API
        default_decimals: Token decimals (BTC = 18)

    Returns:
        Parsed deposit record or None if invalid
    """
    try:
        decoded = log.get("decoded", {})
        if not decoded:
            return None

        # Extract from decoded parameters
        params_list = decoded.get("parameters", [])
        params = {p["name"]: p["value"] for p in params_list}

        provider = params.get("provider")
        if provider:
            provider = provider.lower()

        token_id = params.get("tokenId")
        if token_id:
            try:
                token_id = int(token_id)
            except (ValueError, TypeError):
                token_id = 0

        deposit_type = params.get("depositType")
        if deposit_type is not None:
            try:
                deposit_type = int(deposit_type)
            except (ValueError, TypeError):
                deposit_type = 0
        else:
            deposit_type = 0

        value_raw = params.get("value", "0")
        try:
            value = float(value_raw) / (10 ** default_decimals)
        except (ValueError, TypeError):
            value = 0.0

        locktime = params.get("locktime")
        try:
            locktime = int(locktime) if locktime else 0
        except (ValueError, TypeError):
            locktime = 0

        ts = params.get("ts")
        try:
            ts = int(ts) if ts else 0
        except (ValueError, TypeError):
            ts = 0

        # Get transaction hash for matching
        tx_hash = log.get("transaction_hash") or log.get("tx_hash")

        # Calculate lock duration
        if locktime > 0 and ts > 0:
            duration = locktime - ts
            # Check if it's a max lock (22-28 days, tied to epoch length)
            is_max_lock = MAX_LOCK_MIN_DURATION <= duration <= MAX_LOCK_MAX_DURATION
        else:
            duration = 0
            is_max_lock = False

        # Parse timestamps
        try:
            dt = datetime.fromtimestamp(ts) if ts > 0 else None
        except (ValueError, OSError):
            dt = None

        try:
            unlock_dt = datetime.fromtimestamp(locktime) if locktime > 0 else None
        except (ValueError, OSError):
            unlock_dt = None

        return {
            "tx_hash": tx_hash,
            "provider": provider,
            "token_id": token_id,
            "deposit_type": deposit_type,
            "deposit_type_name": get_deposit_type_name(deposit_type),
            "value": value,
            "locktime": locktime,
            "unlock_date": unlock_dt.isoformat() if unlock_dt else None,
            "timestamp": ts,
            "date": dt.isoformat() if dt else None,
            "duration_seconds": duration,
            "duration_days": round(duration / 86400, 2) if duration > 0 else 0,
            "is_max_lock": is_max_lock
        }

    except Exception as e:
        print(f"Error parsing deposit event: {e}")
        return None


def parse_deposits(deposits: List[Dict[str, Any]],
                   default_decimals: int = 18) -> List[Dict[str, Any]]:
    """Parse list of deposit logs.

    Args:
        deposits: Raw deposit logs
        default_decimals: Token decimals

    Returns:
        List of parsed deposit records
    """
    parsed = []

    print(f"Processing {len(deposits)} deposit events...")

    for log in deposits:
        parsed_deposit = parse_deposit_event(log, default_decimals)
        if parsed_deposit:
            parsed.append(parsed_deposit)

    print(f"Parsed {len(parsed)} valid deposit events")
    return parsed
