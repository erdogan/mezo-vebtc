"""Parser for vote events."""
from datetime import datetime
from typing import List, Dict, Any


def parse_votes(votes: List[Dict[str, Any]],
                voted_topic_0: str,
                default_decimals: int = 18) -> List[Dict[str, Any]]:
    """Parse vote event logs.

    Args:
        votes: Raw vote logs from API
        voted_topic_0: Topic 0 hash for Voted event
        default_decimals: Token decimal places

    Returns:
        List of parsed vote records
    """
    vote_list = []

    print(f"Processing {len(votes)} logs for Votes...")

    for log in votes:
        try:
            # 1. Filter by Topic 0
            topics = log.get("topics", [])
            if not topics or topics[0] != voted_topic_0:
                continue

            # 2. Extract Data
            weight_val = 0.0
            total_weight_val = 0.0
            voter_addr = "Unknown"
            pool_addr = "Unknown"
            event_ts = None
            found_weight = False

            decoded = log.get("decoded")

            # METHOD A: Decoded
            if decoded and isinstance(decoded, dict):
                params = decoded.get("parameters", [])
                for p in params:
                    name = p.get("name", "")
                    val = p.get("value")
                    if name == "weight":
                        weight_val = float(val)
                        found_weight = True
                    elif name == "totalWeight":
                        total_weight_val = float(val)
                    elif name == "voter":
                        voter_addr = str(val)
                    elif name == "pool":
                        pool_addr = str(val)
                    elif name == "timestamp":
                        event_ts = int(val)

            # METHOD B: Hex Backup
            if not found_weight:
                data_hex = log.get("data", "").replace("0x", "")
                if len(data_hex) >= 192:
                    weight_val = float(int(data_hex[0:64], 16))
                    total_weight_val = float(int(data_hex[64:128], 16))
                    event_ts = int(data_hex[128:192], 16)
                    if len(topics) > 1:
                        voter_addr = "0x" + topics[1][26:]
                    if len(topics) > 2:
                        pool_addr = "0x" + topics[2][26:]
                    found_weight = True

            # 3. Timestamp
            if event_ts and event_ts > 0:
                dt = datetime.fromtimestamp(event_ts)
            else:
                ts_str = log.get("timestamp") or log.get("block_timestamp")
                if ts_str:
                    dt = datetime.strptime(ts_str.split('.')[0].replace('Z', ''), "%Y-%m-%dT%H:%M:%S")
                else:
                    continue

            # 4. Save
            if found_weight and weight_val > 0:
                vote_list.append({
                    "date": dt.date(),
                    "ts": dt,
                    "voting_power": weight_val / (10 ** default_decimals),
                    "total_weight": total_weight_val / (10 ** default_decimals),
                    "voter": voter_addr,
                    "pool": pool_addr,
                    "type": "vote"
                })

        except Exception as e:
            continue

    print(f"Parsed {len(vote_list)} valid vote events.")

    return vote_list
