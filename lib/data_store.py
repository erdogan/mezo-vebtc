"""Data persistence module for veBTC dashboard."""
import json
import os
import tempfile
from typing import List, Dict, Any, Tuple


def load_data(data_file: str = "vebtc_data.json") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load existing data from JSON file.

    Args:
        data_file: Path to data file

    Returns:
        Tuple of (locks, votes) lists
    """
    if os.path.exists(data_file):
        print(f"Loading existing data from {data_file}...")
        try:
            with open(data_file, "r") as f:
                data = json.load(f)
                return data.get("locks", []), data.get("votes", [])
        except Exception as e:
            print(f"Error loading file: {e}. Starting fresh.")
    return [], []


def save_data(locks: List[Dict[str, Any]], votes: List[Dict[str, Any]],
              data_file: str = "vebtc_data.json") -> None:
    """Save combined data to JSON file atomically, preserving other fields.

    Args:
        locks: List of lock events
        votes: List of vote events
        data_file: Path to data file
    """
    print(f"Saving {len(locks)} locks and {len(votes)} votes to {data_file}...")
    # Load existing data to preserve epochs, participants, etc.
    existing = {}
    if os.path.exists(data_file):
        try:
            with open(data_file, "r") as f:
                existing = json.load(f)
        except Exception:
            pass

    existing["locks"] = locks
    existing["votes"] = votes

    # Atomic write: write to temp file then rename
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=os.path.dirname(os.path.abspath(data_file)) or "."
        ) as f:
            json.dump(existing, f, indent=2)
            temp_name = f.name
        os.replace(temp_name, data_file)
    except Exception as e:
        print(f"Error saving data: {e}")
        if 'temp_name' in locals() and os.path.exists(temp_name):
            os.remove(temp_name)


def load_extended_data(data_file: str = "vebtc_data.json") -> Dict[str, Any]:
    """Load full data structure including epochs and participants.

    Args:
        data_file: Path to data file

    Returns:
        Full data dictionary with locks, votes, epochs, participants
    """
    if os.path.exists(data_file):
        print(f"Loading existing data from {data_file}...")
        try:
            with open(data_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading file: {e}. Starting fresh.")

    return {
        "version": "2.0",
        "locks": [],
        "votes": [],
        "epochs": {},
        "participants": {},
        "pool_metadata": {}
    }


def save_extended_data(data: Dict[str, Any], data_file: str = "vebtc_data.json") -> None:
    """Save extended data structure atomically.

    Args:
        data: Full data dictionary
        data_file: Path to data file
    """
    locks_count = len(data.get("locks", []))
    votes_count = len(data.get("votes", []))
    epochs_count = len(data.get("epochs", {}))
    participants_count = len(data.get("participants", {}))

    print(f"Saving {locks_count} locks, {votes_count} votes, {epochs_count} epochs, "
          f"{participants_count} participants to {data_file}...")

    # Atomic write
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=os.path.dirname(os.path.abspath(data_file)) or "."
        ) as f:
            json.dump(data, f, indent=2)
            temp_name = f.name
        os.replace(temp_name, data_file)
    except Exception as e:
        print(f"Error saving data: {e}")
        if 'temp_name' in locals() and os.path.exists(temp_name):
            os.remove(temp_name)
