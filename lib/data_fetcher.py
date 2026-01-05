"""Fetch data from GitHub or local file."""
import json
import os
import urllib.request
from typing import Dict, Any


def fetch_data_json(local_path: str = "vebtc_data.json",
                    github_raw_url: str = None,
                    use_github: bool = None) -> Dict[str, Any]:
    """Fetch veBTC data from GitHub raw URL or local file.

    Args:
        local_path: Path to local JSON file
        github_raw_url: GitHub raw URL (e.g., https://raw.githubusercontent.com/user/repo/main/vebtc_data.json)
        use_github: Force GitHub fetch. If None, auto-detect (use GitHub if local file doesn't exist)

    Returns:
        Parsed JSON data
    """
    # Auto-detect: use GitHub if local file doesn't exist (e.g., on Railway)
    if use_github is None:
        use_github = not os.path.exists(local_path)

    if use_github and github_raw_url:
        print(f"Fetching data from GitHub: {github_raw_url}")
        try:
            with urllib.request.urlopen(github_raw_url) as response:
                data = json.loads(response.read().decode())
                print(f"Fetched data from GitHub successfully")
                return data
        except Exception as e:
            print(f"Error fetching from GitHub: {e}")
            # Fallback to local if available
            if os.path.exists(local_path):
                print(f"Falling back to local file: {local_path}")
            else:
                raise

    # Use local file
    print(f"Loading data from local file: {local_path}")
    with open(local_path, 'r') as f:
        return json.load(f)
