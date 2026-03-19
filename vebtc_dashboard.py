"""veBTC Dashboard - Refactored modular version."""
import os
from datetime import datetime, date
from typing import List, Dict, Any, Tuple

import pandas as pd

# Import our new modules
from lib.config import load_config
from lib.data_store import load_data, save_data, save_extended_data
from lib.fetchers.api_fetcher import fetch_incremental, fetch_current_balance
from lib.parsers.lock_parser import parse_locks
from lib.parsers.vote_parser import parse_votes
from lib.analytics.epoch_tracker import get_current_epoch_info
from lib.utils.time_utils import get_current_timestamp
from lib.utils.mezo_username import batch_resolve_usernames


def parse_data(locks: List[Dict[str, Any]],
                votes: List[Dict[str, Any]],
                config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse and merge both datasets.

    Args:
        locks: Raw lock transactions
        votes: Raw vote logs
        config: Configuration object

    Returns:
        Tuple of (df_main, dist_df, df_raw_locks, df_raw_votes)
    """
    # Parse using new modules
    lock_list = parse_locks(
        locks,
        contract_address=config.vebtc_address,
        default_decimals=config.default_decimals
    )

    vote_list = parse_votes(
        votes,
        voted_topic_0=config.voted_topic_0,
        default_decimals=config.default_decimals
    )

    # Create DataFrames
    df_locks = pd.DataFrame(lock_list)
    df_votes = pd.DataFrame(vote_list)

    # Aggregations
    if not df_locks.empty:
        daily_locks = df_locks.groupby("date").agg({
            "amount": "sum",
            "type": "count"
        }).rename(columns={"type": "lock_count"}).reset_index()
    else:
        daily_locks = pd.DataFrame(columns=["date", "amount", "lock_count"])

    if not df_votes.empty:
        daily_votes = df_votes.groupby("date").agg({
            "voting_power": "sum",
            "voter": "count"
        }).rename(columns={"voter": "vote_count"}).reset_index()
    else:
        daily_votes = pd.DataFrame(columns=["date", "voting_power", "vote_count"])

    # Merge on all dates
    all_dates = sorted(list(set(daily_locks["date"].tolist() + daily_votes["date"].tolist())))
    df_main = pd.DataFrame({"date": all_dates})

    df_main = df_main.merge(daily_locks, on="date", how="left").fillna(0)
    df_main = df_main.merge(daily_votes, on="date", how="left").fillna(0)

    df_main["cumulative_locks"] = df_main["amount"].cumsum()
    df_main["cumulative_votes"] = df_main["voting_power"].cumsum()

    # Distribution
    if not df_locks.empty:
        dist_df = df_locks.groupby(["cat", "order"]).agg({
            "type": "count",
            "amount": "sum"
        }).rename(columns={"type": "tx_count"}).reset_index()
        dist_df.sort_values("order", inplace=True)
        dist_df["legend"] = dist_df.apply(
            lambda r: f"{r['cat']} (Txs: {r['tx_count']}, Total: {r['amount']:.2f})",
            axis=1
        )
    else:
        dist_df = pd.DataFrame()

    # Raw data sorted by timestamp
    df_raw_locks = df_locks.sort_values("ts", ascending=False) if not df_locks.empty else pd.DataFrame()
    df_raw_votes = df_votes.sort_values("ts", ascending=False) if not df_votes.empty else pd.DataFrame()

    return df_main, dist_df, df_raw_locks, df_raw_votes


def generate_dashboard(locks: List[Dict[str, Any]],
                        votes: List[Dict[str, Any]],
                        current_balance: str,
                        total_voted: str,
                        total_supply: str,
                        epoch_info: Dict[str, Any],
                        incentives_data: List[Dict[str, Any]] = None,
                        previous_incentives_data: List[Dict[str, Any]] = None,
                        participant_data: Dict[str, Any] = None,
                        epochs_data: Dict[str, Any] = None,
                        lock_analytics_data: Dict[str, Any] = None) -> None:
    """Generate the HTML dashboard.

    Args:
        locks: Parsed lock records
        votes: Parsed vote records
        current_balance: Current BTC balance string
        total_voted: Total voted veBTC string
        total_supply: Total supply string
        epoch_info: Current epoch information
        incentives_data: Pool incentives data for current epoch (optional)
        previous_incentives_data: Pool incentives data for previous epoch (optional)
        participant_data: Participant analytics data (optional)
        epochs_data: Historical epoch data (optional)
        lock_analytics_data: Lock analytics data (optional)
    """
    print("Generating Dashboard...")

    def json_serial(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return str(obj)

    # Import epoch HTML generators
    from lib.generators.html_epoch import (
        generate_epoch_banner,
        generate_epoch_banner_css,
        generate_epoch_countdown_js
    )

    # Import incentives HTML generators
    from lib.generators.html_incentives import (
        generate_incentives_section,
        generate_incentives_css,
        generate_incentives_js
    )

    # Import participant HTML generators
    from lib.generators.html_leaderboards import (
        generate_leaderboards_section,
        generate_leaderboards_css
    )

    from lib.generators.html_past_epochs import (
        generate_past_epochs_section,
        generate_past_epochs_css
    )

    from lib.generators.html_search import (
        generate_search_bar,
        generate_search_css,
        generate_search_js
    )

    from lib.generators.html_lock_analytics import (
        generate_lock_analytics_section,
        generate_lock_analytics_css,
        generate_lock_analytics_js
    )

    from lib.generators.html_fees import (
        generate_fees_section,
        generate_fees_css,
        generate_fees_js
    )

    # Calculate epoch-specific metrics
    # Filter votes by current epoch voting window
    epoch_votes = []
    for v in votes:
        vote_ts = v.get('ts')
        if vote_ts:
            # Handle different timestamp formats
            if isinstance(vote_ts, str):
                from datetime import datetime
                dt = datetime.fromisoformat(vote_ts.replace('Z', '+00:00'))
                vote_ts = dt.timestamp()
            elif hasattr(vote_ts, 'timestamp'):
                vote_ts = vote_ts.timestamp()
            else:
                vote_ts = float(vote_ts)

            # Check if vote is within current epoch voting window
            if epoch_info['vote_start_ts'] <= vote_ts <= epoch_info['vote_end_ts']:
                epoch_votes.append(v)

    # Calculate unique voters in current epoch only
    unique_voters = len(set(v.get('voter') for v in epoch_votes if v.get('voter') != 'Unknown'))

    # Calculate total voted in current epoch
    epoch_total_voted = sum(v.get('voting_power', 0) for v in epoch_votes)

    # Generate epoch banner HTML
    epoch_banner_html = generate_epoch_banner(epoch_info, epoch_total_voted, unique_voters)
    epoch_css = generate_epoch_banner_css()
    epoch_js = generate_epoch_countdown_js()

    # Generate incentives HTML
    incentives_html = ""
    incentives_css = ""
    incentives_js = ""
    if incentives_data:
        incentives_html = generate_incentives_section(incentives_data, previous_incentives_data, epoch_info['epoch_number'])
        incentives_css = generate_incentives_css()
        incentives_js = generate_incentives_js()

    # Generate participant features HTML
    search_html = ""
    leaderboards_html = ""
    search_css = ""
    leaderboards_css = ""
    search_js = ""
    participants_json = "{}"

    if participant_data:
        search_html = generate_search_bar()
        search_css = generate_search_css()
        search_js = generate_search_js()

        if participant_data.get('leaderboards'):
            leaderboards_html = generate_leaderboards_section(
                participant_data['leaderboards']['top_lockers'],
                participant_data['leaderboards']['top_voters']
            )
            leaderboards_css = generate_leaderboards_css()

        # Export participants data as JSON for search
        import json
        if participant_data.get('all_participants'):
            participants_json = json.dumps(participant_data['all_participants'], default=json_serial)

    # Generate past epochs HTML and CSS
    past_epochs_html = ""
    past_epochs_css = ""
    if epochs_data:
        past_epochs_html = generate_past_epochs_section(epochs_data, epoch_info['epoch_number'])
        past_epochs_css = generate_past_epochs_css()

    # Generate lock analytics HTML, CSS, and JS
    lock_analytics_html = ""
    lock_analytics_css = ""
    lock_analytics_js = ""
    if lock_analytics_data:
        profiles = lock_analytics_data.get('profiles', [])
        statistics = lock_analytics_data.get('statistics', {})
        lock_analytics_html = generate_lock_analytics_section(profiles, statistics)
        lock_analytics_css = generate_lock_analytics_css()
        lock_analytics_js = generate_lock_analytics_js()

    # Generate fees HTML, CSS, and JS
    fees_html = ""
    fees_css = ""
    fees_js = ""
    if epochs_data:
        fees_html = generate_fees_section(epochs_data, epoch_info['epoch_number'])
        fees_css = generate_fees_css()
        fees_js = generate_fees_js(epochs_data)

    # For now, we'll still use the original template but inject all new sections
    # Read the original vebtc.py HTML generation
    from vebtc import generate_dashboard as original_generate_dashboard
    original_generate_dashboard(locks, votes, current_balance, total_voted, total_supply)

    # Now inject all new sections into the generated HTML
    with open("index.html", "r") as f:
        html_content = f.read()

    # Generate tabs CSS
    tabs_css = """
    /* Tabs Styles */
    .tabs-container {
        margin-top: 30px;
        margin-bottom: 30px;
    }

    .tabs-nav {
        display: flex;
        gap: 0;
        border-bottom: 2px solid #e1e8ed;
        margin-bottom: 25px;
    }

    .tab-btn {
        padding: 12px 24px;
        background: none;
        border: none;
        border-bottom: 3px solid transparent;
        font-size: 16px;
        font-weight: 600;
        color: #7f8c8d;
        cursor: pointer;
        transition: all 0.2s;
        position: relative;
        bottom: -2px;
    }

    .tab-btn:hover {
        color: #667eea;
        background: rgba(102, 126, 234, 0.05);
    }

    .tab-btn.active {
        color: #667eea;
        border-bottom-color: #667eea;
    }

    .tabs-content {
        position: relative;
    }

    .tab-panel {
        display: none;
    }

    .tab-panel.active {
        display: block;
        animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (max-width: 768px) {
        .tabs-nav {
            flex-direction: column;
            gap: 0;
            border-bottom: none;
        }

        .tab-btn {
            border-bottom: 1px solid #e1e8ed;
            border-left: 3px solid transparent;
            bottom: 0;
            text-align: left;
        }

        .tab-btn.active {
            border-bottom-color: #e1e8ed;
            border-left-color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }
    }
    """

    # Inject CSS into the <style> section
    combined_css = epoch_css + "\n" + incentives_css + "\n" + fees_css + "\n" + past_epochs_css + "\n" + lock_analytics_css + "\n" + leaderboards_css + "\n" + search_css + "\n" + tabs_css
    html_content = html_content.replace("</style>", combined_css + "\n    </style>")

    # Inject epoch banner after the header section
    header_end = html_content.find("</div>\n\n        <!-- Controls -->")
    if header_end != -1:
        controls_comment = "<!-- Controls -->"
        controls_pos = html_content.find(controls_comment, header_end)
        if controls_pos != -1:
            html_content = html_content[:header_end] + "</div>\n\n" + epoch_banner_html + "\n        " + html_content[controls_pos:]

    # Create tabbed interface
    # Find the controls marker to identify where stats content begins
    controls_marker = "<!-- Controls -->"
    controls_start = html_content.find(controls_marker)

    if controls_start != -1:
        # Find where the main script tag starts (end of stats content)
        # Look for the script that contains "// --- Raw Data ---"
        script_start_marker = '    <script>\n        // --- Raw Data ---'
        script_tag = html_content.find(script_start_marker)

        if script_tag != -1:
            # Extract everything from controls to script (this is stats content)
            stats_content = html_content[controls_start:script_tag]

            # Build tabs HTML
            tabs_html = """
        <!-- Tabs Navigation -->
        <div class="tabs-container">
            <div class="tabs-nav">
                <button class="tab-btn active" data-tab="stats">Stats</button>
                <button class="tab-btn" data-tab="incentives">Incentives</button>
                <button class="tab-btn" data-tab="fees">Fees</button>
                <button class="tab-btn" data-tab="past-epochs">Past Epochs</button>
                <button class="tab-btn" data-tab="leaderboards">Leaderboards</button>
                <button class="tab-btn" data-tab="lock-analytics">Lock Analytics</button>
            </div>

            <div class="tabs-content">
                <!-- Stats Tab (Default) -->
                <div class="tab-panel active" id="stats-panel">
""" + stats_content + """
                </div>

                <!-- Incentives Tab -->
                <div class="tab-panel" id="incentives-panel">
""" + (incentives_html if incentives_html else '<p class="empty-state">No incentive data available</p>') + """
                </div>

                <!-- Past Epochs Tab -->
                <div class="tab-panel" id="past-epochs-panel">
""" + (past_epochs_html if past_epochs_html else '<p class="empty-state">No historical epoch data available</p>') + """
                </div>

                <!-- Leaderboards Tab -->
                <div class="tab-panel" id="leaderboards-panel">
""" + (leaderboards_html if leaderboards_html else '<p class="empty-state">No leaderboard data available</p>') + """
                </div>

                <!-- Lock Analytics Tab -->
                <div class="tab-panel" id="lock-analytics-panel">
""" + (lock_analytics_html if lock_analytics_html else '<p class="empty-state">No lock analytics data available</p>') + """
                </div>

                <!-- Fees Tab -->
                <div class="tab-panel" id="fees-panel">
""" + (fees_html if fees_html else '<p class="empty-state">No fee data available</p>') + """
                </div>
            </div>
        </div>
"""

            # Build new sections (search, then tabs)
            new_sections = ""
            if search_html:
                new_sections += "\n\n        " + search_html
            new_sections += "\n" + tabs_html + "\n"

            # Replace the stats content with tabs (insert tabs at controls_start, keep script at script_tag)
            html_content = html_content[:controls_start] + new_sections + html_content[script_tag:]

    # Generate tabs JavaScript with URL routing
    tabs_js = """
    // Tab Switching with URL Hash Navigation
    document.addEventListener('DOMContentLoaded', function() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabPanels = document.querySelectorAll('.tab-panel');

        function switchToTab(tabName) {
            // Remove active class from all buttons and panels
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));

            // Add active class to target button and panel
            const targetBtn = document.querySelector(`.tab-btn[data-tab="${tabName}"]`);
            const targetPanel = document.getElementById(tabName + '-panel');

            if (targetBtn && targetPanel) {
                targetBtn.classList.add('active');
                targetPanel.classList.add('active');
            }
        }

        // Handle tab button clicks
        tabBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const targetTab = this.getAttribute('data-tab');
                // Update URL hash without scrolling
                history.pushState(null, null, '#' + targetTab);
                switchToTab(targetTab);
            });
        });

        // Handle browser back/forward buttons
        window.addEventListener('hashchange', function() {
            const hash = window.location.hash.substring(1); // Remove #
            if (hash) {
                switchToTab(hash);
            } else {
                switchToTab('stats'); // Default to stats
            }
        });

        // Load correct tab on page load based on URL hash
        const initialHash = window.location.hash.substring(1);
        if (initialHash && (initialHash === 'stats' || initialHash === 'incentives' || initialHash === 'fees' || initialHash === 'past-epochs' || initialHash === 'leaderboards' || initialHash === 'lock-analytics')) {
            switchToTab(initialHash);
        } else {
            // Ensure stats tab is active by default
            switchToTab('stats');
        }
    });
    """

    # Inject participants data and JavaScript before </script>
    if participants_json != "{}":
        participants_script = f"\n    // Participants data for search\n    window.PARTICIPANTS_DATA = {participants_json};\n"
        html_content = html_content.replace("    </script>", participants_script + "\n" + epoch_js + "\n" + incentives_js + "\n" + fees_js + "\n" + search_js + "\n" + lock_analytics_js + "\n" + tabs_js + "\n    </script>")
    else:
        html_content = html_content.replace("    </script>", "\n" + epoch_js + "\n" + incentives_js + "\n" + fees_js + "\n" + lock_analytics_js + "\n" + tabs_js + "\n    </script>")

    # Write back the modified HTML
    with open("index.html", "w") as f:
        f.write(html_content)


def main():
    """Main execution function."""
    # Load configuration
    config = load_config("config.json")

    print(f"Using RPC: {config.rpc_url}")
    print(f"veBTC Contract: {config.vebtc_address}")
    print(f"Voter Contract: {config.voter_address}")

    # 1. Load existing data
    existing_locks, existing_votes = load_data("vebtc_data.json")

    # 2. Fetch new data
    new_locks = fetch_incremental(
        url=config.lock_url,
        params={"filter": "to", "token": config.lock_token},
        existing_items=existing_locks,
        type_label="locks"
    )

    new_votes = fetch_incremental(
        url=config.vote_url,
        params={},
        existing_items=existing_votes,
        type_label="votes"
    )

    # 2b. Fetch deposit event logs
    from lib.fetchers.deposit_fetcher import fetch_deposit_logs
    from lib.data_store import load_extended_data

    existing_extended = load_extended_data("vebtc_data.json")
    existing_deposits = existing_extended.get('deposits', [])

    new_deposits = fetch_deposit_logs(
        vebtc_address=config.vebtc_address,
        existing_deposits=existing_deposits,
        explorer_api_base=config.get('network.explorer_api')
    )

    all_deposits = new_deposits + existing_deposits

    # 3. Fetch current balance
    address_details_url = f"{config.get('network.explorer_api')}/addresses/{config.vebtc_address}"
    current_balance = fetch_current_balance(address_details_url)

    # 4. Merge new + old
    all_locks = new_locks + existing_locks
    all_votes = new_votes + existing_votes

    # 5. Save raw data for backward compatibility
    if new_locks or new_votes:
        save_data(all_locks, all_votes, "vebtc_data.json")

    # 6. Parse data
    df_main, dist_df, raw_locks_df, raw_votes_df = parse_data(all_locks, all_votes, config)

    # Convert DataFrames to lists for JSON serialization
    # Convert datetime objects to ISO format strings for JSON compatibility
    import json

    locks_list = json.loads(raw_locks_df.to_json(orient='records', date_format='iso'))
    votes_list = json.loads(raw_votes_df.to_json(orient='records', date_format='iso'))

    # 6b. Parse deposit events
    from lib.parsers.deposit_parser import parse_deposits

    deposits_list = parse_deposits(all_deposits, default_decimals=18)

    # 7. Calculate totals
    # A. Total Voted: sum of latest totalWeight for each unique pool (gauge)
    latest_gauge_tw = {}
    for v in reversed(votes_list):
        pool = v.get('pool')
        if pool and pool != "Unknown":
            latest_gauge_tw[pool] = v.get('total_weight', 0)

    total_voted_val = sum(latest_gauge_tw.values())
    total_voted_str = f"{total_voted_val:,.2f}"

    # B. Total Supply: Voted + unvoted delta
    UNVOTED_DELTA = 2.1769
    total_supply_val = total_voted_val + UNVOTED_DELTA
    total_supply_str = f"{total_supply_val:,.6f}"

    # 8. Get current epoch info
    current_ts = get_current_timestamp()
    epoch_info = get_current_epoch_info(current_ts)
    print(f"\nEpoch {epoch_info['epoch_number']}: {epoch_info['time_remaining_formatted']} remaining")
    print(f"Voting: {'OPEN' if epoch_info['is_voting_open'] else 'CLOSED'}")

    # 9. Fetch and calculate incentives data (current + previous epoch)
    incentives_data = None
    previous_incentives_data = None
    try:
        print("\nFetching incentives data...")
        from lib.fetchers.rpc_fetcher import RPCFetcher
        from lib.fetchers.contract_fetcher import ContractFetcher
        from lib.fetchers.price_fetcher import PriceFetcher
        from lib.fetchers.cache_manager import CacheManager
        from lib.analytics.incentives import IncentivesCalculator

        # Initialize fetchers
        cache_manager = CacheManager()
        rpc_fetcher = RPCFetcher(config.rpc_url, retry_count=3)
        contract_fetcher = ContractFetcher(rpc_fetcher)
        price_fetcher = PriceFetcher(cache_manager)

        # Fetch token prices (BTC, USDC, USDT, WBTC, etc.)
        print("Fetching token prices...")
        token_prices = price_fetcher.get_prices(["BTC", "USDC", "USDT", "WBTC", "ETH"])
        print(f"Token prices: {token_prices}")

        # Calculate APRs for each pool
        incentives_calculator = IncentivesCalculator(token_prices)

        # Fetch pool incentives from contracts for CURRENT epoch
        print("Querying on-chain incentives for current epoch...")
        pools_raw = contract_fetcher.get_all_pool_incentives(config.voter_address, current_ts)
        print(f"Found {len(pools_raw)} pools with incentives")

        incentives_data = []

        for pool_raw in pools_raw:
            # Get pool name (token pair)
            pool_name = contract_fetcher.get_pool_name(pool_raw["pool_address"])

            # Get token symbols for bribes
            bribes_tokens = {}
            for token_addr, amount in pool_raw["bribes"]["amounts"].items():
                symbol = contract_fetcher.get_token_symbol(token_addr)
                bribes_tokens[symbol] = amount

            # Get token symbols for fees
            fees_tokens = {}
            for token_addr, amount in pool_raw["fees"]["amounts"].items():
                symbol = contract_fetcher.get_token_symbol(token_addr)
                fees_tokens[symbol] = amount

            # Calculate pool incentives with APR
            pool_incentives = incentives_calculator.calculate_pool_incentives(
                pool_address=pool_raw["pool_address"],
                pool_name=pool_name,
                current_votes=pool_raw["voting_weight"],
                current_epoch_bribes=bribes_tokens,
                current_epoch_fees=fees_tokens,
                historical_fees=None  # TODO: Fetch historical fees for better fee APR
            )

            # Convert to dict for HTML generation
            incentives_data.append({
                "pool_address": pool_incentives.pool_address,
                "pool_name": pool_incentives.pool_name,
                "current_votes": pool_incentives.current_votes,
                "bribes": pool_incentives.bribes,
                "bribes_usd": pool_incentives.bribes_usd,
                "fees": pool_incentives.fees,
                "fees_usd": pool_incentives.fees_usd,
                "apr_bribes": pool_incentives.apr_bribes,
                "apr_fees": pool_incentives.apr_fees,
                "apr_total": pool_incentives.apr_total,
                "usd_per_vote": pool_incentives.usd_per_vote
            })

        print(f"Calculated APRs for {len(incentives_data)} pools (current epoch)")

        # Load PREVIOUS epoch incentives from cached data
        # Note: We don't query blockchain for previous epoch because contracts return current state,
        # not historical state. Instead, we use cached data from epochs_data.
        try:
            print("\nLoading previous epoch incentives from cache...")
            previous_incentives_data = []

            # Load existing extended data to get cached previous epoch
            from lib.data_store import load_extended_data
            existing_extended = load_extended_data("vebtc_data.json")
            existing_epochs = existing_extended.get('epochs', {})

            previous_epoch_key = str(epoch_info['epoch_number'] - 1)
            if previous_epoch_key in existing_epochs:
                previous_epoch = existing_epochs[previous_epoch_key]
                previous_pools = previous_epoch.get('incentives', {}).get('pools', [])

                # Convert cached pool data to the format expected by HTML generator
                for pool in previous_pools:
                    # Reconstruct bribes dict (we only have USD total in cache)
                    previous_incentives_data.append({
                        "pool_address": pool.get('pool_address'),
                        "pool_name": pool.get('pool_name'),
                        "current_votes": pool.get('votes', 0),
                        "bribes": {},  # We don't have token breakdown in cache
                        "bribes_usd": pool.get('bribes_usd', 0),
                        "fees": {},  # We don't have token breakdown in cache
                        "fees_usd": pool.get('fees_usd', 0),
                        "apr_bribes": 0,  # Not stored separately
                        "apr_fees": 0,  # Not stored separately
                        "apr_total": pool.get('apr_total', 0),
                        "usd_per_vote": (pool.get('bribes_usd', 0) + pool.get('fees_usd', 0)) / pool.get('votes', 1) if pool.get('votes', 0) > 0 else 0
                    })

                print(f"Loaded {len(previous_incentives_data)} pools from cached previous epoch (Epoch {previous_epoch_key})")
            else:
                print(f"No cached data for previous epoch (Epoch {previous_epoch_key}), skipping comparison")

        except Exception as e:
            print(f"Warning: Failed to fetch previous epoch incentives: {e}")
            previous_incentives_data = None

        # If no pools found (e.g., due to rate limiting), set to None
        if len(incentives_data) == 0:
            print("No pools found (likely due to RPC rate limiting)")
            incentives_data = None

    except Exception as e:
        print(f"Warning: Failed to fetch incentives data: {e}")
        print("Dashboard will be generated without incentives section")
        incentives_data = None
        previous_incentives_data = None

    # 9b. Fetch and aggregate historical epoch data
    epochs_data = {}
    try:
        print("\nAggregating historical epoch data...")
        from lib.analytics.epoch_aggregator import EpochAggregator
        from lib.analytics.epoch_tracker import get_epoch_info_by_number
        from lib.data_store import load_extended_data

        aggregator = EpochAggregator(votes_list)
        current_epoch_num = epoch_info['epoch_number']

        # Determine which epochs to fetch
        existing_extended = load_extended_data("vebtc_data.json")
        existing_epochs = existing_extended.get('epochs', {})

        # Strategy: Only fetch last 10 epochs, reuse existing data where possible
        epochs_to_fetch = []
        for i in range(10):
            epoch_num = current_epoch_num - i
            if epoch_num >= 0:
                epoch_key = str(epoch_num)

                # Skip if already cached and not current/previous epoch
                if epoch_key in existing_epochs and epoch_num not in [current_epoch_num, current_epoch_num - 1]:
                    epochs_data[epoch_key] = existing_epochs[epoch_key]
                    print(f"  Using cached data for epoch {epoch_num}")
                else:
                    epochs_to_fetch.append(epoch_num)

        # Fetch incentives for uncached epochs
        for epoch_num in epochs_to_fetch:
            try:
                print(f"  Fetching epoch {epoch_num}...")

                # Get epoch timestamp
                epoch_info_hist = get_epoch_info_by_number(epoch_num)
                epoch_ts = epoch_info_hist['start_ts']

                # Aggregate votes for this epoch
                vote_metrics = aggregator.aggregate_votes_by_epoch(epoch_num)

                # Fetch incentives for this epoch
                pools_raw = contract_fetcher.get_all_pool_incentives(config.voter_address, epoch_ts)

                epoch_incentives = []
                for pool_raw in pools_raw:
                    pool_name = contract_fetcher.get_pool_name(pool_raw["pool_address"])

                    bribes_tokens = {}
                    for token_addr, amount in pool_raw["bribes"]["amounts"].items():
                        symbol = contract_fetcher.get_token_symbol(token_addr)
                        bribes_tokens[symbol] = amount

                    fees_tokens = {}
                    for token_addr, amount in pool_raw["fees"]["amounts"].items():
                        symbol = contract_fetcher.get_token_symbol(token_addr)
                        fees_tokens[symbol] = amount

                    pool_incentives = incentives_calculator.calculate_pool_incentives(
                        pool_address=pool_raw["pool_address"],
                        pool_name=pool_name,
                        current_votes=pool_raw["voting_weight"],
                        current_epoch_bribes=bribes_tokens,
                        current_epoch_fees=fees_tokens,
                        historical_fees=None
                    )

                    epoch_incentives.append({
                        "pool_address": pool_incentives.pool_address,
                        "pool_name": pool_incentives.pool_name,
                        "current_votes": pool_incentives.current_votes,
                        "bribes_usd": pool_incentives.bribes_usd,
                        "fees_usd": pool_incentives.fees_usd,
                        "apr_total": pool_incentives.apr_total
                    })

                # Calculate complete metrics
                epoch_metrics = aggregator.calculate_epoch_metrics(
                    epoch_num,
                    vote_metrics,
                    epoch_incentives
                )

                # Convert to dict for storage
                epochs_data[str(epoch_num)] = {
                    "epoch_number": epoch_metrics.epoch_number,
                    "start_ts": epoch_metrics.start_ts,
                    "end_ts": epoch_metrics.end_ts,
                    "start_date": epoch_metrics.start_date,
                    "end_date": epoch_metrics.end_date,
                    "votes": {
                        "total_voted": epoch_metrics.total_voted,
                        "unique_voters": epoch_metrics.unique_voters,
                        "vote_tx_count": epoch_metrics.vote_tx_count
                    },
                    "incentives": {
                        "total_bribes_usd": epoch_metrics.total_bribes_usd,
                        "total_fees_usd": epoch_metrics.total_fees_usd,
                        "average_apr": epoch_metrics.average_apr,
                        "pool_count": epoch_metrics.pool_count,
                        "pools": epoch_metrics.pools
                    }
                }

                print(f"  ✓ Epoch {epoch_num}: {vote_metrics['vote_tx_count']} votes, {len(epoch_incentives)} pools")

            except Exception as e:
                print(f"  Warning: Failed to fetch epoch {epoch_num}: {e}")
                continue

        print(f"Historical epochs aggregated: {len(epochs_data)} epochs")

    except Exception as e:
        print(f"Warning: Failed to aggregate epoch data: {e}")
        epochs_data = {}

    # 10. Calculate participant analytics
    participant_data = None
    try:
        print("\nCalculating participant analytics...")
        from lib.analytics.participant import ParticipantAnalyzer

        analyzer = ParticipantAnalyzer(locks_list, votes_list)

        # Get top participants
        top_lockers = analyzer.get_top_lockers(limit=20)
        top_voters = analyzer.get_top_voters(limit=20)

        print(f"Top 20 lockers calculated")
        print(f"Top 20 voters calculated")

        # Get all participants for search
        all_participants = analyzer.get_all_participants()
        print(f"Total participants: {len(all_participants)}")

        # Resolve usernames for participants
        print("Resolving Mezo usernames...")
        all_addresses = list(all_participants.keys())
        username_map = batch_resolve_usernames(all_addresses)
        print(f"Resolved {len(username_map)} usernames")

        # Assign usernames to profiles
        for addr, mezo_id in username_map.items():
            if addr in all_participants:
                all_participants[addr].mezo_id = mezo_id

        # Convert to dicts for HTML generation
        participant_data = {
            'leaderboards': {
                'top_lockers': [
                    {
                        'lock_rank': p.lock_rank,
                        'address': p.address,
                        'mezo_id': p.mezo_id,
                        'total_locked': p.total_locked,
                        'num_locks': p.num_locks
                    } for p in top_lockers
                ],
                'top_voters': [
                    {
                        'vote_rank': p.vote_rank,
                        'address': p.address,
                        'mezo_id': p.mezo_id,
                        'current_voting_power': p.current_voting_power,
                        'total_votes_cast': p.total_votes_cast,
                        'pools_voted': p.pools_voted
                    } for p in top_voters
                ]
            },
            'all_participants': {
                addr: {
                    'address': p.address,
                    'mezo_id': p.mezo_id,
                    'total_locked': p.total_locked,
                    'num_locks': p.num_locks,
                    'current_voting_power': p.current_voting_power,
                    'total_votes_cast': p.total_votes_cast,
                    'token_ids': p.token_ids,
                    'pools_voted': p.pools_voted,
                    'first_lock_date': p.first_lock_date,
                    'last_lock_date': p.last_lock_date,
                    'first_vote_date': p.first_vote_date,
                    'last_vote_date': p.last_vote_date
                } for addr, p in all_participants.items()
            }
        }

        # Get statistics
        stats = analyzer.get_statistics()
        print(f"  - {stats['participants_with_locks']} lockers")
        print(f"  - {stats['participants_with_votes']} voters")

    except Exception as e:
        print(f"Warning: Failed to calculate participant analytics: {e}")
        print("Dashboard will be generated without participant features")
        participant_data = None

    # 10b. Calculate lock analytics
    lock_analytics_data = None
    try:
        print("\nCalculating lock analytics...")
        from lib.analytics.lock_analytics import LockAnalyzer

        # Pass both deposits (for duration info) and locks (for ownership)
        analyzer = LockAnalyzer(deposits_list, locks_list)

        # Get wallet profiles sorted by lock count (more useful when no max locks exist)
        top_profiles = analyzer.get_top_by_lock_count(limit=200)

        # Get statistics
        stats = analyzer.get_statistics()
        print(f"  - {stats['total_wallets']} wallets")
        print(f"  - {stats['total_locks']} locks")
        print(f"  - {stats['total_max_locks']} max-duration locks")
        print(f"  - {stats['max_lock_rate']:.1f}% max lock rate")

        lock_analytics_data = {
            'profiles': top_profiles,
            'statistics': stats
        }

    except Exception as e:
        print(f"Warning: Failed to calculate lock analytics: {e}")
        import traceback
        traceback.print_exc()
        print("Dashboard will be generated without lock analytics features")
        lock_analytics_data = None

    # 11. Generate dashboard
    generate_dashboard(locks_list, votes_list, current_balance, total_voted_str, total_supply_str, epoch_info, incentives_data, previous_incentives_data, participant_data, epochs_data, lock_analytics_data)

    # 12. Save extended data including incentives and parsed data for Telegram bot
    try:
        extended_data = {
            "version": "2.2",
            "locks": all_locks,  # Raw locks for backward compatibility
            "votes": all_votes,  # Raw votes for backward compatibility
            "deposits": all_deposits,  # Raw deposit event logs
            "parsed_locks": locks_list,  # Parsed locks for Telegram bot
            "parsed_votes": votes_list,  # Parsed votes for Telegram bot
            "parsed_deposits": deposits_list,  # Parsed deposit events
            "incentives": incentives_data if incentives_data else [],
            "previous_incentives": previous_incentives_data if previous_incentives_data else [],
            "epochs": epochs_data,  # Historical epoch data
            "last_updated": get_current_timestamp()
        }
        save_extended_data(extended_data, "vebtc_data.json")
        print(f"\nSaved extended data: {len(locks_list)} parsed locks, {len(votes_list)} parsed votes, {len(incentives_data) if incentives_data else 0} pools")
    except Exception as e:
        print(f"Warning: Failed to save extended data: {e}")

    print("\nDashboard generated successfully!")
    print(f"Total BTC Locked: {current_balance}")
    print(f"Total Voted: {total_voted_str} veBTC")
    print(f"Locks processed: {len(locks_list)}")
    print(f"Votes processed: {len(votes_list)}")


if __name__ == "__main__":
    main()
