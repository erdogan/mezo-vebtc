import os
import json
import time
import tempfile
from datetime import datetime, date
from typing import List, Dict, Any, Tuple, Set

import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration ---
DATA_FILE = "vebtc_data.json"
CONTRACT_ADDRESS = "0x3D4b1b884A7a1E59fE8589a3296EC8f8cBB6f279"
LOCK_URL = f"https://api.explorer.mezo.org/api/v2/addresses/{CONTRACT_ADDRESS}/token-transfers"
LOCK_TOKEN = "0x7b7C000000000000000000000000000000000000"
VOTE_URL = "https://api.explorer.mezo.org/api/v2/addresses/0x48233cCC97B87Ba93bCA212cbEe48e3210211f03/logs"
# Voted Topic 0
# Topics
VOTED_TOPIC_0 = "0x452d440efc30dfa14a0ef803ccb55936af860ec6a6960ed27f129bef913f296a"

ADDRESS_DETAILS_URL = f"https://api.explorer.mezo.org/api/v2/addresses/{CONTRACT_ADDRESS}"
VEBTC_LOGS_URL = f"https://api.explorer.mezo.org/api/v2/addresses/{CONTRACT_ADDRESS}/logs"
DEFAULT_DECIMALS = 18

def load_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads existing data from JSON file."""
    if os.path.exists(DATA_FILE):
        print(f"Loading existing data from {DATA_FILE}...")
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                return data.get("locks", []), data.get("votes", [])
        except Exception as e:
            print(f"Error loading file: {e}. Starting fresh.")
    return [], []

def save_data(locks: List[Dict[str, Any]], votes: List[Dict[str, Any]]) -> None:
    """Saves combined data to JSON file atomically."""
    print(f"Saving {len(locks)} locks and {len(votes)} votes to {DATA_FILE}...")
    # Atomic write: write to temp file then rename
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(os.path.abspath(DATA_FILE)) or ".") as f:
            json.dump({"locks": locks, "votes": votes}, f, indent=2)
            temp_name = f.name
        os.replace(temp_name, DATA_FILE)
    except Exception as e:
        print(f"Error saving data: {e}")
        if 'temp_name' in locals() and os.path.exists(temp_name):
            os.remove(temp_name)

def get_unique_id(item: Dict[str, Any]) -> str:
    """Generates a unique ID for deduplication."""
    # Locks use 'tx_hash' + 'log_index'
    # Logs use 'transaction_hash' + 'index'
    uid = item.get("tx_hash") or item.get("transaction_hash") or item.get("hash")
    idx = str(item.get("index", item.get("log_index", "0")))
    return f"{uid}_{idx}"

def fetch_incremental(url: str, params: Dict[str, Any], existing_items: List[Dict[str, Any]], type_label: str = "items") -> List[Dict[str, Any]]:
    """Fetches only NEW items until a known item is found."""
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
                    existing_ids.add(uid) # Add to set to prevent dups in same run
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

def fetch_current_balance() -> str:
    """Fetches the current coin balance of the veBTC contract."""
    print("Fetching current balance...")
    try:
        response = requests.get(ADDRESS_DETAILS_URL)
        response.raise_for_status()
        data = response.json()
        raw_balance = data.get("coin_balance", "0")
        # remove 18 digits, 2 decimals
        balance_val = float(raw_balance) / (10 ** DEFAULT_DECIMALS)
        return f"{balance_val:,.2f}"
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return "N/A"

def parse_data(locks: List[Dict[str, Any]], votes: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parses and merges both datasets."""
    
    # --- 1. Process Locks ---
    lock_list = []
    
    print(f"Processing {len(locks)} lock records...")
    
    for tx in locks:
        try:
            ts = tx.get("timestamp")
            if not ts: continue
            dt = datetime.strptime(ts.split('.')[0].replace('Z', ''), "%Y-%m-%dT%H:%M:%S")
            
            raw_val = tx.get("total")
            if isinstance(raw_val, dict): raw_val = raw_val.get("value")
            amount = float(raw_val or 0) / (10 ** DEFAULT_DECIMALS)
            
            from_obj = tx.get("from", {})
            sender = from_obj.get("hash", str(from_obj)) if isinstance(from_obj, dict) else str(from_obj)

            # Filter out self-transfers (Contract -> Contract)
            if sender.lower() == CONTRACT_ADDRESS.lower():
                continue

            if amount < 0.001: cat, order = "< 0.001", 1
            elif amount < 0.01: cat, order = "0.001 - 0.01", 2
            elif amount < 0.1: cat, order = "0.01 - 0.1", 3
            elif amount < 1: cat, order = "0.1 - 1", 4
            else: cat, order = "1 and above", 5

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

    # --- 2. Process Votes ---
    vote_list = []
    print(f"Processing {len(votes)} logs for Votes...")

    for log in votes:
        try:
            # 1. Filter by Topic 0
            topics = log.get("topics", [])
            if not topics or topics[0] != VOTED_TOPIC_0:
                continue
            
            # 2. Extract Data
            weight_val = 0.0
            total_weight_val = 0.0
            voter_addr = "Unknown"
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
                    elif name == "timestamp":
                        event_ts = int(val)

            # METHOD B: Hex Backup
            if not found_weight:
                data_hex = log.get("data", "").replace("0x", "")
                if len(data_hex) >= 192:
                    weight_val = float(int(data_hex[0:64], 16))
                    total_weight_val = float(int(data_hex[64:128], 16))
                    event_ts = int(data_hex[128:192], 16)
                    if len(topics) > 1: voter_addr = "0x" + topics[1][26:]
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
                    "voting_power": weight_val / (10 ** DEFAULT_DECIMALS),
                    "total_weight": total_weight_val / (10 ** DEFAULT_DECIMALS),
                    "voter": voter_addr,
                    "type": "vote"
                })

        except Exception as e: 
            continue

    print(f"Parsed {len(vote_list)} valid vote events.")
    
    # --- 3. DataFrames ---
    df_locks = pd.DataFrame(lock_list)
    df_votes = pd.DataFrame(vote_list)
    
    # --- 4. Aggregations ---
    if not df_locks.empty:
        daily_locks = df_locks.groupby("date").agg({"amount": "sum", "type": "count"}).rename(columns={"type": "lock_count"}).reset_index()
    else:
        daily_locks = pd.DataFrame(columns=["date", "amount", "lock_count"])

    if not df_votes.empty:
        daily_votes = df_votes.groupby("date").agg({"voting_power": "sum", "voter": "count"}).rename(columns={"voter": "vote_count"}).reset_index()
    else:
        daily_votes = pd.DataFrame(columns=["date", "voting_power", "vote_count"])

    # --- 5. Merge ---
    # Union of all dates
    all_dates = sorted(list(set(daily_locks["date"].tolist() + daily_votes["date"].tolist())))
    df_main = pd.DataFrame({"date": all_dates})
    
    df_main = df_main.merge(daily_locks, on="date", how="left").fillna(0)
    df_main = df_main.merge(daily_votes, on="date", how="left").fillna(0)
    
    df_main["cumulative_locks"] = df_main["amount"].cumsum()
    
    df_main["cumulative_votes"] = df_main["voting_power"].cumsum()
    
    # --- 6. Distribution ---
    if not df_locks.empty:
        dist_df = df_locks.groupby(["cat", "order"]).agg({"type": "count", "amount": "sum"}).rename(columns={"type": "tx_count"}).reset_index()
        dist_df.sort_values("order", inplace=True)
        dist_df["legend"] = dist_df.apply(lambda r: f"{r['cat']} (Txs: {r['tx_count']}, Total: {r['amount']:.2f})", axis=1)
    else:
        dist_df = pd.DataFrame()

    # --- 7. Raw ---
    df_raw_locks = df_locks.sort_values("ts", ascending=False) if not df_locks.empty else pd.DataFrame()
    df_raw_votes = df_votes.sort_values("ts", ascending=False) if not df_votes.empty else pd.DataFrame()

    return df_main, dist_df, df_raw_locks, df_raw_votes

def generate_dashboard(locks: List[Dict[str, Any]], votes: List[Dict[str, Any]], current_balance: str) -> None:
    print("Generating Dashboard...")
    
    def json_serial(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return str(obj)

    # HTML Template
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>veBTC Dashboard</title>
    <meta http-equiv="refresh" content="60">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background: #f4f4f4; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 60px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; margin-top: 0; margin-bottom: 10px; padding-top: 20px; }}
        h2, h3 {{ color: #333; margin-top: 0; margin-bottom: 10px; }}
        .header {{ display: flex; justify-content: space-between; align-items: start; border-bottom: 2px solid #eee; padding-bottom: 30px; margin-bottom: 50px; gap: 80px; }}
        .header-left {{ display: flex; flex-direction: column; gap: 10px; }}
        .controls {{ display: flex; gap: 15px; align-items: center; background: #f8f9fa; padding: 15px; border-radius: 6px; margin-bottom: 20px; border: 1px solid #ddd; flex-wrap: wrap; }}
        .control-group {{ display: flex; flex-direction: column; gap: 5px; }}
        .control-group label {{ font-size: 12px; font-weight: 600; color: #666; text-transform: uppercase; }}
        input[type="date"] {{ padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }}
        
        /* Grid */
        .grid-row {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .col {{ flex: 1; min-width: 300px; }}
        
        .card {{ background: white; border: 1px solid #eee; border-radius: 6px; padding: 15px; height: 100%; box-sizing: border-box; }}
        .card h3 {{ font-size: 16px; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }}
        
        /* Mobile Resp */
        @media (max-width: 1024px) {{
            .grid-row {{ flex-direction: column; }}
            .header {{ flex-direction: column; gap: 20px; text-align: center; align-items: center; }}
            .header-left {{ align-items: center; }}
        }}

        /* Table Styles */
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #444; white-space: nowrap; }}
        tr:hover {{ background: #f5f5f5; cursor: pointer; }}
        
        /* Buttons */
        .btn {{ padding: 6px 12px; border: 1px solid #ccc; background: white; border-radius: 4px; cursor: pointer; font-size: 12px; text-decoration: none; color: #333; display: inline-block; }}
        .btn:hover {{ background: #f0f0f0; }}
        
        .legend-color {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 8px; vertical-align: middle; }}
        .mono {{ font-family: "SF Mono", "Monaco", "Consolas", monospace; }}
        
        .summary-stats {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .stat-box {{ background: #f8f9fa; padding: 15px; border-radius: 6px; flex: 1; text-align: center; border: 1px solid #ddd; }}
        .stat-value {{ font-size: 24px; font-weight: 700; color: #2c3e50; }}
        .stat-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 5px; }}

        /* Tabs */
        .tabs {{ display: flex; border-bottom: 1px solid #ddd; margin-bottom: 20px; }}
        .tab {{ padding: 10px 20px; cursor: pointer; border-bottom: 2px solid transparent; font-weight: 500; color: #666; }}
        .tab.active {{ border-bottom: 2px solid #007bff; color: #007bff; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-left">
                <h1 style="margin-bottom: 0;">veBTC Dashboard</h1>
                <div style="font-size: 14px; color: #666;">Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
            </div>
            <div class="stat-box" style="min-width: 200px;">
                 <div class="stat-value">{current_balance} BTC</div>
                 <div class="stat-label">Current Total Locked (On-Chain)</div>
            </div>
        </div>

        <!-- Controls -->
        <div class="controls">
            <div class="control-group">
                <label>Start Date</label>
                <input type="date" id="startDate" onchange="updateDashboard()">
            </div>
            <div class="control-group">
                <label>End Date</label>
                <input type="date" id="endDate" onchange="updateDashboard()">
            </div>
             <div class="control-group" style="margin-left: auto;">
                <button onclick="resetDates()" style="padding: 8px 16px; background: #fff; border: 1px solid #ccc; cursor: pointer; border-radius: 4px;">Reset Range</button>
            </div>
        </div>

        <!-- Summary Stats (Filtered) -->
        <div class="summary-stats">
            <div class="stat-box">
                <div class="stat-value" id="disp-locked">0.00</div>
                <div class="stat-label">Locked in Range (BTC)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="disp-txs">0</div>
                <div class="stat-label">Lock Transactions</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="disp-votes">0.00</div>
                <div class="stat-label">Voting Power (veBTC)</div>
            </div>
             <div class="stat-box">
                <div class="stat-value" id="disp-voters">0</div>
                <div class="stat-label">Vote Events</div>
            </div>
        </div>

        <!-- Main Chart -->
        <div class="card" style="margin-bottom: 20px;">
            <div id="mainChart" style="height: 500px;"></div>
        </div>
        
        <!-- Count Chart -->
         <div class="card" style="margin-bottom: 30px;">
            <div id="countChart" style="height: 300px;"></div>
        </div>

        <!-- Distribution & Config -->
        <div class="grid-row">
            <!-- Pie Chart Vol -->
            <div class="col card">
                <h3>Volume Distribution</h3>
                <div id="pieChart" style="height: 300px;"></div>
            </div>
            <!-- Pie Chart Count -->
             <div class="col card">
                <h3>Count Distribution</h3>
                <div id="pieChartCount" style="height: 300px;"></div>
            </div>
            
            <!-- Interact Table -->
            <div class="col card">
                <h3>Summary</h3>
                <table id="distTable">
                    <thead>
                        <tr>
                            <th>Range</th>
                            <th>Txs</th>
                            <th>Vol (BTC)</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
        
        <!-- Tabbed Data View -->
         <div class="card">
            <div class="tabs">
                <div class="tab active" onclick="switchTab('locks')">Lock Data (<span id="tcount-locks">0</span>)</div>
                <div class="tab" onclick="switchTab('votes')">Vote Data (<span id="tcount-votes">0</span>)</div>
            </div>
            
            <div id="tab-locks" class="tab-content active">
                 <div style="margin-bottom: 10px;">
                    <button class="btn" onclick="downloadCSV('locks')">Download CSV</button>
                 </div>
                 <div style="overflow-x: auto;">
                    <table id="lockTable">
                        <thead><tr><th>Time</th><th>Sender</th><th>Amount (BTC)</th><th>Type</th></tr></thead>
                        <tbody></tbody>
                    </table>
                 </div>
            </div>
             <div id="tab-votes" class="tab-content">
                 <div style="margin-bottom: 10px;">
                    <button class="btn" onclick="downloadCSV('votes')">Download CSV</button>
                 </div>
                 <div style="overflow-x: auto;">
                    <table id="voteTable">
                        <thead><tr><th>Time</th><th>Voter</th><th>Weight (veBTC)</th><th>Global Total</th></tr></thead>
                        <tbody></tbody>
                    </table>
                 </div>
            </div>
        </div>
    </div>

    <script>
        // --- Raw Data ---
        const rawLocks = {json.dumps(locks, default=json_serial)};
        const rawVotes = {json.dumps(votes, default=json_serial)};

        // --- Config ---
        const colors = {{
            "< 0.001": "#e3f2fd", 
            "0.001 - 0.01": "#90caf9", 
            "0.01 - 0.1": "#42a5f5", 
            "0.1 - 1": "#1e88e5", 
            "1 and above": "#1565c0"
        }};
        
        const catOrder = ["< 0.001", "0.001 - 0.01", "0.01 - 0.1", "0.1 - 1", "1 and above"];

        // --- State ---
        let hiddenCategories = new Set(); // For Pie Toggle

        // --- Init ---
        window.onload = function() {{
            try {{
                // Set default dates
                const defaultStart = "2025-12-18";
                
                // Find max date in data
                let maxDate = "2026-01-01";
                
                // Sort raw data using UTC helper
                rawLocks.sort((a, b) => parseDateUTC(a.date) - parseDateUTC(b.date));
                rawVotes.sort((a, b) => parseDateUTC(a.date) - parseDateUTC(b.date));
                
                if (rawLocks.length > 0) maxDate = rawLocks[rawLocks.length-1].date;
                
                document.getElementById("startDate").value = defaultStart;
                document.getElementById("endDate").value = maxDate;
                
                console.log("Init complete. rawLocks:", rawLocks.length, "rawVotes:", rawVotes.length);
                
                updateDashboard();
                
                // Resize Handler
                window.onresize = function() {{
                    Plotly.Plots.resize('mainChart');
                    Plotly.Plots.resize('countChart');
                    Plotly.Plots.resize('pieChart');
                    Plotly.Plots.resize('pieChartCount');
                }};
            }} catch (e) {{
                console.error("Dashboard Init Error:", e);
                alert("Error initializing dashboard: " + e.message);
            }}
        }};

        function resetDates() {{
             document.getElementById("startDate").value = "2025-12-18";
             if (rawLocks.length > 0) {{
                document.getElementById("endDate").value = rawLocks[rawLocks.length-1].date;
             }}
             updateDashboard();
        }}
        
        // --- CSV Download ---
        function downloadCSV(type) {{
             const startStr = document.getElementById("startDate").value;
             const endStr = document.getElementById("endDate").value;
             const startTs = new Date(startStr).getTime();
             const endTs = new Date(endStr).getTime() + 86400000;
             
             let data, filename, headers, rowMapper;
             
             if(type === 'locks') {{
                 data = rawLocks.filter(l => {{ const d = parseDateUTC(l.date); return d >= startTs && d < endTs; }});
                 filename = `vebtc_locks_${{startStr}}_${{endStr}}.csv`;
                 headers = ["Date", "Timestamp", "Sender", "Amount", "Category"];
                 rowMapper = l => [l.date, l.ts, l.sender, l.amount, l.cat];
             }} else {{
                 data = rawVotes.filter(l => {{ const d = parseDateUTC(l.date); return d >= startTs && d < endTs; }});
                 filename = `vebtc_votes_${{startStr}}_${{endStr}}.csv`;
                 headers = ["Date", "Timestamp", "Voter", "VoteWeight", "TotalWeight"];
                 rowMapper = v => [v.date, v.ts, v.voter, v.voting_power, v.total_weight];
             }}
             
             let csvContent = "data:text/csv;charset=utf-8," + headers.join(",") + "\\n";
             data.forEach(item => {{
                 const row = rowMapper(item).map(val => `"${{val}}"`).join(",");
                 csvContent += row + "\\n";
             }});
             
             const encodedUri = encodeURI(csvContent);
             const link = document.createElement("a");
             link.setAttribute("href", encodedUri);
             link.setAttribute("download", filename);
             document.body.appendChild(link);
             link.click();
             document.body.removeChild(link);
        }}
        
        function switchTab(tab) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            if (tab === 'locks') {{
                document.querySelectorAll('.tab')[0].classList.add('active');
                document.getElementById('tab-locks').classList.add('active');
            }} else {{
                document.querySelectorAll('.tab')[1].classList.add('active');
                document.getElementById('tab-votes').classList.add('active');
            }}
        }}

        // Helper to parse date as UTC timestamp to avoid timezone issues
        function parseDateUTC(str) {{
            if (!str) return 0;
            const parts = str.split('-');
            // year, monthIndex (0-11), day
            return Date.UTC(parseInt(parts[0]), parseInt(parts[1]) - 1, parseInt(parts[2]));
        }}

        // --- Core Logic ---
        function updateDashboard() {{
            const startStr = document.getElementById("startDate").value;
            const endStr = document.getElementById("endDate").value;
            
            console.log("Updating dashboard for range:", startStr, "to", endStr);
            
            const startTs = parseDateUTC(startStr);
            const endTs = parseDateUTC(endStr) + 86400000; // End of day

            // Filter
            const filteredLocks = rawLocks.filter(l => {{
                const d = parseDateUTC(l.date);
                return d >= startTs && d < endTs;
            }});
            const filteredVotes = rawVotes.filter(l => {{
                const d = parseDateUTC(l.date);
                return d >= startTs && d < endTs;
            }});
            
            console.log("Filtered Locks:", filteredLocks.length, "Filtered Votes:", filteredVotes.length);

            // Calc Stats
            const totalLocked = filteredLocks.reduce((acc, curr) => acc + curr.amount, 0);
            const totalVoteWeight = filteredVotes.reduce((acc, curr) => acc + curr.voting_power, 0);
            
            // DOM Updates
            document.getElementById("disp-locked").innerText = totalLocked.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
            document.getElementById("disp-txs").innerText = filteredLocks.length;
            document.getElementById("disp-votes").innerText = totalVoteWeight.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
            document.getElementById("disp-voters").innerText = filteredVotes.length;
            
            document.getElementById("tcount-locks").innerText = filteredLocks.length;
            document.getElementById("tcount-votes").innerText = filteredVotes.length;

            renderMainChart(filteredLocks, filteredVotes);
            renderCountChart(filteredLocks, filteredVotes);
            renderDistribution(filteredLocks);
            renderTables(filteredLocks, filteredVotes);
        }}

        function renderMainChart(locks, votes) {{
            // Group by date
            const dataMap = {{}};
            
            // Build map of all dates in range
            locks.forEach(l => {{
                if (!dataMap[l.date]) dataMap[l.date] = {{locked: 0, votes: 0, tx_lock: 0, tx_vote: 0}};
                dataMap[l.date].locked += l.amount;
                dataMap[l.date].tx_lock += 1;
            }});
            
            votes.forEach(v => {{
                if (!dataMap[v.date]) dataMap[v.date] = {{locked: 0, votes: 0, tx_lock: 0, tx_vote: 0}};
                dataMap[v.date].votes += v.voting_power;
                dataMap[v.date].tx_vote += 1;
            }});

            const dates = Object.keys(dataMap).sort();
            const lockVals = [];
            const voteVals = [];
            
            // Calc cumulatives (Local to filter window? Or Global?)
            // User likely wants trend of CURRENT data.
            // If we filter, cumulative should start from 0 at start? Or true cumulative?
            // "True Cumulative" requires processing ALL data before filter.
            // But let's simplify: Cumulative of the displayed data.
            const cumLockVals = [];
            const cumVoteVals = [];
            let rLock = 0;
            let rVote = 0;
            
            let totalLock = 0;
            let totalVote = 0;
            
            dates.forEach(d => {{
                const l = dataMap[d].locked;
                const v = dataMap[d].votes;
                rLock += l;
                rVote += v;
                totalLock += l;
                totalVote += v;
                
                lockVals.push(l);
                voteVals.push(v);
                cumLockVals.push(rLock);
                cumVoteVals.push(rVote);
            }});

            const trace1 = {{
                x: dates, y: lockVals, name: 'Daily New Locks (BTC)', type: 'bar', 
                marker: {{color: '#1565c0'}}, offsetgroup: 1,
                text: lockVals.map(v => v > 0 ? v.toFixed(2) : ""), textposition: 'auto'
            }};
            
            const trace2 = {{
                x: dates, y: voteVals, name: 'Daily Votes Cast (veBTC)', type: 'bar', 
                marker: {{color: '#28a745', opacity: 0.7}}, offsetgroup: 2,
                text: voteVals.map(v => v > 0 ? v.toFixed(2) : ""), textposition: 'auto'
            }};
            
            const trace3 = {{
                x: dates, y: cumLockVals, name: 'Cumulative Locked', type: 'scatter', mode: 'lines',
                line: {{color: '#ff9800', width: 3}}, yaxis: 'y2'
            }};
            
             const trace4 = {{
                x: dates, y: cumVoteVals, name: 'Cumulative Votes', type: 'scatter', mode: 'lines',
                line: {{color: '#9400D3', width: 3, dash: 'dot'}}, yaxis: 'y2'
            }};
            
            const layout = {{
                title: {{
                    text: 'BTC Locked & Voting Power',
                    y: 0.98,
                    yanchor: 'top'
                }},
                barmode: 'group',
                margin: {{t: 120, l: 50, r: 50, b: 50}},
                legend: {{orientation: 'h', y: 1.3}},
                hovermode: 'x unified',
                autosize: true,
                xaxis: {{
                    type: 'date',
                    tickformat: '%b %d',
                    automargin: true,
                    showline: true,
                    linewidth: 1,
                    linecolor: '#888',
                    ticks: 'outside'
                }},
                yaxis: {{
                    title: 'Daily Amount',
                    rangemode: 'tozero',
                    zeroline: false,
                    showline: true,
                    linewidth: 1,
                    linecolor: '#ddd'
                }},
                yaxis2: {{
                    title: 'Cumulative Amount',
                    overlaying: 'y',
                    side: 'right',
                    rangemode: 'tozero',
                    zeroline: false,
                    showgrid: false
                }}
            }};
            
            Plotly.newPlot('mainChart', [trace1, trace2, trace3, trace4], layout, {{responsive: true}});
        }}
        
        function renderCountChart(locks, votes) {{
            const dataMap = {{}};
             locks.forEach(l => {{
                if (!dataMap[l.date]) dataMap[l.date] = {{locked: 0, votes: 0}};
                dataMap[l.date].locked += 1;
            }});
            votes.forEach(v => {{
                if (!dataMap[v.date]) dataMap[v.date] = {{locked: 0, votes: 0}};
                dataMap[v.date].votes += 1;
            }});
            
            const dates = Object.keys(dataMap).sort();
            const lockCounts = dates.map(d => dataMap[d].locked);
            const voteCounts = dates.map(d => dataMap[d].votes);
            
            const totalLocks = lockCounts.reduce((a, b) => a + b, 0);
            const totalVotes = voteCounts.reduce((a, b) => a + b, 0);
            
            const trace1 = {{
                x: dates, y: lockCounts, name: 'Lock Txs', type: 'bar', marker: {{color: '#90caf9'}},
                text: lockCounts.map(v => v > 0 ? v : ""), textposition: 'auto'
            }};
            const trace2 = {{
                x: dates, y: voteCounts, name: 'Vote Txs', type: 'bar', marker: {{color: '#a5d6a7'}},
                text: voteCounts.map(v => v > 0 ? v : ""), textposition: 'auto'
            }};
            
             const layout = {{
                title: {{
                    text: 'Daily Transaction Counts',
                    y: 0.98,
                    yanchor: 'top'
                }},
                barmode: 'group',
                margin: {{t: 120, l: 50, r: 50, b: 50}},
                legend: {{orientation: 'h', y: 1.3}},
                hovermode: 'x unified',
                autosize: true,
                xaxis: {{ 
                    type: 'date',
                    tickformat: '%b %d',
                    automargin: true,
                    showline: true,
                    linewidth: 1,
                    linecolor: '#888',
                    ticks: 'outside'
                }},
                yaxis: {{
                    title: 'Transaction Count',
                    rangemode: 'tozero',
                    zeroline: false,
                    showline: true,
                    linewidth: 1,
                    linecolor: '#ddd'
                }}
            }};
            Plotly.newPlot('countChart', [trace1, trace2], layout, {{responsive: true}});
        }}

        function renderDistribution(locks) {{
            // bucket stats
            const buckets = {{}};
            catOrder.forEach(c => buckets[c] = {{count:0, vol:0}});
            
            locks.forEach(l => {{
                if (buckets[l.cat]) {{
                    buckets[l.cat].count++;
                    buckets[l.cat].vol += l.amount;
                }}
            }});
            
            // Update Table
            const tbody = document.querySelector("#distTable tbody");
            tbody.innerHTML = "";
            
            const pieLabels = [];
            const pieVolValues = [];
            const pieCountValues = [];
            const pieColors = [];
            
            catOrder.forEach(cat => {{
                const b = buckets[cat];
                const color = colors[cat];
                const isHidden = hiddenCategories.has(cat);
                
                // Table Row
                const tr = document.createElement("tr");
                tr.style.opacity = isHidden ? 0.5 : 1;
                tr.onclick = () => toggleCategory(cat);
                
                tr.innerHTML = `
                    <td><span class="legend-color" style="background:${{color}}"></span>${{cat}}</td>
                    <td class="mono">${{b.count}}</td>
                    <td class="mono">${{b.vol.toFixed(2)}}</td>
                `;
                tbody.appendChild(tr);
                
                // Pie Data
                if (!isHidden) {{ // Include 0 counts? Usually no.
                     if (b.count > 0 || b.vol > 0) {{
                        pieLabels.push(cat);
                        pieVolValues.push(b.vol);
                        pieCountValues.push(b.count);
                        pieColors.push(colors[cat]);
                     }}
                }}
            }});
            
            // Common Pie Layout
            const layout = {{
                margin: {{t: 10, l: 0, r: 0, b: 10}},
                showlegend: false,
                autosize: true
            }};
            
            // Render Pie Vol
            const traceVol = {{
                labels: pieLabels, values: pieVolValues, type: 'pie',
                marker: {{colors: pieColors}},
                textinfo: 'percent', hoverinfo: 'label+value+percent', name: 'Volume'
            }};
            Plotly.newPlot('pieChart', [traceVol], layout, {{responsive: true}});
            
            // Render Pie Count
            const traceCount = {{
                labels: pieLabels, values: pieCountValues, type: 'pie',
                marker: {{colors: pieColors}},
                textinfo: 'percent', hoverinfo: 'label+value+percent', name: 'Count'
            }};
            Plotly.newPlot('pieChartCount', [traceCount], layout, {{responsive: true}});
        }}
        
        function toggleCategory(cat) {{
            if (hiddenCategories.has(cat)) hiddenCategories.delete(cat);
            else hiddenCategories.add(cat);
            updateDashboard();
        }}

        function renderTables(locks, votes) {{
            // Lock Table (All)
            const lockBody = document.querySelector("#lockTable tbody");
            lockBody.innerHTML = "";
            
            // sort desc
            const sortedLocks = [...locks].sort((a,b) => new Date(b.date) - new Date(a.date));
            
            sortedLocks.forEach(l => {{
                const tr = document.createElement("tr");
                tr.innerHTML = `<td>${{l.date}}</td><td class="mono">${{l.sender}}</td><td class="mono">${{l.amount.toFixed(4)}}</td><td>${{l.cat}}</td>`;
                lockBody.appendChild(tr);
            }});
            
            // Vote Table
            const voteBody = document.querySelector("#voteTable tbody");
            voteBody.innerHTML = "";
            const sortedVotes = [...votes].sort((a,b) => new Date(b.date) - new Date(a.date));
            
             sortedVotes.forEach(v => {{
                const tr = document.createElement("tr");
                tr.innerHTML = `<td>${{v.date}}</td><td class="mono">${{v.voter}}</td><td class="mono">${{v.voting_power.toFixed(4)}}</td><td class="mono">${{v.total_weight.toFixed(2)}}</td>`;
                voteBody.appendChild(tr);
            }});
        }}

    </script>
</body>
</html>
    """
    
    with open("index.html", "w") as f:
        f.write(html_content)
        print("Dashboard saved as index.html")

if __name__ == "__main__":
    # 1. Load old
    existing_locks, existing_votes = load_data()
    
    # 2. Fetch new
    new_locks = fetch_incremental(LOCK_URL, {"filter": "to", "token": LOCK_TOKEN}, existing_locks, "locks")
    new_votes = fetch_incremental(VOTE_URL, {}, existing_votes, "votes")
    
    # 3. Balance
    current_balance = fetch_current_balance()

    # 4. Merge (New + Old)
    all_locks = new_locks + existing_locks
    all_votes = new_votes + existing_votes
    
    # 5. Save & Gen
    if new_locks or new_votes:
        save_data(all_locks, all_votes)
        
    df_main, dist_df, raw_locks_df, raw_votes_df = parse_data(all_locks, all_votes)
    
    # Convert DFs to lists for JSON serialization
    # raw_locks_df has cols: date, ts, amount, type, sender, cat, order
    locks_list = raw_locks_df.to_dict('records')
    votes_list = raw_votes_df.to_dict('records')
    
    generate_dashboard(locks_list, votes_list, current_balance)