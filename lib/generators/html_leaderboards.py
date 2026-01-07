"""HTML generator for leaderboards section."""
from typing import List, Dict, Any


def generate_leaderboards_section(top_lockers: List[Dict[str, Any]],
                                  top_voters: List[Dict[str, Any]]) -> str:
    """Generate HTML for leaderboards section.

    Args:
        top_lockers: List of top locker profiles (as dicts)
        top_voters: List of top voter profiles (as dicts)

    Returns:
        HTML string for leaderboards section
    """
    # Generate top lockers table
    lockers_html = generate_lockers_table(top_lockers)

    # Generate top voters table
    voters_html = generate_voters_table(top_voters)

    html = f"""
    <div class="leaderboards-section">
        <h2>Leaderboards</h2>
        <p class="section-description">Top participants by BTC locked and voting power</p>

        <div class="leaderboards-grid">
            <div class="leaderboard-column">
                <h3>🏆 Top Lockers</h3>
                <p class="leaderboard-subtitle">By total BTC locked</p>
                {lockers_html}
            </div>

            <div class="leaderboard-column">
                <h3>🗳️ Top Voters</h3>
                <p class="leaderboard-subtitle">By current voting power</p>
                {voters_html}
            </div>
        </div>
    </div>
    """

    return html


def generate_lockers_table(lockers: List[Dict[str, Any]]) -> str:
    """Generate HTML table for top lockers.

    Args:
        lockers: List of locker profiles

    Returns:
        HTML string
    """
    if not lockers:
        return '<p class="empty-state">No data available</p>'

    rows_html = ""
    for locker in lockers:
        rank = locker.get('lock_rank', '?')
        address = locker.get('address', 'Unknown')
        display_addr = f"{address[:6]}...{address[-4:]}"
        total_locked = locker.get('total_locked', 0)
        num_locks = locker.get('num_locks', 0)

        # Medal emoji for top 3
        rank_display = rank
        if rank == 1:
            rank_display = "🥇"
        elif rank == 2:
            rank_display = "🥈"
        elif rank == 3:
            rank_display = "🥉"

        rows_html += f"""
        <tr data-address="{address}">
            <td class="rank-cell">{rank_display}</td>
            <td class="address-cell">
                <code class="address-code" title="{address}">{display_addr}</code>
            </td>
            <td class="number-cell">{total_locked:.4f} BTC</td>
            <td class="count-cell">{num_locks} txs</td>
        </tr>
        """

    html = f"""
    <div class="leaderboard-table-wrapper">
        <table class="leaderboard-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Address</th>
                    <th>Total Locked</th>
                    <th>Transactions</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """

    return html


def generate_voters_table(voters: List[Dict[str, Any]]) -> str:
    """Generate HTML table for top voters.

    Args:
        voters: List of voter profiles

    Returns:
        HTML string
    """
    if not voters:
        return '<p class="empty-state">No data available</p>'

    rows_html = ""
    for voter in voters:
        rank = voter.get('vote_rank', '?')
        address = voter.get('address', 'Unknown')
        display_addr = f"{address[:6]}...{address[-4:]}"
        voting_power = voter.get('current_voting_power', 0)
        votes_cast = voter.get('total_votes_cast', 0)
        pools_count = len(voter.get('pools_voted', []))

        # Medal emoji for top 3
        rank_display = rank
        if rank == 1:
            rank_display = "🥇"
        elif rank == 2:
            rank_display = "🥈"
        elif rank == 3:
            rank_display = "🥉"

        rows_html += f"""
        <tr data-address="{address}">
            <td class="rank-cell">{rank_display}</td>
            <td class="address-cell">
                <code class="address-code" title="{address}">{display_addr}</code>
            </td>
            <td class="number-cell">{voting_power:.4f} veBTC</td>
            <td class="count-cell">{votes_cast} votes</td>
            <td class="count-cell">{pools_count} pools</td>
        </tr>
        """

    html = f"""
    <div class="leaderboard-table-wrapper">
        <table class="leaderboard-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Address</th>
                    <th>Voting Power</th>
                    <th>Votes Cast</th>
                    <th>Pools</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """

    return html


def generate_leaderboards_css() -> str:
    """Generate CSS for leaderboards section.

    Returns:
        CSS string
    """
    css = """
    /* Leaderboards Section Styles */
    .leaderboards-section {
        margin-top: 30px;
        margin-bottom: 30px;
    }

    .leaderboards-section h2 {
        font-size: 28px;
        margin-bottom: 10px;
        color: #2c3e50;
    }

    .leaderboards-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 30px;
        margin-top: 20px;
    }

    .leaderboard-column {
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .leaderboard-column h3 {
        margin: 0 0 5px 0;
        font-size: 20px;
        color: #2c3e50;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .leaderboard-subtitle {
        font-size: 13px;
        color: #7f8c8d;
        margin: 0 0 20px 0;
    }

    .leaderboard-table-wrapper {
        overflow-x: auto;
        border-radius: 8px;
        border: 1px solid #e1e8ed;
    }

    .leaderboard-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }

    .leaderboard-table thead {
        background: #f8f9fa;
    }

    .leaderboard-table th {
        padding: 12px 10px;
        text-align: left;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #e1e8ed;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .leaderboard-table td {
        padding: 12px 10px;
        border-bottom: 1px solid #f0f0f0;
    }

    .leaderboard-table tbody tr:last-child td {
        border-bottom: none;
    }

    .leaderboard-table tbody tr:hover {
        background: #f8f9fa;
        cursor: pointer;
    }

    .rank-cell {
        text-align: center;
        font-weight: 600;
        font-size: 16px;
        width: 60px;
    }

    .address-cell {
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
    }

    .address-code {
        font-size: 13px;
        color: #FF004D;
        background: #fff0f4;
        padding: 4px 8px;
        border-radius: 4px;
        cursor: help;
        transition: background 0.2s;
    }

    .address-code:hover {
        background: #ffe0e9;
    }

    .number-cell {
        text-align: right;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        font-weight: 600;
        color: #2c3e50;
    }

    .count-cell {
        text-align: center;
        color: #7f8c8d;
        font-size: 13px;
    }

    @media (max-width: 1024px) {
        .leaderboards-grid {
            grid-template-columns: 1fr;
        }

        .leaderboard-table {
            font-size: 12px;
        }

        .leaderboard-table th,
        .leaderboard-table td {
            padding: 8px 6px;
        }

        .address-code {
            font-size: 11px;
        }
    }

    @media (max-width: 768px) {
        .leaderboard-table-wrapper {
            overflow-x: scroll;
        }

        .leaderboard-table {
            min-width: 500px;
        }
    }
    """

    return css
