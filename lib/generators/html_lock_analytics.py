"""HTML generator for lock analytics section."""
from typing import List, Dict, Any
import json


def generate_lock_analytics_section(wallet_profiles: List[Any],
                                    statistics: Dict[str, Any]) -> str:
    """Generate HTML for lock analytics section.

    Args:
        wallet_profiles: List of WalletLockProfile objects
        statistics: Aggregate statistics

    Returns:
        HTML string for lock analytics section
    """
    # Generate summary cards
    summary_html = generate_summary_cards(statistics)

    # Generate wallet table
    table_html = generate_wallet_table(wallet_profiles)

    html = f"""
    <div class="lock-analytics-section">
        <h2>Lock Analytics</h2>
        <p class="section-description">
            Wallet-level lock statistics showing lock duration patterns. Max locks are 30-day duration locks.
            Data includes locks created or extended since December 5, 2025.
        </p>

        {summary_html}

        <div class="analytics-controls">
            <div class="filter-group">
                <label>Sort By:</label>
                <select id="lock-sort-select">
                    <option value="max_count">Most Max Locks</option>
                    <option value="max_pct">Highest Max %</option>
                    <option value="total_btc">Most BTC Locked</option>
                    <option value="lock_count">Most Locks</option>
                </select>
            </div>
            <div class="filter-group">
                <label>
                    <input type="checkbox" id="max-only-filter">
                    Show Only Max Lock Users
                </label>
            </div>
        </div>

        {table_html}
    </div>
    """

    return html


def generate_summary_cards(stats: Dict[str, Any]) -> str:
    """Generate summary statistics cards."""
    total_wallets = stats.get('total_wallets', 0)
    total_locks = stats.get('total_locks', 0)
    total_max_locks = stats.get('total_max_locks', 0)
    max_lock_rate = stats.get('max_lock_rate', 0)

    html = f"""
    <div class="lock-summary">
        <div class="summary-card">
            <div class="card-icon">👥</div>
            <div class="card-content">
                <div class="card-label">Total Wallets</div>
                <div class="card-value">{total_wallets:,}</div>
            </div>
        </div>
        <div class="summary-card">
            <div class="card-icon">🔒</div>
            <div class="card-content">
                <div class="card-label">Total Locks</div>
                <div class="card-value">{total_locks:,}</div>
            </div>
        </div>
        <div class="summary-card">
            <div class="card-icon">⏱️</div>
            <div class="card-content">
                <div class="card-label">Max Duration Locks (30d)</div>
                <div class="card-value">{total_max_locks:,}</div>
            </div>
        </div>
        <div class="summary-card">
            <div class="card-icon">📊</div>
            <div class="card-content">
                <div class="card-label">Max Lock Rate</div>
                <div class="card-value">{max_lock_rate:.1f}%</div>
            </div>
        </div>
    </div>
    """

    return html


def generate_wallet_table(profiles: List[Any]) -> str:
    """Generate wallet analytics table."""
    if not profiles:
        return '<p class="empty-state">No lock data available</p>'

    rows_html = ""

    for profile in profiles:
        address = profile.address
        display_addr = profile.display_address
        total_btc = profile.total_btc_locked
        lock_count = profile.lock_count
        max_lock_count = profile.max_lock_count
        max_lock_pct = profile.max_lock_percentage

        # Generate badge for high max-lock usage
        badge = ""
        if max_lock_pct >= 80:
            badge = '<span class="badge badge-high">High Max %</span>'
        elif max_lock_pct >= 50:
            badge = '<span class="badge badge-medium">Med Max %</span>'

        rows_html += f"""
        <tr data-address="{address}"
            data-max-count="{max_lock_count}"
            data-max-pct="{max_lock_pct}"
            data-total-btc="{total_btc}"
            data-lock-count="{lock_count}">
            <td class="address-cell">
                <code class="address-code" title="{address}">{display_addr}</code>
                {badge}
            </td>
            <td class="number-cell">{total_btc:.4f} BTC</td>
            <td class="count-cell">{lock_count}</td>
            <td class="count-cell highlight">{max_lock_count}</td>
            <td class="percent-cell">
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: {min(max_lock_pct, 100)}%"></div>
                    <span class="progress-text">{max_lock_pct:.1f}%</span>
                </div>
            </td>
        </tr>
        """

    html = f"""
    <div class="lock-table-wrapper">
        <table class="lock-table" id="lock-analytics-table">
            <thead>
                <tr>
                    <th>Address</th>
                    <th>Total BTC Locked</th>
                    <th>Total Locks</th>
                    <th>Max Duration Locks</th>
                    <th>Max Lock %</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """

    return html


def generate_lock_analytics_css() -> str:
    """Generate CSS for lock analytics section."""
    css = """
    /* Lock Analytics Styles */
    .lock-analytics-section {
        margin-top: 30px;
        margin-bottom: 30px;
    }

    .lock-analytics-section h2 {
        font-size: 28px;
        margin-bottom: 10px;
        color: #2c3e50;
    }

    .info-banner {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 12px 16px;
        margin: 15px 0;
        border-radius: 4px;
        font-size: 14px;
        color: #1565c0;
        line-height: 1.5;
    }

    .info-banner strong {
        color: #0d47a1;
    }

    .lock-summary {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }

    .analytics-controls {
        display: flex;
        gap: 30px;
        align-items: center;
        margin-bottom: 20px;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 8px;
        flex-wrap: wrap;
    }

    .filter-group {
        display: flex;
        gap: 10px;
        align-items: center;
    }

    .filter-group label {
        font-weight: 500;
        color: #2c3e50;
        font-size: 14px;
    }

    .filter-group select {
        padding: 8px 12px;
        border: 1px solid #e1e8ed;
        border-radius: 6px;
        font-size: 14px;
        background: white;
        cursor: pointer;
    }

    .filter-group input[type="checkbox"] {
        cursor: pointer;
        width: 16px;
        height: 16px;
    }

    .lock-table-wrapper {
        overflow-x: auto;
        border-radius: 8px;
        border: 1px solid #e1e8ed;
        background: white;
    }

    .lock-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }

    .lock-table thead {
        background: #f8f9fa;
    }

    .lock-table th {
        padding: 12px 10px;
        text-align: left;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #e1e8ed;
    }

    .lock-table td {
        padding: 12px 10px;
        border-bottom: 1px solid #f0f0f0;
    }

    .lock-table tbody tr:hover {
        background: #f8f9fa;
    }

    .address-cell {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .address-code {
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        font-size: 12px;
        padding: 4px 8px;
        background: #f8f9fa;
        border-radius: 4px;
    }

    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        white-space: nowrap;
    }

    .badge-high {
        background: #d4edda;
        color: #155724;
    }

    .badge-medium {
        background: #fff3cd;
        color: #856404;
    }

    .number-cell {
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        font-weight: 500;
    }

    .count-cell {
        text-align: center;
        font-weight: 500;
    }

    .percent-cell {
        padding: 8px 10px;
    }

    .progress-bar-container {
        position: relative;
        width: 120px;
        height: 24px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.3s;
    }

    .progress-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 12px;
        font-weight: 600;
        color: #2c3e50;
    }

    .highlight {
        font-weight: 600;
        color: #667eea;
    }

    @media (max-width: 768px) {
        .lock-table {
            font-size: 12px;
        }

        .lock-table th,
        .lock-table td {
            padding: 8px 6px;
        }

        .progress-bar-container {
            width: 80px;
        }

        .analytics-controls {
            flex-direction: column;
            align-items: flex-start;
        }
    }
    """

    return css


def generate_lock_analytics_js() -> str:
    """Generate JavaScript for lock analytics interactions."""
    js = """
    // Lock Analytics JavaScript

    document.addEventListener('DOMContentLoaded', function() {
        const sortSelect = document.getElementById('lock-sort-select');
        const maxOnlyFilter = document.getElementById('max-only-filter');
        const table = document.getElementById('lock-analytics-table');

        if (!sortSelect || !table) return;

        function sortTable(criteria) {
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            rows.sort((a, b) => {
                let aVal, bVal;

                switch(criteria) {
                    case 'max_count':
                        aVal = parseFloat(a.dataset.maxCount);
                        bVal = parseFloat(b.dataset.maxCount);
                        break;
                    case 'max_pct':
                        aVal = parseFloat(a.dataset.maxPct);
                        bVal = parseFloat(b.dataset.maxPct);
                        break;
                    case 'total_btc':
                        aVal = parseFloat(a.dataset.totalBtc);
                        bVal = parseFloat(b.dataset.totalBtc);
                        break;
                    case 'lock_count':
                        aVal = parseFloat(a.dataset.lockCount);
                        bVal = parseFloat(b.dataset.lockCount);
                        break;
                }

                return bVal - aVal;  // Descending
            });

            tbody.innerHTML = '';
            rows.forEach(row => tbody.appendChild(row));
        }

        function filterTable() {
            const tbody = table.querySelector('tbody');
            const rows = tbody.querySelectorAll('tr');
            const showMaxOnly = maxOnlyFilter.checked;

            rows.forEach(row => {
                if (showMaxOnly) {
                    const maxCount = parseFloat(row.dataset.maxCount);
                    row.style.display = maxCount > 0 ? '' : 'none';
                } else {
                    row.style.display = '';
                }
            });
        }

        sortSelect.addEventListener('change', (e) => sortTable(e.target.value));
        maxOnlyFilter.addEventListener('change', filterTable);

        // Initial sort
        sortTable('max_count');
    });
    """

    return js
