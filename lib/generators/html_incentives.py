"""HTML generator for incentives dashboard section."""
from typing import List, Dict, Any
from lib.analytics.incentives import format_apr, format_usd


def generate_incentives_section(pool_incentives: List[Dict[str, Any]],
                                previous_pool_incentives: List[Dict[str, Any]] = None,
                                current_epoch_number: int = None) -> str:
    """Generate HTML for incentives section.

    Args:
        pool_incentives: List of PoolIncentives objects for current epoch (as dicts)
        previous_pool_incentives: List of PoolIncentives objects for previous epoch (as dicts)
        current_epoch_number: Current epoch number for labeling

    Returns:
        HTML string for incentives section
    """
    if not pool_incentives:
        return """
        <div class="incentives-section">
            <h2>Pool Incentives</h2>
            <p class="empty-state">No incentive data available</p>
        </div>
        """

    # Create lookup dict for previous epoch data by pool address
    previous_pools_map = {}
    if previous_pool_incentives:
        for pool in previous_pool_incentives:
            previous_pools_map[pool.get("pool_address")] = pool

    # Calculate totals
    total_bribes_usd = sum(p.get("bribes_usd", 0) for p in pool_incentives)
    total_fees_usd = sum(p.get("fees_usd", 0) for p in pool_incentives)
    total_votes = sum(p.get("current_votes", 0) for p in pool_incentives)

    # Generate summary cards
    summary_html = f"""
    <div class="incentives-summary">
        <div class="summary-card" title="Bribes are incentives deposited by third parties into pool reward contracts to attract voting power. Voters who allocate their veBTC to a pool can claim a proportional share of the bribes.">
            <div class="card-icon">💰</div>
            <div class="card-content">
                <div class="card-label">Total Bribes ⓘ</div>
                <div class="card-value">{format_usd(total_bribes_usd)}</div>
            </div>
        </div>
        <div class="summary-card">
            <div class="card-icon">💵</div>
            <div class="card-content">
                <div class="card-label">Total Fees</div>
                <div class="card-value">{format_usd(total_fees_usd)}</div>
            </div>
        </div>
        <div class="summary-card">
            <div class="card-icon">🗳️</div>
            <div class="card-content">
                <div class="card-label">Total Votes</div>
                <div class="card-value">{total_votes:.2f} veBTC</div>
            </div>
        </div>
        <div class="summary-card" title="USD value of incentives per veBTC voting power. Calculation: ({format_usd(total_bribes_usd)} bribes + {format_usd(total_fees_usd)} fees) / {total_votes:.2f} votes = ${(total_bribes_usd + total_fees_usd) / total_votes if total_votes > 0 else 0:.4f} per vote">
            <div class="card-icon">📊</div>
            <div class="card-content">
                <div class="card-label">$/Vote ⓘ</div>
                <div class="card-value">${(total_bribes_usd + total_fees_usd) / total_votes if total_votes > 0 else 0:.2f}</div>
            </div>
        </div>
    </div>
    """

    # Sort pools by APR (descending)
    sorted_pools = sorted(pool_incentives, key=lambda p: p.get("apr_total", 0), reverse=True)

    # Generate pool cards
    pool_cards_html = ""
    for pool in sorted_pools:
        pool_name = pool.get("pool_name", "Unknown Pool")
        pool_address = pool.get("pool_address", "")
        current_votes = pool.get("current_votes", 0)
        bribes = pool.get("bribes", {})
        fees = pool.get("fees", {})
        bribes_usd = pool.get("bribes_usd", 0)
        fees_usd = pool.get("fees_usd", 0)
        apr_total = pool.get("apr_total", 0)
        apr_bribes = pool.get("apr_bribes", 0)
        apr_fees = pool.get("apr_fees", 0)
        usd_per_vote = pool.get("usd_per_vote", 0)

        # Get previous epoch data for this pool
        prev_pool = previous_pools_map.get(pool_address)
        prev_apr_total = prev_pool.get("apr_total", 0) if prev_pool else None

        # Format bribes token list (filter out zero amounts and format appropriately)
        bribes_html = ""
        if bribes:
            for token, amount in bribes.items():
                if amount > 0:  # Only show non-zero bribes
                    # Use more decimals for small amounts
                    if amount >= 1:
                        formatted_amount = f"{amount:.2f}"
                    elif amount >= 0.01:
                        formatted_amount = f"{amount:.4f}"
                    else:
                        formatted_amount = f"{amount:.6f}"
                    bribes_html += f'<span class="token-badge">{token}: {formatted_amount}</span>'

        if not bribes_html:
            bribes_html = '<span class="empty-badge">No bribes</span>'

        # Determine APR color class
        apr_class = ""
        if apr_total >= 50:
            apr_class = "high-apr"
        elif apr_total >= 20:
            apr_class = "medium-apr"
        else:
            apr_class = "low-apr"

        # Tooltip explaining APR calculation
        tooltip_text = f"Current Epoch APR Calculation: ({format_usd(bribes_usd)} bribes / {current_votes:.2f} votes) × 52 weeks / BTC price × 100"

        # Format $/Vote with tooltip
        usd_per_vote_display = f'<span class="usd-per-vote" title="USD value of incentives per veBTC voting power. Calculation: ({format_usd(bribes_usd)} bribes + {format_usd(fees_usd)} fees) / {current_votes:.2f} votes = ${usd_per_vote:.4f} per vote">${usd_per_vote:.4f}/vote ⓘ</span>'

        # Build APR display with both current and previous
        apr_display_html = f'<div class="apr-badge {apr_class}" title="{tooltip_text}">{format_apr(apr_total)} APR ⓘ</div>'

        # Add previous epoch APR if available
        if prev_apr_total is not None:
            prev_apr_class = ""
            if prev_apr_total >= 50:
                prev_apr_class = "high-apr"
            elif prev_apr_total >= 20:
                prev_apr_class = "medium-apr"
            else:
                prev_apr_class = "low-apr"

            # Calculate change
            apr_change = apr_total - prev_apr_total
            change_indicator = ""
            if apr_change > 0.1:
                change_indicator = f' <span class="apr-change positive">▲ {format_apr(abs(apr_change))}</span>'
            elif apr_change < -0.1:
                change_indicator = f' <span class="apr-change negative">▼ {format_apr(abs(apr_change))}</span>'
            else:
                change_indicator = ' <span class="apr-change neutral">—</span>'

            apr_display_html = f'''
                <div class="apr-container">
                    <div class="apr-row">
                        <span class="epoch-label">Current:</span>
                        <div class="apr-badge {apr_class}" title="{tooltip_text}">{format_apr(apr_total)} APR ⓘ</div>
                        {change_indicator}
                    </div>
                    <div class="apr-row previous">
                        <span class="epoch-label">Previous:</span>
                        <div class="apr-badge {prev_apr_class}">{format_apr(prev_apr_total)} APR</div>
                    </div>
                </div>
            '''

        pool_cards_html += f"""
        <div class="pool-item">
            <div class="pool-header">
                <div class="pool-title">
                    <span class="pool-name">{pool_name}</span>
                    <button class="copy-address-btn" onclick="copyPoolAddress('{pool_address}')" title="Copy pool address">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                    </button>
                </div>
            </div>
            <div class="pool-body">
                <div class="pool-apr-section">
                    {apr_display_html}
                </div>
                <div class="pool-rewards-section">
                    <div class="pool-bribes">
                        <span class="bribes-label">Bribes:</span>
                        <div class="bribes-tokens">
                            {bribes_html}
                        </div>
                    </div>
                    {usd_per_vote_display}
                </div>
            </div>
        </div>
        """

    # Generate epoch labels
    epoch_label = f"Epoch {current_epoch_number}" if current_epoch_number else "Current Epoch"
    previous_epoch_label = f"Epoch {current_epoch_number - 1}" if current_epoch_number else "Previous Epoch"

    # Description text
    description = f"Bribes, fees, and projected APR for each pool. Showing {epoch_label}"
    if previous_pool_incentives:
        description += f" (current) and {previous_epoch_label} (previous) for comparison."
    else:
        description += "."

    # Combine into full section
    html = f"""
    <div class="incentives-section">
        <h2>Pool Incentives</h2>
        <p class="section-description">{description}</p>
        {summary_html}
        <div class="pool-grid">
            {pool_cards_html}
        </div>
    </div>
    """

    return html


def generate_incentives_css() -> str:
    """Generate CSS for incentives section.

    Returns:
        CSS string
    """
    css = """
    /* Incentives Section Styles */
    .incentives-section {
        margin-top: 30px;
        margin-bottom: 30px;
    }

    .incentives-section h2 {
        font-size: 28px;
        margin-bottom: 10px;
        color: #2c3e50;
    }

    .section-description {
        color: #7f8c8d;
        margin-bottom: 25px;
        font-size: 14px;
    }

    .incentives-summary {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }

    .summary-card {
        background: #FF004D;
        color: white;
        padding: 20px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }

    .summary-card[title] {
        cursor: help;
    }

    .summary-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    .card-icon {
        font-size: 32px;
        line-height: 1;
    }

    .card-content {
        flex: 1;
    }

    .card-label {
        font-size: 12px;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }

    .card-value {
        font-size: 24px;
        font-weight: 700;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
    }

    .pool-grid {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .pool-item {
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 8px;
        padding: 16px 20px;
        display: flex;
        flex-direction: column;
        gap: 12px;
        transition: box-shadow 0.2s, border-color 0.2s;
    }

    .pool-item:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-color: #FF004D;
    }

    .pool-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .pool-body {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 20px;
        align-items: center;
    }

    .pool-apr-section {
        display: flex;
        justify-content: center;
        min-width: 200px;
    }

    .pool-rewards-section {
        display: flex;
        flex-direction: column;
        gap: 8px;
        align-items: flex-end;
        text-align: right;
    }

    .pool-title {
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .pool-name {
        font-size: 16px;
        color: #2c3e50;
        font-weight: 600;
    }

    .copy-address-btn {
        background: none;
        border: none;
        padding: 4px;
        cursor: pointer;
        color: #7f8c8d;
        display: flex;
        align-items: center;
        transition: color 0.2s;
    }

    .copy-address-btn:hover {
        color: #FF004D;
    }

    .copy-address-btn:active {
        transform: scale(0.95);
    }

    .usd-per-vote {
        font-size: 14px;
        color: #FF004D;
        font-weight: 600;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        cursor: help;
    }

    .apr-badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        cursor: help;
    }

    .apr-badge.high-apr {
        background: rgba(76, 175, 80, 0.2);
        color: #2e7d32;
    }

    .apr-badge.medium-apr {
        background: rgba(255, 152, 0, 0.2);
        color: #e65100;
    }

    .apr-badge.low-apr {
        background: rgba(158, 158, 158, 0.2);
        color: #616161;
    }

    .apr-container {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .apr-row {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .apr-row.previous {
        opacity: 0.75;
    }

    .epoch-label {
        font-size: 11px;
        color: #7f8c8d;
        font-weight: 500;
        min-width: 60px;
    }

    .apr-change {
        font-size: 12px;
        font-weight: 600;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
    }

    .apr-change.positive {
        color: #2e7d32;
    }

    .apr-change.negative {
        color: #c62828;
    }

    .apr-change.neutral {
        color: #95a5a6;
    }

    .pool-bribes {
        display: flex;
        flex-direction: column;
        gap: 6px;
        align-items: flex-end;
    }

    .bribes-label {
        font-size: 12px;
        color: #7f8c8d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .bribes-tokens {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        justify-content: flex-end;
    }

    .token-badge {
        display: inline-block;
        padding: 3px 8px;
        background: #f8f9fa;
        border: 1px solid #e1e8ed;
        border-radius: 4px;
        font-size: 12px;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        color: #2c3e50;
    }

    .empty-badge {
        display: inline-block;
        padding: 3px 8px;
        color: #95a5a6;
        font-size: 12px;
        font-style: italic;
    }

    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #95a5a6;
        font-size: 16px;
    }

    @media (max-width: 768px) {
        .incentives-summary {
            grid-template-columns: repeat(2, 1fr);
        }

        .pool-body {
            grid-template-columns: 1fr;
            gap: 12px;
        }

        .pool-apr-section {
            justify-content: flex-start;
        }

        .pool-rewards-section {
            align-items: flex-start;
            text-align: left;
        }

        .bribes-tokens {
            justify-content: flex-start;
        }

        .pool-bribes {
            align-items: flex-start;
        }
    }
    """

    return css


def generate_incentives_js() -> str:
    """Generate JavaScript for incentives section.

    Returns:
        JavaScript string
    """
    js = """
    // Copy pool address to clipboard
    function copyPoolAddress(address) {
        navigator.clipboard.writeText(address).then(() => {
            // Show temporary feedback (could add a toast notification)
            console.log('Pool address copied:', address);

            // Optional: Add visual feedback
            const buttons = document.querySelectorAll('.copy-address-btn');
            buttons.forEach(btn => {
                if (btn.getAttribute('onclick').includes(address)) {
                    const originalColor = btn.style.color;
                    btn.style.color = '#FF004D';
                    setTimeout(() => {
                        btn.style.color = originalColor;
                    }, 300);
                }
            });
        }).catch(err => {
            console.error('Failed to copy address:', err);
            alert('Failed to copy address. Please copy manually: ' + address);
        });
    }
    """

    return js
