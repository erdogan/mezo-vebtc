"""HTML generator for past epochs dashboard section."""
from typing import Dict, Any


def generate_past_epochs_section(epochs_data: Dict[str, Any],
                                  current_epoch_number: int) -> str:
    """Generate HTML for past epochs section.

    Args:
        epochs_data: Dictionary of epoch data keyed by epoch number
        current_epoch_number: Current epoch number

    Returns:
        HTML string for past epochs section
    """
    if not epochs_data:
        return """
        <div class="past-epochs-section">
            <h2>Past Epochs</h2>
            <p class="empty-state">No historical epoch data available</p>
        </div>
        """

    # Sort epochs by number (descending)
    sorted_epochs = sorted(
        epochs_data.items(),
        key=lambda x: int(x[0]),
        reverse=True
    )

    # Generate epoch cards
    epochs_html = ""
    for epoch_key, epoch in sorted_epochs:
        epoch_num = epoch['epoch_number']

        # Skip current epoch (shown in banner)
        if epoch_num == current_epoch_number:
            continue

        votes = epoch.get('votes', {})
        incentives = epoch.get('incentives', {})

        total_voted = votes.get('total_voted', 0)
        unique_voters = votes.get('unique_voters', 0)
        vote_tx_count = votes.get('vote_tx_count', 0)

        total_bribes = incentives.get('total_bribes_usd', 0)
        total_fees = incentives.get('total_fees_usd', 0)
        avg_apr = incentives.get('average_apr', 0)
        pool_count = incentives.get('pool_count', 0)

        # Format dates
        start_date = epoch.get('start_date', 'Unknown')
        end_date = epoch.get('end_date', 'Unknown')

        # APR badge styling
        apr_class = ""
        if avg_apr >= 50:
            apr_class = "high-apr"
        elif avg_apr >= 20:
            apr_class = "medium-apr"
        else:
            apr_class = "low-apr"

        # Format values
        def format_usd(value):
            if value >= 1000:
                return f"${value:,.0f}"
            else:
                return f"${value:,.2f}"

        def format_apr(value):
            return f"{value:.1f}%"

        # Build epoch card
        epochs_html += f"""
        <div class="epoch-card">
            <div class="epoch-header">
                <h3 class="epoch-title">Epoch {epoch_num}</h3>
                <span class="epoch-dates">{start_date[:10]} - {end_date[:10]}</span>
            </div>

            <div class="epoch-stats">
                <div class="epoch-stat-row">
                    <div class="epoch-stat">
                        <div class="stat-icon">🗳️</div>
                        <div class="stat-content">
                            <div class="stat-label">Total Voted</div>
                            <div class="stat-value">{total_voted:.2f} veBTC</div>
                        </div>
                    </div>

                    <div class="epoch-stat">
                        <div class="stat-icon">👥</div>
                        <div class="stat-content">
                            <div class="stat-label">Unique Voters</div>
                            <div class="stat-value">{unique_voters}</div>
                        </div>
                    </div>

                    <div class="epoch-stat">
                        <div class="stat-icon">📊</div>
                        <div class="stat-content">
                            <div class="stat-label">Vote Transactions</div>
                            <div class="stat-value">{vote_tx_count}</div>
                        </div>
                    </div>
                </div>

                <div class="epoch-stat-row">
                    <div class="epoch-stat">
                        <div class="stat-icon">💰</div>
                        <div class="stat-content">
                            <div class="stat-label">Total Incentives</div>
                            <div class="stat-value">{format_usd(total_bribes + total_fees)}</div>
                            <div class="stat-detail">Bribes: {format_usd(total_bribes)} | Fees: {format_usd(total_fees)}</div>
                        </div>
                    </div>

                    <div class="epoch-stat">
                        <div class="stat-icon">📈</div>
                        <div class="stat-content">
                            <div class="stat-label">Average APR</div>
                            <div class="stat-value">
                                <span class="apr-badge {apr_class}">{format_apr(avg_apr)}</span>
                            </div>
                            <div class="stat-detail">{pool_count} pools</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """

    html = f"""
    <div class="past-epochs-section">
        <h2>Past Epochs</h2>
        <p class="section-description">Historical voting activity and incentives for the last 10 epochs</p>

        <div class="epochs-grid">
            {epochs_html}
        </div>
    </div>
    """

    return html


def generate_past_epochs_css() -> str:
    """Generate CSS for past epochs section.

    Returns:
        CSS string
    """
    css = """
    /* Past Epochs Section Styles */
    .past-epochs-section {
        margin-top: 30px;
        margin-bottom: 30px;
    }

    .past-epochs-section h2 {
        font-size: 28px;
        margin-bottom: 10px;
        color: #2c3e50;
    }

    .section-description {
        color: #7f8c8d;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .epochs-grid {
        display: flex;
        flex-direction: column;
        gap: 15px;
    }

    .epoch-card {
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s, border-color 0.2s;
    }

    .epoch-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-color: #FF004D;
    }

    .epoch-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        padding-bottom: 12px;
        border-bottom: 2px solid #f0f0f0;
    }

    .epoch-title {
        margin: 0;
        font-size: 20px;
        color: #2c3e50;
        font-weight: 700;
    }

    .epoch-dates {
        font-size: 13px;
        color: #7f8c8d;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
    }

    .epoch-stats {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .epoch-stat-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
    }

    .epoch-stat {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px;
        background: #f8f9fa;
        border-radius: 8px;
    }

    .stat-icon {
        font-size: 24px;
        line-height: 1;
    }

    .stat-content {
        flex: 1;
        min-width: 0;
    }

    .stat-label {
        font-size: 11px;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 3px;
    }

    .stat-value {
        font-size: 16px;
        font-weight: 600;
        color: #2c3e50;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
    }

    .stat-detail {
        font-size: 11px;
        color: #95a5a6;
        margin-top: 2px;
    }

    @media (max-width: 768px) {
        .epoch-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }

        .epoch-stat-row {
            grid-template-columns: 1fr;
        }
    }
    """

    return css
