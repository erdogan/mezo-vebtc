"""HTML generator for fees dashboard section."""
from typing import Dict, Any, List
import json


def format_usd(value: float) -> str:
    """Format USD value for display."""
    if value >= 1000:
        return f"${value:,.0f}"
    else:
        return f"${value:,.2f}"


def generate_fees_section(epochs_data: Dict[str, Any], current_epoch_number: int) -> str:
    """Generate HTML for fees section.

    Args:
        epochs_data: Dictionary of epoch data keyed by epoch number
        current_epoch_number: Current epoch number

    Returns:
        HTML string for fees section
    """
    if not epochs_data:
        return """
        <div class="fees-section">
            <h2>Fee Distribution</h2>
            <p class="empty-state">No fee data available</p>
        </div>
        """

    # Aggregate all-time fees
    all_time_total_fees = 0.0
    current_epoch_fees = 0.0
    pool_cumulative_fees: Dict[str, float] = {}
    epoch_fees_history: List[Dict[str, Any]] = []

    # Sort epochs by number for chronological order
    sorted_epochs = sorted(epochs_data.items(), key=lambda x: int(x[0]))

    for epoch_key, epoch in sorted_epochs:
        epoch_num = int(epoch_key)
        incentives = epoch.get('incentives', {})
        epoch_total_fees = incentives.get('total_fees_usd', 0)

        all_time_total_fees += epoch_total_fees

        # Track current epoch fees
        if epoch_num == current_epoch_number:
            current_epoch_fees = epoch_total_fees

        # Build epoch history for chart
        epoch_fees_history.append({
            'epoch': epoch_num,
            'fees_usd': epoch_total_fees,
            'start_date': epoch.get('start_date', '')[:10] if epoch.get('start_date') else ''
        })

        # Aggregate per-pool fees
        pools = incentives.get('pools', [])
        for pool in pools:
            pool_name = pool.get('pool_name', 'Unknown Pool')
            pool_fees = pool.get('fees_usd', 0)
            if pool_name not in pool_cumulative_fees:
                pool_cumulative_fees[pool_name] = 0.0
            pool_cumulative_fees[pool_name] += pool_fees

    # Calculate average fees per epoch
    num_epochs = len(epochs_data)
    avg_fees_per_epoch = all_time_total_fees / num_epochs if num_epochs > 0 else 0

    # Sort pools by cumulative fees (descending)
    sorted_pools = sorted(pool_cumulative_fees.items(), key=lambda x: x[1], reverse=True)

    # Generate summary cards
    summary_html = f"""
    <div class="fees-summary">
        <div class="summary-card">
            <div class="card-icon">💵</div>
            <div class="card-content">
                <div class="card-label">All-Time Total Fees</div>
                <div class="card-value">{format_usd(all_time_total_fees)}</div>
            </div>
        </div>
        <div class="summary-card">
            <div class="card-icon">📅</div>
            <div class="card-content">
                <div class="card-label">Current Epoch Fees</div>
                <div class="card-value">{format_usd(current_epoch_fees)}</div>
            </div>
        </div>
        <div class="summary-card">
            <div class="card-icon">📊</div>
            <div class="card-content">
                <div class="card-label">Avg Fees / Epoch</div>
                <div class="card-value">{format_usd(avg_fees_per_epoch)}</div>
            </div>
        </div>
        <div class="summary-card">
            <div class="card-icon">🔢</div>
            <div class="card-content">
                <div class="card-label">Epochs Tracked</div>
                <div class="card-value">{num_epochs}</div>
            </div>
        </div>
    </div>
    """

    # Generate pool breakdown table
    pool_rows_html = ""
    for pool_name, cumulative_fees in sorted_pools:
        if cumulative_fees <= 0:
            continue
        pct_of_total = (cumulative_fees / all_time_total_fees * 100) if all_time_total_fees > 0 else 0
        pool_rows_html += f"""
        <tr>
            <td class="pool-name-cell">{pool_name}</td>
            <td class="fees-cell">{format_usd(cumulative_fees)}</td>
            <td class="pct-cell">{pct_of_total:.1f}%</td>
        </tr>
        """

    pool_table_html = f"""
    <div class="fees-pool-breakdown">
        <h3>Fee Distribution by Pool (All-Time)</h3>
        <div class="table-container">
            <table class="fees-table">
                <thead>
                    <tr>
                        <th>Pool</th>
                        <th>Cumulative Fees</th>
                        <th>% of Total</th>
                    </tr>
                </thead>
                <tbody>
                    {pool_rows_html if pool_rows_html else '<tr><td colspan="3" class="empty-row">No pool fee data available</td></tr>'}
                </tbody>
            </table>
        </div>
    </div>
    """

    # Generate chart container (data will be injected via JS)
    chart_html = """
    <div class="fees-chart-container">
        <h3>Fees by Epoch</h3>
        <div id="feesEpochChart" style="height: 400px;"></div>
    </div>
    """

    # Combine into full section
    html = f"""
    <div class="fees-section">
        <h2>Fee Distribution</h2>
        <p class="section-description">All-time fee distribution across pools and epochs. Tracking {num_epochs} epochs.</p>
        {summary_html}
        <div class="fees-content-grid">
            <div class="fees-chart-col">
                {chart_html}
            </div>
            <div class="fees-table-col">
                {pool_table_html}
            </div>
        </div>
    </div>
    """

    return html


def generate_fees_css() -> str:
    """Generate CSS for fees section.

    Returns:
        CSS string
    """
    css = """
    /* Fees Section Styles */
    .fees-section {
        margin-top: 30px;
        margin-bottom: 30px;
    }

    .fees-section h2 {
        font-size: 28px;
        margin-bottom: 10px;
        color: #2c3e50;
    }

    .fees-section h3 {
        font-size: 18px;
        margin-bottom: 15px;
        color: #2c3e50;
        font-weight: 600;
    }

    .fees-summary {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }

    .fees-content-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 30px;
        margin-top: 30px;
    }

    .fees-chart-col {
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 20px;
    }

    .fees-table-col {
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 20px;
    }

    .fees-pool-breakdown .table-container {
        max-height: 400px;
        overflow-y: auto;
    }

    .fees-table {
        width: 100%;
        border-collapse: collapse;
    }

    .fees-table th,
    .fees-table td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #e1e8ed;
    }

    .fees-table th {
        background: #f8f9fa;
        font-weight: 600;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #7f8c8d;
        position: sticky;
        top: 0;
    }

    .fees-table tbody tr:hover {
        background: #f8f9fa;
    }

    .pool-name-cell {
        font-weight: 500;
        color: #2c3e50;
    }

    .fees-cell {
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        color: #27ae60;
        font-weight: 600;
    }

    .pct-cell {
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        color: #7f8c8d;
    }

    .empty-row {
        text-align: center;
        color: #95a5a6;
        font-style: italic;
    }

    .fees-chart-container {
        min-height: 400px;
    }

    @media (max-width: 1024px) {
        .fees-content-grid {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 768px) {
        .fees-summary {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    """

    return css


def generate_fees_js(epochs_data: Dict[str, Any]) -> str:
    """Generate JavaScript for fees section chart.

    Args:
        epochs_data: Dictionary of epoch data keyed by epoch number

    Returns:
        JavaScript string
    """
    # Prepare chart data
    epoch_fees_history = []
    sorted_epochs = sorted(epochs_data.items(), key=lambda x: int(x[0]))

    for epoch_key, epoch in sorted_epochs:
        epoch_num = int(epoch_key)
        incentives = epoch.get('incentives', {})
        epoch_total_fees = incentives.get('total_fees_usd', 0)
        start_date = epoch.get('start_date', '')[:10] if epoch.get('start_date') else f'Epoch {epoch_num}'

        epoch_fees_history.append({
            'epoch': epoch_num,
            'fees_usd': epoch_total_fees,
            'label': f'E{epoch_num}'
        })

    chart_data_json = json.dumps(epoch_fees_history)

    js = f"""
    // Fees Epoch Chart
    (function() {{
        const feesData = {chart_data_json};

        function renderFeesChart() {{
            const chartEl = document.getElementById('feesEpochChart');
            if (!chartEl || typeof Plotly === 'undefined') return;

            const epochs = feesData.map(d => d.label);
            const fees = feesData.map(d => d.fees_usd);

            const trace = {{
                x: epochs,
                y: fees,
                type: 'bar',
                marker: {{
                    color: '#667eea',
                    line: {{
                        color: '#5a6fd6',
                        width: 1
                    }}
                }},
                hovertemplate: '<b>%{{x}}</b><br>Fees: $%{{y:,.2f}}<extra></extra>'
            }};

            const layout = {{
                margin: {{ t: 20, r: 20, b: 50, l: 60 }},
                xaxis: {{
                    title: 'Epoch',
                    tickangle: -45,
                    tickfont: {{ size: 11 }}
                }},
                yaxis: {{
                    title: 'Fees (USD)',
                    tickformat: '$,.0f',
                    gridcolor: '#e1e8ed'
                }},
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                hoverlabel: {{
                    bgcolor: '#2c3e50',
                    font: {{ color: 'white' }}
                }}
            }};

            const config = {{
                responsive: true,
                displayModeBar: false
            }};

            Plotly.newPlot(chartEl, [trace], layout, config);
        }}

        // Render when DOM is ready and when tab becomes visible
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', renderFeesChart);
        }} else {{
            renderFeesChart();
        }}

        // Re-render when fees tab becomes visible (for proper sizing)
        document.addEventListener('click', function(e) {{
            if (e.target.classList.contains('tab-btn') && e.target.getAttribute('data-tab') === 'fees') {{
                setTimeout(renderFeesChart, 100);
            }}
        }});
    }})();
    """

    return js
