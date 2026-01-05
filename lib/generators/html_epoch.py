"""HTML generator for epoch banner and timeline."""
from typing import Dict, Any


def generate_epoch_banner(epoch_info: Dict[str, Any], total_votes: float, unique_voters: int) -> str:
    """Generate HTML for epoch banner.

    Args:
        epoch_info: Epoch information dictionary
        total_votes: Total voting power in current epoch
        unique_voters: Number of unique voters

    Returns:
        HTML string for epoch banner
    """
    epoch_num = epoch_info['epoch_number']
    start_date = epoch_info['start_date'][:10]  # Just the date
    end_date = epoch_info['end_date'][:10]
    is_voting_open = epoch_info['is_voting_open']
    time_remaining = epoch_info['time_remaining_seconds']
    voting_time_remaining = epoch_info.get('voting_time_remaining_seconds', 0)

    status_class = "open" if is_voting_open else "closed"
    status_text = "Voting Open" if is_voting_open else "Voting Closed"

    html = f"""
    <!-- Epoch Banner -->
    <div class="epoch-banner">
        <div class="epoch-current">
            <h2>Epoch {epoch_num}</h2>
            <div class="epoch-dates">{start_date} - {end_date}</div>
        </div>
        <div class="epoch-timer">
            <div class="timer-label">Time Remaining</div>
            <div class="timer-value" id="epoch-countdown" data-target="{time_remaining}">
                Loading...
            </div>
        </div>
        <div class="voting-status {status_class}" id="voting-status">
            <span class="status-badge {status_class}">{status_text}</span>
        </div>
        <div class="epoch-stats">
            <div class="stat">
                <span class="label">Total Votes</span>
                <span class="value">{total_votes:.2f} veBTC</span>
            </div>
            <div class="stat">
                <span class="label">Participants</span>
                <span class="value">{unique_voters}</span>
            </div>
        </div>
    </div>
    """

    return html


def generate_epoch_banner_css() -> str:
    """Generate CSS for epoch banner.

    Returns:
        CSS string
    """
    css = """
    /* Epoch Banner Styles */
    .epoch-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px 40px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        flex-wrap: wrap;
        gap: 20px;
    }

    .epoch-current h2 {
        margin: 0 0 8px 0;
        font-size: 28px;
        font-weight: 700;
        color: white;
    }

    .epoch-dates {
        font-size: 14px;
        opacity: 0.9;
        font-weight: 500;
    }

    .epoch-timer, .voting-status {
        text-align: center;
        padding: 0 20px;
    }

    .timer-label {
        font-size: 12px;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    .timer-value {
        font-size: 32px;
        font-weight: 700;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
    }

    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .status-badge.open {
        background: rgba(76, 175, 80, 0.9);
    }

    .status-badge.closed {
        background: rgba(244, 67, 54, 0.9);
    }

    .epoch-stats {
        display: flex;
        gap: 30px;
    }

    .epoch-stats .stat {
        text-align: center;
    }

    .epoch-stats .label {
        display: block;
        font-size: 12px;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .epoch-stats .value {
        display: block;
        font-size: 24px;
        font-weight: 700;
    }

    @media (max-width: 1024px) {
        .epoch-banner {
            flex-direction: column;
            text-align: center;
        }

        .epoch-stats {
            width: 100%;
            justify-content: center;
        }
    }
    """

    return css


def generate_epoch_countdown_js() -> str:
    """Generate JavaScript for epoch countdown timer.

    Returns:
        JavaScript string
    """
    js = """
    // Epoch Countdown Timer
    function updateEpochCountdown() {
        const epochCountdown = document.getElementById('epoch-countdown');

        if (!epochCountdown) return;

        const epochTarget = parseInt(epochCountdown.getAttribute('data-target'));

        // Calculate time remaining
        const epochRemaining = Math.max(0, epochTarget);

        // Update epoch countdown
        epochCountdown.textContent = formatDuration(epochRemaining);

        // Decrease counter each second
        setTimeout(() => {
            if (epochTarget > 0) {
                epochCountdown.setAttribute('data-target', epochTarget - 1);
            }
        }, 1000);
    }

    function formatDuration(seconds) {
        if (seconds <= 0) return '0s';

        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;

        const parts = [];
        if (days > 0) parts.push(days + 'd');
        if (hours > 0) parts.push(hours + 'h');
        if (minutes > 0) parts.push(minutes + 'm');
        parts.push(secs + 's');  // Always show seconds

        return parts.join(' ') || '0s';
    }

    // Start countdown
    if (document.getElementById('epoch-countdown')) {
        updateEpochCountdown();
        setInterval(updateEpochCountdown, 1000);
    }
    """

    return js
