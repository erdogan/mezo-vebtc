"""Epoch tracking and calculations for veBTC dashboard."""
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EpochInfo:
    """Information about an epoch."""
    epoch_number: int
    start_ts: int
    end_ts: int
    vote_start_ts: int
    vote_end_ts: int
    is_voting_open: bool
    time_remaining: timedelta
    voting_time_remaining: Optional[timedelta]
    current_ts: int

    @property
    def start_date(self) -> datetime:
        """Get epoch start as datetime."""
        return datetime.fromtimestamp(self.start_ts)

    @property
    def end_date(self) -> datetime:
        """Get epoch end as datetime."""
        return datetime.fromtimestamp(self.end_ts)

    @property
    def vote_start_date(self) -> datetime:
        """Get voting start as datetime."""
        return datetime.fromtimestamp(self.vote_start_ts)

    @property
    def vote_end_date(self) -> datetime:
        """Get voting end as datetime."""
        return datetime.fromtimestamp(self.vote_end_ts)


class EpochTracker:
    """Tracks and calculates epoch information."""

    WEEK_SECONDS = 604800  # 7 days
    VOTE_BUFFER_SECONDS = 3600  # 1 hour

    def __init__(self, current_timestamp: Optional[int] = None):
        """Initialize epoch tracker.

        Args:
            current_timestamp: Current Unix timestamp (None for now)
        """
        self.current_timestamp = current_timestamp or int(datetime.now().timestamp())

    def epoch_start(self, timestamp: int) -> int:
        """Calculate epoch start timestamp.

        Args:
            timestamp: Unix timestamp

        Returns:
            Epoch start timestamp
        """
        return timestamp - (timestamp % self.WEEK_SECONDS)

    def epoch_next(self, timestamp: int) -> int:
        """Calculate next epoch start timestamp.

        Args:
            timestamp: Unix timestamp

        Returns:
            Next epoch start timestamp
        """
        return self.epoch_start(timestamp) + self.WEEK_SECONDS

    def epoch_vote_start(self, timestamp: int) -> int:
        """Calculate voting window start timestamp.

        Args:
            timestamp: Unix timestamp

        Returns:
            Voting start timestamp (1 hour after epoch start)
        """
        return self.epoch_start(timestamp) + self.VOTE_BUFFER_SECONDS

    def epoch_vote_end(self, timestamp: int) -> int:
        """Calculate voting window end timestamp.

        Args:
            timestamp: Unix timestamp

        Returns:
            Voting end timestamp (1 hour before epoch end)
        """
        return self.epoch_next(timestamp) - self.VOTE_BUFFER_SECONDS

    def get_epoch_number(self, timestamp: int, genesis_timestamp: int = 1733836800) -> int:
        """Calculate epoch number since genesis.

        Args:
            timestamp: Unix timestamp
            genesis_timestamp: First epoch start (default: Dec 10, 2024)

        Returns:
            Epoch number (0-indexed)
        """
        epoch_start_ts = self.epoch_start(timestamp)
        return (epoch_start_ts - genesis_timestamp) // self.WEEK_SECONDS

    def is_voting_open(self, timestamp: Optional[int] = None) -> bool:
        """Check if voting window is currently open.

        Args:
            timestamp: Unix timestamp (None for current)

        Returns:
            True if voting is open
        """
        ts = timestamp or self.current_timestamp
        vote_start = self.epoch_vote_start(ts)
        vote_end = self.epoch_vote_end(ts)
        return vote_start <= ts <= vote_end

    def get_current_epoch(self, timestamp: Optional[int] = None) -> EpochInfo:
        """Get information about the current epoch.

        Args:
            timestamp: Unix timestamp (None for current)

        Returns:
            EpochInfo object with epoch details
        """
        ts = timestamp or self.current_timestamp

        start_ts = self.epoch_start(ts)
        end_ts = self.epoch_next(ts)
        vote_start_ts = self.epoch_vote_start(ts)
        vote_end_ts = self.epoch_vote_end(ts)

        epoch_number = self.get_epoch_number(ts)
        is_voting = self.is_voting_open(ts)

        # Calculate time remaining in epoch
        time_remaining = timedelta(seconds=end_ts - ts)

        # Calculate voting time remaining (if voting is open)
        voting_time_remaining = None
        if is_voting:
            voting_time_remaining = timedelta(seconds=vote_end_ts - ts)

        return EpochInfo(
            epoch_number=epoch_number,
            start_ts=start_ts,
            end_ts=end_ts,
            vote_start_ts=vote_start_ts,
            vote_end_ts=vote_end_ts,
            is_voting_open=is_voting,
            time_remaining=time_remaining,
            voting_time_remaining=voting_time_remaining,
            current_ts=ts
        )

    def get_epoch_by_number(self, epoch_number: int, genesis_timestamp: int = 1733836800) -> EpochInfo:
        """Get information about a specific epoch by number.

        Args:
            epoch_number: Epoch number (0-indexed)
            genesis_timestamp: First epoch start

        Returns:
            EpochInfo object
        """
        start_ts = genesis_timestamp + (epoch_number * self.WEEK_SECONDS)
        return self.get_current_epoch(timestamp=start_ts)

    def format_time_remaining(self, td: timedelta) -> str:
        """Format timedelta as human-readable string.

        Args:
            td: Timedelta object

        Returns:
            Formatted string (e.g., "2d 15h 32m")
        """
        total_seconds = int(td.total_seconds())

        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")

        return " ".join(parts) if parts else "< 1m"

    def get_epoch_timeline(self, num_epochs: int = 10, current_timestamp: Optional[int] = None) -> list:
        """Get timeline of recent epochs.

        Args:
            num_epochs: Number of past epochs to include
            current_timestamp: Current timestamp (None for now)

        Returns:
            List of EpochInfo objects
        """
        ts = current_timestamp or self.current_timestamp
        current_epoch_num = self.get_epoch_number(ts)

        timeline = []
        for i in range(num_epochs, -1, -1):
            epoch_num = current_epoch_num - i
            if epoch_num >= 0:
                epoch_info = self.get_epoch_by_number(epoch_num)
                timeline.append(epoch_info)

        return timeline


def get_current_epoch_info(current_timestamp: Optional[int] = None) -> Dict[str, Any]:
    """Get current epoch information as dictionary.

    Args:
        current_timestamp: Current Unix timestamp (None for now)

    Returns:
        Dictionary with epoch information
    """
    tracker = EpochTracker(current_timestamp)
    epoch = tracker.get_current_epoch()

    return {
        "epoch_number": epoch.epoch_number,
        "start_ts": epoch.start_ts,
        "end_ts": epoch.end_ts,
        "vote_start_ts": epoch.vote_start_ts,
        "vote_end_ts": epoch.vote_end_ts,
        "start_date": epoch.start_date.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": epoch.end_date.strftime("%Y-%m-%d %H:%M:%S"),
        "is_voting_open": epoch.is_voting_open,
        "time_remaining_seconds": int(epoch.time_remaining.total_seconds()),
        "time_remaining_formatted": tracker.format_time_remaining(epoch.time_remaining),
        "voting_time_remaining_seconds": int(epoch.voting_time_remaining.total_seconds()) if epoch.voting_time_remaining else None,
        "voting_time_remaining_formatted": tracker.format_time_remaining(epoch.voting_time_remaining) if epoch.voting_time_remaining else None
    }
