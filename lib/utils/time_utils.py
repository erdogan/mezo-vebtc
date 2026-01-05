"""Time utilities for formatting and conversions."""
from datetime import datetime, timedelta
from typing import Union


def format_timestamp(ts: Union[int, float], fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format Unix timestamp as string.

    Args:
        ts: Unix timestamp
        fmt: strftime format string

    Returns:
        Formatted datetime string
    """
    return datetime.fromtimestamp(ts).strftime(fmt)


def format_date(ts: Union[int, float]) -> str:
    """Format Unix timestamp as date only.

    Args:
        ts: Unix timestamp

    Returns:
        Date string (YYYY-MM-DD)
    """
    return format_timestamp(ts, "%Y-%m-%d")


def format_datetime_short(ts: Union[int, float]) -> str:
    """Format Unix timestamp as short datetime.

    Args:
        ts: Unix timestamp

    Returns:
        Datetime string (MMM DD, YYYY HH:MM)
    """
    return format_timestamp(ts, "%b %d, %Y %H:%M")


def format_duration(seconds: int) -> str:
    """Format duration in seconds as human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2d 15h 32m")
    """
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")

    return " ".join(parts) if parts else "< 1m"


def format_duration_long(seconds: int) -> str:
    """Format duration in seconds as long form.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2 days, 15 hours, 32 minutes")
    """
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60

    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")

    return ", ".join(parts) if parts else "less than 1 minute"


def get_current_timestamp() -> int:
    """Get current Unix timestamp.

    Returns:
        Current Unix timestamp as integer
    """
    return int(datetime.now().timestamp())


def timestamp_to_datetime(ts: Union[int, float]) -> datetime:
    """Convert Unix timestamp to datetime object.

    Args:
        ts: Unix timestamp

    Returns:
        datetime object
    """
    return datetime.fromtimestamp(ts)


def datetime_to_timestamp(dt: datetime) -> int:
    """Convert datetime to Unix timestamp.

    Args:
        dt: datetime object

    Returns:
        Unix timestamp as integer
    """
    return int(dt.timestamp())
