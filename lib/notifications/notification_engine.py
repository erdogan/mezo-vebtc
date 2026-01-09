"""Notification engine for Telegram bot."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import os

from lib.data_store import load_data, load_extended_data
from lib.data_fetcher import fetch_data_json
from lib.analytics.epoch_tracker import EpochTracker, get_current_epoch_info
from lib.analytics.participant import ParticipantAnalyzer
from lib.utils.time_utils import get_current_timestamp
from .subscriber_manager import SubscriberManager, Subscriber

logger = logging.getLogger(__name__)


class NotificationEngine:
    """Core notification logic and integration with analytics."""

    def __init__(self, data_file: str = "vebtc_data.json",
                 github_raw_url: str = None):
        """Initialize notification engine.

        Args:
            data_file: Path to veBTC data JSON file
            github_raw_url: Optional GitHub raw URL for fetching data (for cloud deployments)
        """
        self.data_file = data_file
        self.github_raw_url = github_raw_url or os.getenv('GITHUB_DATA_URL')
        self.epoch_tracker = EpochTracker()

    def load_current_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load latest vote/lock data.

        Returns:
            Tuple of (locks, votes)
        """
        try:
            # Fetch data from GitHub or local file
            data = fetch_data_json(self.data_file, self.github_raw_url)
            locks = data.get('parsed_locks', data.get('locks', []))
            votes = data.get('parsed_votes', data.get('votes', []))
            return locks, votes
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return [], []

    def load_incentives_data(self) -> Optional[List[Dict[str, Any]]]:
        """Load pool incentives data if available.

        Returns:
            List of pool incentives or None
        """
        try:
            data = fetch_data_json(self.data_file, self.github_raw_url)
            return data.get('incentives', None)
        except Exception as e:
            logger.error(f"Error loading incentives data: {e}")
            return None

    def check_if_user_voted(self, wallet_address: str, epoch_number: int) -> Tuple[bool, List[Dict[str, Any]]]:
        """Check if user voted in specific epoch.

        Args:
            wallet_address: Ethereum wallet address
            epoch_number: Epoch number to check

        Returns:
            Tuple of (has_voted, list_of_votes)
        """
        try:
            locks, votes = self.load_current_data()
            epoch_info = self.epoch_tracker.get_epoch_by_number(epoch_number)

            if not epoch_info:
                return False, []

            # Filter votes by user and epoch time window
            user_votes = []
            for v in votes:
                if v.get('voter', '').lower() != wallet_address.lower():
                    continue

                # Check if vote is within epoch voting window
                vote_ts = v.get('ts')
                if vote_ts:
                    if isinstance(vote_ts, str):
                        # Parse ISO string (could have 'T' or space separator)
                        dt = datetime.fromisoformat(vote_ts.replace('Z', '+00:00'))
                        vote_ts = dt.timestamp()
                    elif hasattr(vote_ts, 'timestamp'):
                        vote_ts = vote_ts.timestamp()
                    else:
                        vote_ts = float(vote_ts)

                    if epoch_info.vote_start_ts <= vote_ts <= epoch_info.vote_end_ts:
                        user_votes.append(v)

            has_voted = len(user_votes) > 0
            return has_voted, user_votes

        except Exception as e:
            logger.error(f"Error checking if user voted: {e}")
            return False, []

    def get_user_voting_power(self, wallet_address: str) -> float:
        """Get user's total voting power from all their locks.

        Args:
            wallet_address: Ethereum wallet address

        Returns:
            Total voting power in veBTC (sum of all locks)
        """
        try:
            locks, votes = self.load_current_data()

            # Sum up all locks for this user
            total_locked = 0.0
            for lock in locks:
                if lock.get('sender', '').lower() == wallet_address.lower():
                    total_locked += lock.get('amount', 0)

            return total_locked
        except Exception as e:
            logger.error(f"Error getting voting power: {e}")
            return 0.0

    def get_total_voted_in_epoch(self, epoch_number: int) -> float:
        """Get total voting power used in specific epoch.

        Args:
            epoch_number: Epoch number to check

        Returns:
            Total veBTC voted in the epoch
        """
        try:
            locks, votes = self.load_current_data()
            epoch_info = self.epoch_tracker.get_epoch_by_number(epoch_number)

            if not epoch_info:
                return 0.0

            # Sum all voting power from votes in this epoch
            total_voted = 0.0
            for v in votes:
                vote_ts = v.get('ts')
                if vote_ts:
                    if isinstance(vote_ts, str):
                        dt = datetime.fromisoformat(vote_ts.replace('Z', '+00:00'))
                        vote_ts = dt.timestamp()
                    elif hasattr(vote_ts, 'timestamp'):
                        vote_ts = vote_ts.timestamp()
                    else:
                        vote_ts = float(vote_ts)

                    if epoch_info.vote_start_ts <= vote_ts <= epoch_info.vote_end_ts:
                        total_voted += v.get('voting_power', 0)

            return total_voted
        except Exception as e:
            logger.error(f"Error getting total voted: {e}")
            return 0.0

    def get_unique_voters_in_epoch(self, epoch_number: int) -> int:
        """Get number of unique voters in specific epoch.

        Args:
            epoch_number: Epoch number

        Returns:
            Number of unique voters
        """
        try:
            locks, votes = self.load_current_data()
            epoch_info = self.epoch_tracker.get_epoch_by_number(epoch_number)

            if not epoch_info:
                return 0

            # Count unique voters in this epoch
            unique_voters = set()
            for v in votes:
                vote_ts = v.get('ts')
                if vote_ts:
                    if isinstance(vote_ts, str):
                        dt = datetime.fromisoformat(vote_ts.replace('Z', '+00:00'))
                        vote_ts = dt.timestamp()
                    elif hasattr(vote_ts, 'timestamp'):
                        vote_ts = vote_ts.timestamp()
                    else:
                        vote_ts = float(vote_ts)

                    if epoch_info.vote_start_ts <= vote_ts <= epoch_info.vote_end_ts:
                        voter = v.get('voter')
                        if voter and voter != 'Unknown':
                            unique_voters.add(voter.lower())

            return len(unique_voters)
        except Exception as e:
            logger.error(f"Error getting unique voters: {e}")
            return 0

    def get_pool_name(self, pool_address: str) -> str:
        """Get pool name from address.

        Args:
            pool_address: Pool contract address

        Returns:
            Pool name (e.g., "WBTC/ETH") or shortened address if not found
        """
        try:
            incentives_data = self.load_incentives_data()
            if not incentives_data:
                # Return shortened address if no data
                return f"{pool_address[:6]}...{pool_address[-4:]}"

            # Find pool by address
            for pool in incentives_data:
                if pool.get('pool_address', '').lower() == pool_address.lower():
                    pool_name = pool.get('pool_name', '')
                    if pool_name:
                        return pool_name

            # Not found, return shortened address
            return f"{pool_address[:6]}...{pool_address[-4:]}"
        except Exception as e:
            logger.error(f"Error getting pool name: {e}")
            return f"{pool_address[:6]}...{pool_address[-4:]}"

    def get_top_pools(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get top pools by APR.

        Args:
            limit: Number of top pools to return

        Returns:
            List of pool data dicts
        """
        try:
            incentives_data = self.load_incentives_data()
            if not incentives_data:
                return []

            # Sort by APR descending
            sorted_pools = sorted(
                incentives_data,
                key=lambda p: p.get('apr_total', 0),
                reverse=True
            )

            return sorted_pools[:limit]
        except Exception as e:
            logger.error(f"Error getting top pools: {e}")
            return []

    def get_high_apr_pools(self, threshold: float = 50.0) -> List[Dict[str, Any]]:
        """Get pools with APR above threshold.

        Args:
            threshold: APR threshold percentage

        Returns:
            List of pool data dicts
        """
        try:
            incentives_data = self.load_incentives_data()
            if not incentives_data:
                return []

            high_apr_pools = [
                pool for pool in incentives_data
                if pool.get('apr_total', 0) >= threshold
            ]

            # Sort by APR descending
            high_apr_pools.sort(key=lambda p: p.get('apr_total', 0), reverse=True)

            return high_apr_pools
        except Exception as e:
            logger.error(f"Error getting high APR pools: {e}")
            return []

    def should_send_24h_reminder(self) -> bool:
        """Check if 24h reminder should be sent now (24h after epoch started).

        Returns:
            True if reminder should be sent
        """
        try:
            current_ts = get_current_timestamp()
            epoch_info = self.epoch_tracker.get_current_epoch(current_ts)

            if not epoch_info.is_voting_open:
                return False

            # Calculate time since epoch start
            time_since_start = current_ts - epoch_info.start_ts

            # Send when 23.5h < time since start < 24.5h (84600 - 88200 seconds)
            # This is 24h after epoch started
            return 84600 < time_since_start < 88200

        except Exception as e:
            logger.error(f"Error checking 24h reminder: {e}")
            return False

    def should_send_final_warning(self) -> bool:
        """Check if final warning should be sent now.

        NOTE: Disabled per user request. Final warning notifications are no longer sent.

        Returns:
            False (disabled)
        """
        # Disabled - final warning notifications removed
        return False

    def should_send_epoch_start(self) -> bool:
        """Check if epoch start announcement should be sent now.

        Sends within 5-10 minutes of epoch starting.

        Returns:
            True if announcement should be sent
        """
        try:
            current_ts = get_current_timestamp()
            epoch_info = self.epoch_tracker.get_current_epoch(current_ts)

            if not epoch_info.is_voting_open:
                return False

            # Calculate time since epoch start
            time_since_start = current_ts - epoch_info.start_ts

            # Send when 5min < time since start < 10min (300 - 600 seconds)
            # This ensures we send near the beginning of the epoch
            return 300 < time_since_start < 600

        except Exception as e:
            logger.error(f"Error checking epoch start: {e}")
            return False

    def get_users_to_notify_24h(self) -> Dict[str, List[Subscriber]]:
        """Get subscribers to notify for 24h reminder.

        Returns:
            Dict with 'broadcast', 'not_voted', 'already_voted' lists
        """
        try:
            current_ts = get_current_timestamp()
            epoch_info = self.epoch_tracker.get_current_epoch(current_ts)
            epoch_number = epoch_info.epoch_number

            all_subscribers = self.subscriber_manager.get_all_subscribers()

            # Filter subscribers who want 24h reminders
            subscribers_24h = [
                s for s in all_subscribers
                if s.notify_24h_before
                and self.subscriber_manager.should_send_notification(
                    s.chat_id, '24h_reminder', epoch_number
                )
            ]

            # Separate linked and unlinked subscribers
            broadcast = []
            not_voted = []
            already_voted = []

            for subscriber in subscribers_24h:
                if not subscriber.wallet_address:
                    # Unlinked - send broadcast message
                    broadcast.append(subscriber)
                else:
                    # Linked - check if they voted
                    has_voted, votes = self.check_if_user_voted(
                        subscriber.wallet_address,
                        epoch_number
                    )

                    if has_voted:
                        already_voted.append(subscriber)
                    else:
                        not_voted.append(subscriber)

            return {
                'broadcast': broadcast,
                'not_voted': not_voted,
                'already_voted': already_voted
            }

        except Exception as e:
            logger.error(f"Error getting users for 24h notification: {e}")
            return {'broadcast': [], 'not_voted': [], 'already_voted': []}

    def get_users_to_notify_final_warning(self) -> List[Subscriber]:
        """Get subscribers to notify for final warning (only linked non-voters).

        Returns:
            List of Subscriber objects
        """
        try:
            current_ts = get_current_timestamp()
            epoch_info = self.epoch_tracker.get_current_epoch(current_ts)
            epoch_number = epoch_info.epoch_number

            linked_subscribers = self.subscriber_manager.get_linked_subscribers()

            # Filter: wants final warnings, hasn't been sent yet, hasn't voted
            final_warning_users = []

            for subscriber in linked_subscribers:
                if not subscriber.notify_final_warning:
                    continue

                if not self.subscriber_manager.should_send_notification(
                    subscriber.chat_id, 'final_warning', epoch_number
                ):
                    continue

                # Check if they voted
                has_voted, _ = self.check_if_user_voted(
                    subscriber.wallet_address,
                    epoch_number
                )

                if not has_voted:
                    final_warning_users.append(subscriber)

            return final_warning_users

        except Exception as e:
            logger.error(f"Error getting users for final warning: {e}")
            return []

    def get_users_to_notify_epoch_start(self) -> List[Subscriber]:
        """Get subscribers to notify for epoch start.

        Returns:
            List of Subscriber objects
        """
        try:
            current_ts = get_current_timestamp()
            epoch_info = self.epoch_tracker.get_current_epoch(current_ts)
            epoch_number = epoch_info.epoch_number

            all_subscribers = self.subscriber_manager.get_all_subscribers()

            # Filter subscribers who want epoch start notifications
            epoch_start_users = [
                s for s in all_subscribers
                if s.notify_epoch_start
                and self.subscriber_manager.should_send_notification(
                    s.chat_id, 'epoch_start', epoch_number
                )
            ]

            return epoch_start_users

        except Exception as e:
            logger.error(f"Error getting users for epoch start: {e}")
            return []

    def get_users_to_notify_high_apr(self, pool: Dict[str, Any]) -> List[Subscriber]:
        """Get subscribers to notify for high APR alert.

        Args:
            pool: Pool data dict

        Returns:
            List of Subscriber objects
        """
        try:
            pool_apr = pool.get('apr_total', 0)
            all_subscribers = self.subscriber_manager.get_all_subscribers()

            # Filter subscribers who want high APR alerts and meet threshold
            high_apr_users = [
                s for s in all_subscribers
                if s.notify_high_apr and pool_apr >= s.high_apr_threshold
            ]

            return high_apr_users

        except Exception as e:
            logger.error(f"Error getting users for high APR: {e}")
            return []

    def log_notification_sent(self, chat_id: int, notification_type: str, epoch_number: int) -> bool:
        """Log that notification was sent.

        Args:
            chat_id: Telegram chat ID
            notification_type: Type of notification
            epoch_number: Epoch number

        Returns:
            True if successful
        """
        return self.subscriber_manager.log_notification(chat_id, notification_type, epoch_number)

    def cleanup_old_logs(self) -> int:
        """Cleanup notification logs older than 10 epochs.

        Returns:
            Number of deleted records
        """
        try:
            current_ts = get_current_timestamp()
            current_epoch = self.epoch_tracker.get_current_epoch(current_ts).epoch_number
            cutoff_epoch = current_epoch - 10

            return self.subscriber_manager.cleanup_old_notifications(cutoff_epoch)

        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")
            return 0
