"""Analytics and statistics for Telegram bot."""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from lib.utils.time_utils import get_current_timestamp

logger = logging.getLogger(__name__)


class BotAnalytics:
    """Bot usage analytics and statistics."""

    def __init__(self, subscriber_manager):
        """Initialize analytics.

        Args:
            subscriber_manager: SubscriberManager instance
        """
        self.subscriber_manager = subscriber_manager

    def get_total_subscribers(self) -> int:
        """Get total number of subscribers.

        Returns:
            Total subscriber count
        """
        try:
            subscribers = self.subscriber_manager.get_all_subscribers()
            return len(subscribers)
        except Exception as e:
            logger.error(f"Error getting total subscribers: {e}")
            return 0

    def get_linked_wallets_count(self) -> int:
        """Get number of subscribers with linked wallets.

        Returns:
            Number of linked wallets
        """
        try:
            linked = self.subscriber_manager.get_linked_subscribers()
            return len(linked)
        except Exception as e:
            logger.error(f"Error getting linked wallets count: {e}")
            return 0

    def get_recent_subscribers(self, days: int = 7) -> int:
        """Get number of subscribers added in recent days.

        Args:
            days: Number of days to look back

        Returns:
            Number of recent subscribers
        """
        try:
            current_ts = get_current_timestamp()
            cutoff_ts = current_ts - (days * 86400)

            subscribers = self.subscriber_manager.get_all_subscribers()
            recent = [
                s for s in subscribers
                if s.created_at >= cutoff_ts
            ]
            return len(recent)
        except Exception as e:
            logger.error(f"Error getting recent subscribers: {e}")
            return 0

    def get_notification_stats(self, epoch_number: int = None) -> Dict[str, int]:
        """Get notification statistics.

        Args:
            epoch_number: Optional epoch to filter by (None for all)

        Returns:
            Dictionary with notification counts by type
        """
        try:
            conn = self.subscriber_manager._get_connection()
            cursor = conn.cursor()

            if epoch_number:
                cursor.execute("""
                    SELECT notification_type, COUNT(*) as count
                    FROM notification_log
                    WHERE epoch_number = ?
                    GROUP BY notification_type
                """, (epoch_number,))
            else:
                cursor.execute("""
                    SELECT notification_type, COUNT(*) as count
                    FROM notification_log
                    GROUP BY notification_type
                """)

            stats = {}
            for row in cursor.fetchall():
                notification_type = row[0]
                count = row[1]
                stats[notification_type] = count

            conn.close()
            return stats

        except Exception as e:
            logger.error(f"Error getting notification stats: {e}")
            return {}

    def get_active_users(self, days: int = 7) -> int:
        """Get number of users who received notifications recently.

        Args:
            days: Number of days to look back

        Returns:
            Number of active users
        """
        try:
            current_ts = get_current_timestamp()
            cutoff_ts = current_ts - (days * 86400)

            conn = self.subscriber_manager._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(DISTINCT chat_id)
                FROM notification_log
                WHERE sent_at >= ?
            """, (cutoff_ts,))

            result = cursor.fetchone()
            conn.close()
            return result[0] if result else 0

        except Exception as e:
            logger.error(f"Error getting active users: {e}")
            return 0

    def get_notification_preferences_breakdown(self) -> Dict[str, int]:
        """Get breakdown of notification preferences.

        Returns:
            Dictionary with counts for each preference type
        """
        try:
            subscribers = self.subscriber_manager.get_all_subscribers()

            breakdown = {
                '24h_reminder': 0,
                'final_warning': 0,
                'epoch_start': 0,
                'high_apr': 0
            }

            for sub in subscribers:
                if sub.notify_24h_before:
                    breakdown['24h_reminder'] += 1
                if sub.notify_final_warning:
                    breakdown['final_warning'] += 1
                if sub.notify_epoch_start:
                    breakdown['epoch_start'] += 1
                if sub.notify_high_apr:
                    breakdown['high_apr'] += 1

            return breakdown

        except Exception as e:
            logger.error(f"Error getting preference breakdown: {e}")
            return {}

    def get_comprehensive_stats(self, current_epoch: int = None) -> Dict[str, Any]:
        """Get comprehensive bot statistics.

        Args:
            current_epoch: Current epoch number for epoch-specific stats

        Returns:
            Dictionary with all stats
        """
        total_subs = self.get_total_subscribers()
        linked_wallets = self.get_linked_wallets_count()
        recent_7d = self.get_recent_subscribers(days=7)
        recent_30d = self.get_recent_subscribers(days=30)
        active_7d = self.get_active_users(days=7)
        active_30d = self.get_active_users(days=30)

        # Notification stats
        all_time_notifications = self.get_notification_stats()
        current_epoch_notifications = self.get_notification_stats(current_epoch) if current_epoch else {}

        # Preferences
        preferences = self.get_notification_preferences_breakdown()

        # Calculate percentages
        wallet_link_rate = (linked_wallets / total_subs * 100) if total_subs > 0 else 0
        active_rate_7d = (active_7d / total_subs * 100) if total_subs > 0 else 0
        active_rate_30d = (active_30d / total_subs * 100) if total_subs > 0 else 0

        return {
            'subscribers': {
                'total': total_subs,
                'linked_wallets': linked_wallets,
                'wallet_link_rate': wallet_link_rate,
                'recent_7d': recent_7d,
                'recent_30d': recent_30d
            },
            'engagement': {
                'active_7d': active_7d,
                'active_30d': active_30d,
                'active_rate_7d': active_rate_7d,
                'active_rate_30d': active_rate_30d
            },
            'notifications': {
                'all_time': all_time_notifications,
                'current_epoch': current_epoch_notifications
            },
            'preferences': preferences
        }
