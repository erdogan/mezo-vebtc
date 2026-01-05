"""Subscriber database management for Telegram bot."""

import sqlite3
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Subscriber:
    """Subscriber data model."""
    chat_id: int
    username: Optional[str] = None
    wallet_address: Optional[str] = None
    notify_24h_before: bool = True
    notify_final_warning: bool = True
    notify_epoch_start: bool = True
    notify_high_apr: bool = True
    high_apr_threshold: float = 50.0
    created_at: int = 0
    updated_at: int = 0


class SubscriberManager:
    """Manages subscriber database operations."""

    def __init__(self, db_path: str = "subscribers.db"):
        """Initialize subscriber manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.Connection(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create subscribers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscribers (
                chat_id INTEGER PRIMARY KEY,
                username TEXT,
                wallet_address TEXT,
                notify_24h_before BOOLEAN DEFAULT 1,
                notify_final_warning BOOLEAN DEFAULT 1,
                notify_epoch_start BOOLEAN DEFAULT 1,
                notify_high_apr BOOLEAN DEFAULT 1,
                high_apr_threshold REAL DEFAULT 50.0,
                created_at INTEGER,
                updated_at INTEGER
            )
        """)

        # Create index for wallet lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_wallet
            ON subscribers(wallet_address)
        """)

        # Create notification log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notification_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                notification_type TEXT,
                epoch_number INTEGER,
                sent_at INTEGER,
                FOREIGN KEY (chat_id) REFERENCES subscribers(chat_id)
            )
        """)

        # Create index for notification log lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_notification_log
            ON notification_log(chat_id, notification_type, epoch_number)
        """)

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def add_subscriber(self, chat_id: int, username: Optional[str] = None) -> bool:
        """Add or update a subscriber.

        Args:
            chat_id: Telegram chat ID
            username: Telegram username (optional)

        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            current_time = int(time.time())

            # Check if subscriber exists
            cursor.execute("SELECT chat_id FROM subscribers WHERE chat_id = ?", (chat_id,))
            exists = cursor.fetchone() is not None

            if exists:
                # Update existing subscriber
                cursor.execute("""
                    UPDATE subscribers
                    SET username = ?, updated_at = ?
                    WHERE chat_id = ?
                """, (username, current_time, chat_id))
                logger.info(f"Updated subscriber: {chat_id}")
            else:
                # Insert new subscriber
                cursor.execute("""
                    INSERT INTO subscribers
                    (chat_id, username, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (chat_id, username, current_time, current_time))
                logger.info(f"Added new subscriber: {chat_id}")

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error adding subscriber {chat_id}: {e}")
            return False

    def remove_subscriber(self, chat_id: int) -> bool:
        """Remove a subscriber.

        Args:
            chat_id: Telegram chat ID

        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM subscribers WHERE chat_id = ?", (chat_id,))
            cursor.execute("DELETE FROM notification_log WHERE chat_id = ?", (chat_id,))

            conn.commit()
            conn.close()
            logger.info(f"Removed subscriber: {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing subscriber {chat_id}: {e}")
            return False

    def get_subscriber(self, chat_id: int) -> Optional[Subscriber]:
        """Get subscriber by chat ID.

        Args:
            chat_id: Telegram chat ID

        Returns:
            Subscriber object or None if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM subscribers WHERE chat_id = ?", (chat_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return Subscriber(
                    chat_id=row['chat_id'],
                    username=row['username'],
                    wallet_address=row['wallet_address'],
                    notify_24h_before=bool(row['notify_24h_before']),
                    notify_final_warning=bool(row['notify_final_warning']),
                    notify_epoch_start=bool(row['notify_epoch_start']),
                    notify_high_apr=bool(row['notify_high_apr']),
                    high_apr_threshold=float(row['high_apr_threshold']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            return None
        except Exception as e:
            logger.error(f"Error getting subscriber {chat_id}: {e}")
            return None

    def get_all_subscribers(self) -> List[Subscriber]:
        """Get all subscribers.

        Returns:
            List of Subscriber objects
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM subscribers")
            rows = cursor.fetchall()
            conn.close()

            return [
                Subscriber(
                    chat_id=row['chat_id'],
                    username=row['username'],
                    wallet_address=row['wallet_address'],
                    notify_24h_before=bool(row['notify_24h_before']),
                    notify_final_warning=bool(row['notify_final_warning']),
                    notify_epoch_start=bool(row['notify_epoch_start']),
                    notify_high_apr=bool(row['notify_high_apr']),
                    high_apr_threshold=float(row['high_apr_threshold']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error getting all subscribers: {e}")
            return []

    def link_wallet(self, chat_id: int, wallet_address: str) -> bool:
        """Link a wallet address to a subscriber.

        Args:
            chat_id: Telegram chat ID
            wallet_address: Ethereum wallet address

        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            current_time = int(time.time())

            cursor.execute("""
                UPDATE subscribers
                SET wallet_address = ?, updated_at = ?
                WHERE chat_id = ?
            """, (wallet_address.lower(), current_time, chat_id))

            conn.commit()
            conn.close()
            logger.info(f"Linked wallet {wallet_address} to {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error linking wallet for {chat_id}: {e}")
            return False

    def unlink_wallet(self, chat_id: int) -> bool:
        """Remove wallet link from a subscriber.

        Args:
            chat_id: Telegram chat ID

        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            current_time = int(time.time())

            cursor.execute("""
                UPDATE subscribers
                SET wallet_address = NULL, updated_at = ?
                WHERE chat_id = ?
            """, (current_time, chat_id))

            conn.commit()
            conn.close()
            logger.info(f"Unlinked wallet for {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error unlinking wallet for {chat_id}: {e}")
            return False

    def update_preferences(self, chat_id: int, **kwargs) -> bool:
        """Update subscriber notification preferences.

        Args:
            chat_id: Telegram chat ID
            **kwargs: Preference fields to update

        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Build update query dynamically
            valid_fields = [
                'notify_24h_before', 'notify_final_warning',
                'notify_epoch_start', 'notify_high_apr', 'high_apr_threshold'
            ]

            updates = []
            values = []
            for field, value in kwargs.items():
                if field in valid_fields:
                    updates.append(f"{field} = ?")
                    values.append(value)

            if not updates:
                return False

            # Add updated_at
            updates.append("updated_at = ?")
            values.append(int(time.time()))
            values.append(chat_id)

            query = f"UPDATE subscribers SET {', '.join(updates)} WHERE chat_id = ?"
            cursor.execute(query, values)

            conn.commit()
            conn.close()
            logger.info(f"Updated preferences for {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating preferences for {chat_id}: {e}")
            return False

    def log_notification(self, chat_id: int, notification_type: str, epoch_number: int) -> bool:
        """Log a sent notification.

        Args:
            chat_id: Telegram chat ID
            notification_type: Type of notification
            epoch_number: Epoch number

        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO notification_log
                (chat_id, notification_type, epoch_number, sent_at)
                VALUES (?, ?, ?, ?)
            """, (chat_id, notification_type, epoch_number, int(time.time())))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error logging notification for {chat_id}: {e}")
            return False

    def should_send_notification(self, chat_id: int, notification_type: str, epoch_number: int) -> bool:
        """Check if notification should be sent (not already sent).

        Args:
            chat_id: Telegram chat ID
            notification_type: Type of notification
            epoch_number: Epoch number

        Returns:
            True if notification should be sent
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 1 FROM notification_log
                WHERE chat_id = ? AND notification_type = ? AND epoch_number = ?
            """, (chat_id, notification_type, epoch_number))

            result = cursor.fetchone()
            conn.close()

            return result is None
        except Exception as e:
            logger.error(f"Error checking notification status for {chat_id}: {e}")
            return False

    def cleanup_old_notifications(self, cutoff_epoch: int) -> int:
        """Delete notification logs older than cutoff epoch.

        Args:
            cutoff_epoch: Epoch number cutoff

        Returns:
            Number of deleted records
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM notification_log
                WHERE epoch_number < ?
            """, (cutoff_epoch,))

            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            logger.info(f"Cleaned up {deleted} old notification logs")
            return deleted
        except Exception as e:
            logger.error(f"Error cleaning up notifications: {e}")
            return 0

    def get_subscriber_count(self) -> int:
        """Get total number of subscribers.

        Returns:
            Number of subscribers
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as count FROM subscribers")
            result = cursor.fetchone()
            conn.close()

            return result['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting subscriber count: {e}")
            return 0

    def get_linked_subscribers(self) -> List[Subscriber]:
        """Get all subscribers with linked wallets.

        Returns:
            List of Subscriber objects with wallet addresses
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM subscribers
                WHERE wallet_address IS NOT NULL
            """)
            rows = cursor.fetchall()
            conn.close()

            return [
                Subscriber(
                    chat_id=row['chat_id'],
                    username=row['username'],
                    wallet_address=row['wallet_address'],
                    notify_24h_before=bool(row['notify_24h_before']),
                    notify_final_warning=bool(row['notify_final_warning']),
                    notify_epoch_start=bool(row['notify_epoch_start']),
                    notify_high_apr=bool(row['notify_high_apr']),
                    high_apr_threshold=float(row['high_apr_threshold']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error getting linked subscribers: {e}")
            return []
