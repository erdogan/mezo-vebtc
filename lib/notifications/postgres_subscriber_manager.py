"""PostgreSQL-based subscriber manager for Railway deployment."""

import logging
import os
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class Subscriber:
    """Subscriber data class."""
    chat_id: int
    username: Optional[str] = None
    wallet_address: Optional[str] = None
    notify_24h_before: bool = True
    notify_final_warning: bool = True
    notify_epoch_start: bool = True
    notify_high_apr: bool = True
    high_apr_threshold: float = 50.0
    created_at: Optional[int] = None
    updated_at: Optional[int] = None


class PostgresSubscriberManager:
    """Manage subscribers using PostgreSQL database."""

    def __init__(self, database_url: str = None):
        """Initialize subscriber manager.

        Args:
            database_url: PostgreSQL connection URL (reads from DATABASE_URL env if not provided)
        """
        self.database_url = database_url or os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")

        self._init_database()

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.database_url)

    def _init_database(self):
        """Initialize database tables."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Create subscribers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS subscribers (
                    chat_id BIGINT PRIMARY KEY,
                    username TEXT,
                    wallet_address TEXT,
                    notify_24h_before BOOLEAN DEFAULT TRUE,
                    notify_final_warning BOOLEAN DEFAULT TRUE,
                    notify_epoch_start BOOLEAN DEFAULT TRUE,
                    notify_high_apr BOOLEAN DEFAULT TRUE,
                    high_apr_threshold FLOAT DEFAULT 50.0,
                    created_at BIGINT,
                    updated_at BIGINT
                )
            """)

            # Create index on wallet_address
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_wallet ON subscribers(wallet_address)
            """)

            # Create notification_log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notification_log (
                    id SERIAL PRIMARY KEY,
                    chat_id BIGINT,
                    notification_type TEXT,
                    epoch_number INTEGER,
                    sent_at BIGINT,
                    FOREIGN KEY (chat_id) REFERENCES subscribers(chat_id) ON DELETE CASCADE
                )
            """)

            # Create index on notification_log
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_notification_log
                ON notification_log(chat_id, notification_type, epoch_number)
            """)

            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Database tables initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def add_subscriber(self, chat_id: int, username: Optional[str] = None) -> bool:
        """Add or update subscriber.

        Args:
            chat_id: Telegram chat ID
            username: Telegram username

        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            current_time = int(datetime.now().timestamp())

            cursor.execute("""
                INSERT INTO subscribers (chat_id, username, created_at, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chat_id) DO UPDATE
                SET username = EXCLUDED.username,
                    updated_at = EXCLUDED.updated_at
            """, (chat_id, username, current_time, current_time))

            conn.commit()
            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error adding subscriber: {e}")
            return False

    def remove_subscriber(self, chat_id: int) -> bool:
        """Remove subscriber.

        Args:
            chat_id: Telegram chat ID

        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM subscribers WHERE chat_id = %s", (chat_id,))

            conn.commit()
            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error removing subscriber: {e}")
            return False

    def get_subscriber(self, chat_id: int) -> Optional[Subscriber]:
        """Get subscriber by chat ID.

        Args:
            chat_id: Telegram chat ID

        Returns:
            Subscriber object or None
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("SELECT * FROM subscribers WHERE chat_id = %s", (chat_id,))
            row = cursor.fetchone()

            cursor.close()
            conn.close()

            if row:
                return Subscriber(**dict(row))
            return None

        except Exception as e:
            logger.error(f"Error getting subscriber: {e}")
            return None

    def get_all_subscribers(self) -> List[Subscriber]:
        """Get all subscribers.

        Returns:
            List of Subscriber objects
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("SELECT * FROM subscribers")
            rows = cursor.fetchall()

            cursor.close()
            conn.close()

            return [Subscriber(**dict(row)) for row in rows]

        except Exception as e:
            logger.error(f"Error getting all subscribers: {e}")
            return []

    def link_wallet(self, chat_id: int, wallet_address: str) -> bool:
        """Link wallet address to subscriber.

        Args:
            chat_id: Telegram chat ID
            wallet_address: Ethereum wallet address

        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            current_time = int(datetime.now().timestamp())

            cursor.execute("""
                UPDATE subscribers
                SET wallet_address = %s, updated_at = %s
                WHERE chat_id = %s
            """, (wallet_address.lower(), current_time, chat_id))

            conn.commit()
            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error linking wallet: {e}")
            return False

    def unlink_wallet(self, chat_id: int) -> bool:
        """Unlink wallet from subscriber.

        Args:
            chat_id: Telegram chat ID

        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            current_time = int(datetime.now().timestamp())

            cursor.execute("""
                UPDATE subscribers
                SET wallet_address = NULL, updated_at = %s
                WHERE chat_id = %s
            """, (current_time, chat_id))

            conn.commit()
            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error unlinking wallet: {e}")
            return False

    def get_subscribers_by_wallet(self, wallet_address: str) -> List[Subscriber]:
        """Get subscribers with specific wallet address.

        Args:
            wallet_address: Ethereum wallet address

        Returns:
            List of Subscriber objects
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                "SELECT * FROM subscribers WHERE wallet_address = %s",
                (wallet_address.lower(),)
            )
            rows = cursor.fetchall()

            cursor.close()
            conn.close()

            return [Subscriber(**dict(row)) for row in rows]

        except Exception as e:
            logger.error(f"Error getting subscribers by wallet: {e}")
            return []

    def log_notification(self, chat_id: int, notification_type: str, epoch_number: int) -> bool:
        """Log that a notification was sent.

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

            current_time = int(datetime.now().timestamp())

            cursor.execute("""
                INSERT INTO notification_log (chat_id, notification_type, epoch_number, sent_at)
                VALUES (%s, %s, %s, %s)
            """, (chat_id, notification_type, epoch_number, current_time))

            conn.commit()
            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error logging notification: {e}")
            return False

    def was_notification_sent(self, chat_id: int, notification_type: str, epoch_number: int) -> bool:
        """Check if notification was already sent.

        Args:
            chat_id: Telegram chat ID
            notification_type: Type of notification
            epoch_number: Epoch number

        Returns:
            True if already sent
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM notification_log
                WHERE chat_id = %s AND notification_type = %s AND epoch_number = %s
            """, (chat_id, notification_type, epoch_number))

            count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            return count > 0

        except Exception as e:
            logger.error(f"Error checking notification: {e}")
            return False
