#!/usr/bin/env python3
"""Telegram notification bot for veBTC voting system."""

import os
import sys
import logging
import asyncio
from datetime import datetime
from typing import List

from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram.constants import ParseMode
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from lib.config import load_config
from lib.utils.time_utils import get_current_timestamp, format_datetime_short
from lib.analytics.epoch_tracker import get_current_epoch_info
from lib.notifications.subscriber_manager import SubscriberManager, Subscriber
from lib.notifications.postgres_subscriber_manager import PostgresSubscriberManager
from lib.notifications.notification_engine import NotificationEngine
from lib.notifications.bot_commands import BotCommands
from lib.notifications.message_templates import MessageTemplates

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('telegram_bot.log')
    ]
)

logger = logging.getLogger(__name__)


class VeBTCBot:
    """Main Telegram bot class."""

    def __init__(self):
        """Initialize bot."""
        # Load environment variables
        load_dotenv()

        # Load configuration
        self.config = load_config("config.json")

        # Get bot token from environment
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")

        # Get data file path
        data_file = os.getenv("DATA_FILE", "vebtc_data.json")
        logger.info(f"Using data file: {data_file}")

        # Check if PostgreSQL database URL is available (Railway)
        database_url = os.getenv("DATABASE_URL")

        if database_url:
            logger.info("Using PostgreSQL database (persistent)")
            self.subscriber_manager = PostgresSubscriberManager(database_url)
        else:
            # Fall back to SQLite (local development)
            db_path = os.getenv("DATABASE_PATH", "subscribers.db")
            logger.info(f"Using SQLite database: {db_path}")
            self.subscriber_manager = SubscriberManager(db_path)

        # Initialize notification engine (it doesn't use the DB directly)
        self.notification_engine = NotificationEngine(data_file)
        self.bot_commands = BotCommands(self.subscriber_manager, self.notification_engine)
        self.templates = MessageTemplates()

        # Initialize application
        self.application = Application.builder().token(self.bot_token).build()

        # Initialize scheduler
        self.scheduler = AsyncIOScheduler()

        # Register handlers
        self._register_handlers()

        # Setup post-init to start scheduler
        self.application.post_init = self._post_init
        self.application.post_shutdown = self._post_shutdown

        logger.info("VeBTC Bot initialized successfully")

    async def _post_init(self, application: Application) -> None:
        """Post-initialization hook to start scheduler."""
        self._setup_scheduler()
        self.scheduler.start()
        logger.info("Scheduler started in post_init")

    async def _post_shutdown(self, application: Application) -> None:
        """Post-shutdown hook to stop scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped in post_shutdown")

    def _register_handlers(self):
        """Register command handlers."""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.bot_commands.start_command))
        self.application.add_handler(CommandHandler("subscribe", self.bot_commands.subscribe_command))
        self.application.add_handler(CommandHandler("unsubscribe", self.bot_commands.unsubscribe_command))
        self.application.add_handler(CommandHandler("link", self.bot_commands.link_command))
        self.application.add_handler(CommandHandler("unlink", self.bot_commands.unlink_command))
        self.application.add_handler(CommandHandler("epoch", self.bot_commands.status_command))
        self.application.add_handler(CommandHandler("myvotes", self.bot_commands.myvotes_command))
        self.application.add_handler(CommandHandler("pools", self.bot_commands.pools_command))
        self.application.add_handler(CommandHandler("settings", self.bot_commands.settings_command))
        self.application.add_handler(CommandHandler("help", self.bot_commands.help_command))

        # Hidden test command (not shown in help)
        self.application.add_handler(CommandHandler("test", self.bot_commands.test_notification_command))
        self.application.add_handler(CommandHandler("botstats", self.bot_commands.stats_command))

        # Unknown command handler (must be after all command handlers)
        self.application.add_handler(MessageHandler(filters.COMMAND, self.bot_commands.unknown_command))

        # Error handler
        self.application.add_error_handler(self.bot_commands.error_handler)

        logger.info("Command handlers registered")

    def _setup_scheduler(self):
        """Setup notification scheduler."""
        # Check notifications every 5 minutes
        check_interval = self.config.get('telegram.notification_check_interval', 300)

        self.scheduler.add_job(
            self._check_notifications,
            IntervalTrigger(seconds=check_interval),
            id='check_notifications',
            name='Check and send notifications',
            replace_existing=True
        )

        # Cleanup old logs weekly (Mondays at 00:00)
        self.scheduler.add_job(
            self._cleanup_old_logs,
            'cron',
            day_of_week='mon',
            hour=0,
            minute=0,
            id='cleanup_logs',
            name='Cleanup old notification logs',
            replace_existing=True
        )

        logger.info(f"Scheduler configured (check interval: {check_interval}s)")

    async def _check_notifications(self):
        """Check and send notifications."""
        try:
            logger.info("Checking notifications...")

            # Check 24h reminder
            if self.notification_engine.should_send_24h_reminder():
                await self._send_24h_reminders()

            # Check final warning
            elif self.notification_engine.should_send_final_warning():
                await self._send_final_warnings()

            # Check epoch start
            elif self.notification_engine.should_send_epoch_start():
                await self._send_epoch_start_announcements()

            # Check high APR alerts (run every check)
            await self._check_high_apr_alerts()

        except Exception as e:
            logger.error(f"Error checking notifications: {e}")

    async def _send_24h_reminders(self):
        """Send 24h voting reminders."""
        try:
            logger.info("Sending 24h reminders...")

            current_ts = get_current_timestamp()
            epoch_info = get_current_epoch_info(current_ts)
            epoch_number = epoch_info['epoch_number']

            # Get users to notify
            users = self.notification_engine.get_users_to_notify_24h()

            # Get top pools
            top_pools = self.notification_engine.get_top_pools(limit=3)

            # Close time
            close_time = format_datetime_short(epoch_info['vote_end_ts'])

            # Send broadcast messages
            for subscriber in users['broadcast']:
                try:
                    message = self.templates.notification_24h_reminder_broadcast(
                        epoch_number=epoch_number,
                        close_time=close_time,
                        top_pools=top_pools
                    )

                    await self.application.bot.send_message(
                        chat_id=subscriber.chat_id,
                        text=message,
                        parse_mode=ParseMode.MARKDOWN_V2
                    )

                    self.notification_engine.log_notification_sent(
                        subscriber.chat_id, '24h_reminder', epoch_number
                    )

                    logger.info(f"Sent 24h broadcast to {subscriber.chat_id}")
                    await asyncio.sleep(0.1)  # Rate limiting

                except Exception as e:
                    logger.error(f"Error sending 24h broadcast to {subscriber.chat_id}: {e}")

            # Send personalized messages (not voted)
            for subscriber in users['not_voted']:
                try:
                    voting_power = self.notification_engine.get_user_voting_power(subscriber.wallet_address)

                    message = self.templates.notification_24h_reminder_personalized(
                        username=subscriber.username or 'there',
                        epoch_number=epoch_number,
                        close_time=close_time,
                        voting_power=voting_power,
                        top_pools=top_pools,
                        has_voted=False
                    )

                    await self.application.bot.send_message(
                        chat_id=subscriber.chat_id,
                        text=message,
                        parse_mode=ParseMode.MARKDOWN_V2
                    )

                    self.notification_engine.log_notification_sent(
                        subscriber.chat_id, '24h_reminder', epoch_number
                    )

                    logger.info(f"Sent 24h personalized (not voted) to {subscriber.chat_id}")
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error sending 24h personalized to {subscriber.chat_id}: {e}")

            # Send personalized messages (already voted)
            for subscriber in users['already_voted']:
                try:
                    voting_power = self.notification_engine.get_user_voting_power(subscriber.wallet_address)

                    message = self.templates.notification_24h_reminder_personalized(
                        username=subscriber.username or 'there',
                        epoch_number=epoch_number,
                        close_time=close_time,
                        voting_power=voting_power,
                        top_pools=top_pools,
                        has_voted=True
                    )

                    await self.application.bot.send_message(
                        chat_id=subscriber.chat_id,
                        text=message,
                        parse_mode=ParseMode.MARKDOWN_V2
                    )

                    self.notification_engine.log_notification_sent(
                        subscriber.chat_id, '24h_reminder', epoch_number
                    )

                    logger.info(f"Sent 24h personalized (already voted) to {subscriber.chat_id}")
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error sending 24h personalized to {subscriber.chat_id}: {e}")

            logger.info(f"24h reminders complete: {len(users['broadcast'])} broadcast, "
                       f"{len(users['not_voted'])} not voted, {len(users['already_voted'])} already voted")

        except Exception as e:
            logger.error(f"Error in _send_24h_reminders: {e}")

    async def _send_final_warnings(self):
        """Send final warnings to non-voters."""
        try:
            logger.info("Sending final warnings...")

            current_ts = get_current_timestamp()
            epoch_info = get_current_epoch_info(current_ts)
            epoch_number = epoch_info['epoch_number']

            # Get users to notify (only linked non-voters)
            users = self.notification_engine.get_users_to_notify_final_warning()

            close_time = format_datetime_short(epoch_info['vote_end_ts'])

            for subscriber in users:
                try:
                    voting_power = self.notification_engine.get_user_voting_power(subscriber.wallet_address)

                    message = self.templates.notification_final_warning(
                        username=subscriber.username or 'there',
                        epoch_number=epoch_number,
                        close_time=close_time,
                        voting_power=voting_power
                    )

                    await self.application.bot.send_message(
                        chat_id=subscriber.chat_id,
                        text=message,
                        parse_mode=ParseMode.MARKDOWN_V2
                    )

                    self.notification_engine.log_notification_sent(
                        subscriber.chat_id, 'final_warning', epoch_number
                    )

                    logger.info(f"Sent final warning to {subscriber.chat_id}")
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error sending final warning to {subscriber.chat_id}: {e}")

            logger.info(f"Final warnings complete: {len(users)} sent")

        except Exception as e:
            logger.error(f"Error in _send_final_warnings: {e}")

    async def _send_epoch_start_announcements(self):
        """Send epoch start announcements."""
        try:
            logger.info("Sending epoch start announcements...")

            current_ts = get_current_timestamp()
            epoch_info = get_current_epoch_info(current_ts)
            epoch_number = epoch_info['epoch_number']

            # Get users to notify
            users = self.notification_engine.get_users_to_notify_epoch_start()

            # Get top pools
            top_pools = self.notification_engine.get_top_pools(limit=3)

            close_date = format_datetime_short(epoch_info['vote_end_ts'])
            voting_duration = epoch_info['voting_time_remaining_formatted']

            for subscriber in users:
                try:
                    message = self.templates.notification_epoch_start(
                        epoch_number=epoch_number,
                        close_date=close_date,
                        voting_duration=voting_duration,
                        top_pools=top_pools
                    )

                    await self.application.bot.send_message(
                        chat_id=subscriber.chat_id,
                        text=message,
                        parse_mode=ParseMode.MARKDOWN_V2
                    )

                    self.notification_engine.log_notification_sent(
                        subscriber.chat_id, 'epoch_start', epoch_number
                    )

                    logger.info(f"Sent epoch start to {subscriber.chat_id}")
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error sending epoch start to {subscriber.chat_id}: {e}")

            logger.info(f"Epoch start announcements complete: {len(users)} sent")

        except Exception as e:
            logger.error(f"Error in _send_epoch_start_announcements: {e}")

    async def _check_high_apr_alerts(self):
        """Check and send high APR alerts."""
        try:
            # Get high APR pools
            default_threshold = self.config.get('telegram.high_apr_threshold_default', 50.0)
            high_apr_pools = self.notification_engine.get_high_apr_pools(threshold=default_threshold)

            if not high_apr_pools:
                return

            logger.info(f"Found {len(high_apr_pools)} high APR pools")

            # For each high APR pool, notify eligible subscribers
            for pool in high_apr_pools:
                try:
                    pool_name = pool.get('pool_name', 'Unknown')
                    apr = pool.get('apr_total', 0)
                    bribes_usd = pool.get('bribes_usd', 0)
                    current_votes = pool.get('current_votes', 0)

                    # Get users to notify for this pool
                    users = self.notification_engine.get_users_to_notify_high_apr(pool)

                    # Note: This will send one alert per high APR pool per check
                    # Consider adding additional deduplication logic if needed

                    for subscriber in users:
                        try:
                            message = self.templates.notification_high_apr(
                                pool_name=pool_name,
                                apr=apr,
                                bribes_usd=bribes_usd,
                                current_votes=current_votes
                            )

                            await self.application.bot.send_message(
                                chat_id=subscriber.chat_id,
                                text=message,
                                parse_mode=ParseMode.MARKDOWN_V2
                            )

                            logger.info(f"Sent high APR alert for {pool_name} to {subscriber.chat_id}")
                            await asyncio.sleep(0.1)

                        except Exception as e:
                            logger.error(f"Error sending high APR alert to {subscriber.chat_id}: {e}")

                except Exception as e:
                    logger.error(f"Error processing high APR pool: {e}")

        except Exception as e:
            logger.error(f"Error in _check_high_apr_alerts: {e}")

    async def _cleanup_old_logs(self):
        """Cleanup old notification logs."""
        try:
            logger.info("Cleaning up old notification logs...")
            deleted = self.notification_engine.cleanup_old_logs()
            logger.info(f"Cleaned up {deleted} old notification logs")
        except Exception as e:
            logger.error(f"Error cleaning up logs: {e}")

    def run(self):
        """Run the bot."""
        try:
            logger.info("Starting VeBTC Telegram Bot...")
            logger.info("Bot is running. Press Ctrl-C to stop.")

            # Start the bot with polling (scheduler will start via post_init)
            self.application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )

        except KeyboardInterrupt:
            logger.info("Received stop signal")
        except Exception as e:
            logger.error(f"Error running bot: {e}", exc_info=True)
        finally:
            logger.info("Bot stopped")


def main():
    """Main entry point."""
    try:
        bot = VeBTCBot()
        bot.run()
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
