"""Telegram bot command handlers."""

import logging
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
from web3 import Web3

from lib.utils.time_utils import get_current_timestamp, format_datetime_short
from lib.utils.mezo_username import resolve_address, resolve_username
from lib.analytics.epoch_tracker import get_current_epoch_info
from .subscriber_manager import SubscriberManager
from .notification_engine import NotificationEngine
from .message_templates import MessageTemplates
from .bot_analytics import BotAnalytics

logger = logging.getLogger(__name__)


class BotCommands:
    """Telegram bot command handlers."""

    def __init__(self, subscriber_manager: SubscriberManager, notification_engine: NotificationEngine):
        """Initialize command handlers.

        Args:
            subscriber_manager: SubscriberManager instance
            notification_engine: NotificationEngine instance
        """
        self.subscriber_manager = subscriber_manager
        self.notification_engine = notification_engine
        self.templates = MessageTemplates()
        self.analytics = BotAnalytics(subscriber_manager)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        try:
            chat_id = update.effective_chat.id
            username = update.effective_user.username

            # Check if already subscribed
            subscriber = self.subscriber_manager.get_subscriber(chat_id)

            if subscriber:
                message = self.templates.already_subscribed_message()
            else:
                # Add new subscriber
                success = self.subscriber_manager.add_subscriber(chat_id, username)

                if success:
                    message = self.templates.welcome_message()
                    logger.info(f"New subscriber: {chat_id} (@{username})")
                else:
                    message = self.templates.error_message("Failed to subscribe. Please try again.")

            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN_V2
            )

        except Exception as e:
            logger.error(f"Error in start_command: {e}")
            await update.message.reply_text(
                self.templates.error_message(),
                parse_mode=ParseMode.MARKDOWN_V2
            )

    async def subscribe_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /subscribe command."""
        try:
            chat_id = update.effective_chat.id
            username = update.effective_user.username

            # Add or update subscriber
            success = self.subscriber_manager.add_subscriber(chat_id, username)

            if success:
                message = self.templates.subscribed_message()
                logger.info(f"Subscribed: {chat_id} (@{username})")
            else:
                message = self.templates.error_message("Failed to subscribe. Please try again.")

            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN_V2
            )

        except Exception as e:
            logger.error(f"Error in subscribe_command: {e}")
            await update.message.reply_text(
                self.templates.error_message(),
                parse_mode=ParseMode.MARKDOWN_V2
            )

    async def unsubscribe_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /unsubscribe command."""
        try:
            chat_id = update.effective_chat.id

            success = self.subscriber_manager.remove_subscriber(chat_id)

            if success:
                message = self.templates.unsubscribed_message()
                logger.info(f"Unsubscribed: {chat_id}")
            else:
                message = self.templates.error_message("Failed to unsubscribe. Please try again.")

            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN_V2
            )

        except Exception as e:
            logger.error(f"Error in unsubscribe_command: {e}")
            await update.message.reply_text(
                self.templates.error_message(),
                parse_mode=ParseMode.MARKDOWN_V2
            )

    async def link_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /link <address or username> command."""
        try:
            chat_id = update.effective_chat.id

            # Check if subscriber exists
            subscriber = self.subscriber_manager.get_subscriber(chat_id)
            if not subscriber:
                await update.message.reply_text(
                    "Please use /start first to subscribe.",
                    parse_mode=ParseMode.MARKDOWN_V2
                )
                return

            # Get wallet address or username from command
            if not context.args or len(context.args) == 0:
                await update.message.reply_text(
                    self.templates.invalid_wallet_message(),
                    parse_mode=ParseMode.MARKDOWN_V2
                )
                return

            input_value = context.args[0]
            wallet_address = None
            mezo_id = None

            # Check if input is an address or username
            if input_value.startswith('0x'):
                # Input is a wallet address
                if not Web3.is_address(input_value):
                    await update.message.reply_text(
                        self.templates.invalid_wallet_message(),
                        parse_mode=ParseMode.MARKDOWN_V2
                    )
                    return
                wallet_address = input_value
                # Try to resolve username for display
                mezo_id = resolve_username(wallet_address)
            else:
                # Input is a username - resolve to address
                wallet_address = resolve_address(input_value)
                if not wallet_address:
                    await update.message.reply_text(
                        self.templates.username_not_found_message(input_value),
                        parse_mode=ParseMode.MARKDOWN_V2
                    )
                    return
                # Get the full mezoId (with .mezo suffix)
                mezo_id = resolve_username(wallet_address)

            # Link wallet
            success = self.subscriber_manager.link_wallet(chat_id, wallet_address)

            if success:
                message = self.templates.wallet_linked_message(wallet_address, mezo_id)
                logger.info(f"Wallet linked: {chat_id} -> {wallet_address} ({mezo_id or 'no username'})")
            else:
                message = self.templates.error_message("Failed to link wallet. Please try again.")

            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN_V2
            )

        except Exception as e:
            logger.error(f"Error in link_command: {e}")
            await update.message.reply_text(
                self.templates.error_message(),
                parse_mode=ParseMode.MARKDOWN_V2
            )

    async def unlink_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /unlink command."""
        try:
            chat_id = update.effective_chat.id

            success = self.subscriber_manager.unlink_wallet(chat_id)

            if success:
                message = self.templates.wallet_unlinked_message()
                logger.info(f"Wallet unlinked: {chat_id}")
            else:
                message = self.templates.error_message("Failed to unlink wallet. Please try again.")

            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN_V2
            )

        except Exception as e:
            logger.error(f"Error in unlink_command: {e}")
            await update.message.reply_text(
                self.templates.error_message(),
                parse_mode=ParseMode.MARKDOWN_V2
            )

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /epoch command."""
        try:
            chat_id = update.effective_chat.id
            current_ts = get_current_timestamp()
            epoch_info = get_current_epoch_info(current_ts)

            # Get subscriber info
            subscriber = self.subscriber_manager.get_subscriber(chat_id)
            user_status = None

            if subscriber and subscriber.wallet_address:
                has_voted, votes = self.notification_engine.check_if_user_voted(
                    subscriber.wallet_address,
                    epoch_info['epoch_number']
                )
                voting_power = self.notification_engine.get_user_voting_power(subscriber.wallet_address)

                # Escape periods in voting power
                vp_str = f"{voting_power:.2f}".replace('.', '\\.')

                if has_voted:
                    # Calculate total voting weight used
                    total_voted_weight = sum(v.get('voting_power', 0) for v in votes)
                    voted_str = f"{total_voted_weight:.2f}".replace('.', '\\.')
                    num_pools = len(votes)
                    pools_str = f"{num_pools} pool" if num_pools == 1 else f"{num_pools} pools"
                    user_status = f"Voted ✅ {voted_str} veBTC across {pools_str}"
                else:
                    user_status = f"Not voted \\({vp_str} veBTC available\\)"
            else:
                user_status = "Not linked \\(use /link\\)"

            # Get total voted and unique voters in current epoch
            total_voted = self.notification_engine.get_total_voted_in_epoch(epoch_info['epoch_number'])
            unique_voters = self.notification_engine.get_unique_voters_in_epoch(epoch_info['epoch_number'])

            # Format dates
            start_date = format_datetime_short(epoch_info['start_ts'])
            voting_status = "OPEN" if epoch_info['is_voting_open'] else "CLOSED"

            message = self.templates.status_message(
                epoch_number=epoch_info['epoch_number'],
                start_date=start_date,
                time_remaining=epoch_info['time_remaining_formatted'],
                voting_status=voting_status,
                voting_time_remaining=epoch_info['voting_time_remaining_formatted'],
                total_voted=total_voted,
                unique_voters=unique_voters,
                user_status=user_status
            )

            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN_V2
            )

        except Exception as e:
            logger.error(f"Error in status_command: {e}")
            await update.message.reply_text(
                self.templates.error_message(),
                parse_mode=ParseMode.MARKDOWN_V2
            )

    async def myvotes_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /myvotes command."""
        try:
            chat_id = update.effective_chat.id

            # Check if wallet is linked
            subscriber = self.subscriber_manager.get_subscriber(chat_id)
            if not subscriber or not subscriber.wallet_address:
                await update.message.reply_text(
                    self.templates.wallet_required_message(),
                    parse_mode=ParseMode.MARKDOWN_V2
                )
                return

            # Get current epoch
            current_ts = get_current_timestamp()
            epoch_info = get_current_epoch_info(current_ts)
            epoch_number = epoch_info['epoch_number']

            # Get voting power
            voting_power = self.notification_engine.get_user_voting_power(subscriber.wallet_address)

            # Check if voted
            has_voted, votes = self.notification_engine.check_if_user_voted(
                subscriber.wallet_address,
                epoch_number
            )

            # Format vote date if available
            vote_date = None
            if has_voted and votes:
                vote_ts = votes[0].get('ts')
                if vote_ts:
                    # Handle both datetime objects and ISO strings
                    if hasattr(vote_ts, 'timestamp'):
                        timestamp = int(vote_ts.timestamp())
                    elif isinstance(vote_ts, str):
                        # Parse ISO string to datetime then to timestamp
                        from datetime import datetime
                        dt = datetime.fromisoformat(vote_ts.replace('Z', '+00:00'))
                        timestamp = int(dt.timestamp())
                    else:
                        timestamp = int(vote_ts)
                    vote_date = format_datetime_short(timestamp)

            # Aggregate votes by pool and filter out zero-display votes
            from collections import defaultdict
            pool_votes = defaultdict(float)

            for vote in votes:
                voting_power = vote.get('voting_power', 0)
                pool_address = vote.get('pool', '')

                # Resolve pool address to pool name
                pool_name = self.notification_engine.get_pool_name(pool_address) if pool_address else 'Unknown'

                # Aggregate by pool name
                pool_votes[pool_name] += voting_power

            # Format all votes for display (no filtering)
            formatted_votes = [
                {
                    'pool_name': pool_name,
                    'voting_power': total_vp
                }
                for pool_name, total_vp in pool_votes.items()
            ]

            # Sort by voting power descending
            formatted_votes.sort(key=lambda x: x['voting_power'], reverse=True)

            message = self.templates.myvotes_message(
                wallet_address=subscriber.wallet_address,
                voting_power=voting_power,
                current_epoch_votes=formatted_votes,
                epoch_number=epoch_number,
                vote_date=vote_date
            )

            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN_V2
            )

        except Exception as e:
            logger.error(f"Error in myvotes_command: {e}")
            await update.message.reply_text(
                self.templates.error_message(),
                parse_mode=ParseMode.MARKDOWN_V2
            )

    async def pools_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /pools command."""
        try:
            current_ts = get_current_timestamp()
            epoch_info = get_current_epoch_info(current_ts)

            # Get top pools
            top_pools = self.notification_engine.get_top_pools(limit=5)

            message = self.templates.pools_message(
                pools=top_pools,
                epoch_number=epoch_info['epoch_number']
            )

            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN_V2
            )

        except Exception as e:
            logger.error(f"Error in pools_command: {e}")
            await update.message.reply_text(
                self.templates.error_message(),
                parse_mode=ParseMode.MARKDOWN_V2
            )

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /settings command."""
        try:
            chat_id = update.effective_chat.id

            subscriber = self.subscriber_manager.get_subscriber(chat_id)
            if not subscriber:
                await update.message.reply_text(
                    "Please use /start first to subscribe.",
                    parse_mode=ParseMode.MARKDOWN_V2
                )
                return

            message = self.templates.settings_message(subscriber)

            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN_V2
            )

        except Exception as e:
            logger.error(f"Error in settings_command: {e}")
            await update.message.reply_text(
                self.templates.error_message(),
                parse_mode=ParseMode.MARKDOWN_V2
            )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        try:
            message = self.templates.help_message()

            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN_V2,
                disable_web_page_preview=True
            )

        except Exception as e:
            logger.error(f"Error in help_command: {e}")
            await update.message.reply_text(
                self.templates.error_message(),
                parse_mode=ParseMode.MARKDOWN_V2
            )

    async def unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle unknown commands."""
        try:
            await update.message.reply_text(
                self.templates.unknown_command_message(),
                parse_mode=ParseMode.MARKDOWN_V2
            )
        except Exception as e:
            logger.error(f"Error in unknown_command: {e}")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the bot."""
        logger.error(f"Update {update} caused error {context.error}")

        # Try to notify user
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    self.templates.error_message("An unexpected error occurred. Please try again later."),
                    parse_mode=ParseMode.MARKDOWN_V2
                )
            except Exception as e:
                logger.error(f"Failed to send error message to user: {e}")

    async def test_notification_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /test command (hidden, for testing notifications).

        Usage:
            /test 24h - Test 24h reminder
            /test final - Test final warning
            /test epoch - Test epoch start
            /test apr - Test high APR alert
        """
        try:
            chat_id = update.effective_chat.id

            # Check if subscriber exists
            subscriber = self.subscriber_manager.get_subscriber(chat_id)
            if not subscriber:
                await update.message.reply_text(
                    "Please use /start first to subscribe.",
                    parse_mode=ParseMode.MARKDOWN_V2
                )
                return

            # Get test type from args
            if not context.args or len(context.args) == 0:
                await update.message.reply_text(
                    "Usage: /test \\<type\\>\\n\\nTypes: 24h, final, epoch, apr",
                    parse_mode=ParseMode.MARKDOWN_V2
                )
                return

            test_type = context.args[0].lower()
            current_ts = get_current_timestamp()
            epoch_info = get_current_epoch_info(current_ts)
            epoch_number = epoch_info['epoch_number']

            # Get sample data
            top_pools = self.notification_engine.get_top_pools(limit=3)
            close_time = format_datetime_short(epoch_info['vote_end_ts'])

            message = None

            if test_type == "24h":
                # Test 24h reminder
                time_remaining = epoch_info['voting_time_remaining_formatted']
                if subscriber.wallet_address:
                    voting_power = self.notification_engine.get_user_voting_power(subscriber.wallet_address)
                    message = self.templates.notification_24h_reminder_personalized(
                        username=subscriber.username or 'there',
                        epoch_number=epoch_number,
                        close_time=close_time,
                        time_remaining=time_remaining,
                        voting_power=voting_power,
                        top_pools=top_pools,
                        has_voted=False
                    )
                else:
                    message = self.templates.notification_24h_reminder_broadcast(
                        epoch_number=epoch_number,
                        close_time=close_time,
                        time_remaining=time_remaining,
                        top_pools=top_pools
                    )

            elif test_type == "final":
                # Test final warning
                if not subscriber.wallet_address:
                    await update.message.reply_text(
                        "Final warning requires linked wallet\\. Use /link first\\.",
                        parse_mode=ParseMode.MARKDOWN_V2
                    )
                    return

                voting_power = self.notification_engine.get_user_voting_power(subscriber.wallet_address)
                message = self.templates.notification_final_warning(
                    username=subscriber.username or 'there',
                    epoch_number=epoch_number,
                    close_time=close_time,
                    voting_power=voting_power
                )

            elif test_type == "epoch":
                # Test epoch start
                voting_duration = epoch_info['voting_time_remaining_formatted']
                message = self.templates.notification_epoch_start(
                    epoch_number=epoch_number,
                    close_date=close_time,
                    voting_duration=voting_duration,
                    top_pools=top_pools
                )

            elif test_type == "apr":
                # Test high APR alert
                if top_pools:
                    pool = top_pools[0]
                    message = self.templates.notification_high_apr(
                        pool_name=pool.get('pool_name', 'Unknown Pool'),
                        apr=pool.get('apr_total', 0),
                        bribes_usd=pool.get('bribes_usd', 0),
                        current_votes=pool.get('current_votes', 0)
                    )
                else:
                    message = "No pool data available for APR test\\."

            else:
                await update.message.reply_text(
                    f"Unknown test type: {test_type}\\n\\nValid types: 24h, final, epoch, apr",
                    parse_mode=ParseMode.MARKDOWN_V2
                )
                return

            # Send the test notification
            if message:
                await update.message.reply_text(
                    message,
                    parse_mode=ParseMode.MARKDOWN_V2
                )
                logger.info(f"Sent test notification ({test_type}) to {chat_id}")

        except Exception as e:
            logger.error(f"Error in test_notification_command: {e}")
            await update.message.reply_text(
                self.templates.error_message(),
                parse_mode=ParseMode.MARKDOWN_V2
            )

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /botstats command - show bot usage statistics.

        Usage:
            /botstats - Show comprehensive bot statistics
        """
        try:
            chat_id = update.effective_chat.id

            # Get current epoch for epoch-specific stats
            current_ts = get_current_timestamp()
            epoch_info = get_current_epoch_info(current_ts)
            epoch_number = epoch_info['epoch_number']

            # Get comprehensive stats
            stats = self.analytics.get_comprehensive_stats(current_epoch=epoch_number)

            # Format message
            message = self.templates.stats_message(stats, epoch_number)

            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN_V2
            )

            logger.info(f"Stats viewed by {chat_id}")

        except Exception as e:
            logger.error(f"Error in stats_command: {e}")
            await update.message.reply_text(
                self.templates.error_message(),
                parse_mode=ParseMode.MARKDOWN_V2
            )
