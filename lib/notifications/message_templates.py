"""Message templates for Telegram bot."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from lib.analytics.incentives import format_apr, format_usd


class MessageTemplates:
    """Message formatting and templates."""

    @staticmethod
    def welcome_message() -> str:
        """Welcome message for /start command."""
        return """*Welcome to the veBTC Voting Bot\\!* 🎉

You're now subscribed to:
✅ Voting window reminders
✅ Epoch announcements
✅ High APR pool alerts

*Optional:* Link your wallet to receive personalized notifications
Use /link \\<address\\> to get started

Commands: /help"""

    @staticmethod
    def help_message() -> str:
        """Help message with all commands."""
        return """*veBTC Voting Bot Commands*

*Subscription*
/start \\- Subscribe to notifications
/subscribe \\- Re\\-subscribe if unsubscribed
/unsubscribe \\- Stop all notifications

*Wallet Linking*
/link \\<address\\> \\- Link your wallet address
/unlink \\- Remove wallet linking

*Information*
/status \\- Current epoch \\& voting status
/myvotes \\- Your voting history \\(requires link\\)
/pools \\- Top pools by APR
/settings \\- Configure notification preferences"""

    @staticmethod
    def already_subscribed_message() -> str:
        """Message when user is already subscribed."""
        return """You're already subscribed\\! ✅

Use /settings to configure your notification preferences
Use /help to see all available commands"""

    @staticmethod
    def subscribed_message() -> str:
        """Message when user successfully subscribes."""
        return """*Successfully subscribed\\!* ✅

You'll now receive:
• Voting window reminders
• Epoch announcements
• High APR pool alerts

Use /link \\<address\\> for personalized notifications"""

    @staticmethod
    def unsubscribed_message() -> str:
        """Message when user unsubscribes."""
        return """You've been unsubscribed\\.

We're sad to see you go\\! 😢

Use /start anytime to re\\-subscribe"""

    @staticmethod
    def wallet_linked_message(address: str) -> str:
        """Message when wallet is linked."""
        # No escaping needed inside code blocks
        addr_short = f"{address[:6]}...{address[-4:]}"
        return f"""*Wallet Linked Successfully\\!* ✅

Address: `{addr_short}`

You'll now receive personalized notifications based on your voting activity\\."""

    @staticmethod
    def wallet_unlinked_message() -> str:
        """Message when wallet is unlinked."""
        return """*Wallet Unlinked* ✅

You'll continue to receive broadcast notifications, but won't get personalized voting reminders\\.

Use /link \\<address\\> anytime to re\\-link"""

    @staticmethod
    def invalid_wallet_message() -> str:
        """Message for invalid wallet address."""
        return """*Invalid wallet address* ❌

Please provide a valid Ethereum address:
/link 0x1234\\.\\.\\.5678

Example:
/link 0x742d35Cc6634C0532925a3b844Bc9e7595f96f52"""

    @staticmethod
    def wallet_required_message() -> str:
        """Message when command requires linked wallet."""
        return """*Wallet Required* 🔗

This command requires a linked wallet address\\.

Link your wallet with:
/link \\<your\\_address\\>"""

    @staticmethod
    def status_message(
        epoch_number: int,
        start_date: str,
        time_remaining: str,
        voting_status: str,
        voting_time_remaining: str,
        total_voted: float,
        user_status: Optional[str] = None
    ) -> str:
        """Current epoch status message."""
        # Escape special characters
        start_date = start_date.replace('.', '\\.')
        time_remaining = time_remaining.replace('.', '\\.')
        voting_time_remaining = voting_time_remaining.replace('.', '\\.')

        user_line = f"\n*Your Status:* {user_status}" if user_status else ""

        # Format number with escaped period
        total_voted_str = f"{total_voted:.2f}".replace('.', '\\.')

        return f"""*Current Epoch Status* 📊

*Epoch:* {epoch_number}
*Started:* {start_date}
*Time Remaining:* {time_remaining}

*Voting Status:* {voting_status}
*Voting Closes In:* {voting_time_remaining}

*Total Voted:* {total_voted_str} veBTC{user_line}

View top pools: /pools"""

    @staticmethod
    def myvotes_message(
        wallet_address: str,
        voting_power: float,
        current_epoch_votes: List[Dict[str, Any]],
        epoch_number: int,
        vote_date: Optional[str] = None
    ) -> str:
        """User's voting history message."""
        addr_short = f"{wallet_address[:6]}\\.\\.\\.{wallet_address[-4:]}"

        # Format current epoch votes
        if current_epoch_votes:
            # Escape periods in date
            vote_date_escaped = vote_date.replace('.', '\\.') if vote_date else ''
            votes_text = f"*Epoch {epoch_number} \\(Current\\):*\n✅ Voted on {vote_date_escaped}\n"
            for vote in current_epoch_votes[:5]:  # Show top 5
                pool_name = vote.get('pool_name', 'Unknown Pool')
                vp = vote.get('voting_power', 0)
                vp_str = f"{vp:.2f}".replace('.', '\\.')
                votes_text += f"  • {pool_name}: {vp_str} veBTC\n"
        else:
            votes_text = f"*Epoch {epoch_number} \\(Current\\):*\n❌ Not voted yet\n"

        # Format voting power with escaped period
        vp_str = f"{voting_power:.2f}".replace('.', '\\.')

        return f"""*Your Voting History* 📊

*Wallet:* `{addr_short}`
*Voting Power:* {vp_str} veBTC

{votes_text}

View all pools: /pools"""

    @staticmethod
    def pools_message(pools: List[Dict[str, Any]], epoch_number: int) -> str:
        """Top pools by APR message."""
        if not pools:
            return """*No Pool Data Available* 📊

Pool incentive data is currently unavailable\\. Please try again later\\."""

        pools_text = ""
        for i, pool in enumerate(pools[:5], 1):  # Top 5
            name = pool.get('pool_name', 'Unknown')
            apr = pool.get('apr_total', 0)
            votes = pool.get('current_votes', 0)
            bribes = pool.get('bribes_usd', 0)
            usd_per_vote = pool.get('usd_per_vote', 0)

            # Escape all periods in formatted strings
            apr_str = format_apr(apr).replace('.', '\\.')
            votes_str = f"{votes:.2f}".replace('.', '\\.')
            bribes_str = format_usd(bribes).replace('.', '\\.')
            usd_per_vote_str = f"{usd_per_vote:.2f}".replace('.', '\\.')

            pools_text += f"""
*{i}\\. {name}*
APR: {apr_str} \\| Votes: {votes_str} veBTC
Bribes: {bribes_str} \\| $/vote: ${usd_per_vote_str}
"""

        return f"""*Top Pools by APR* 🏆
*Epoch {epoch_number}*

{pools_text}

Vote now: /status"""

    @staticmethod
    def settings_message(subscriber: Any) -> str:
        """Settings message showing current preferences."""
        wallet_display = 'Not linked'
        if subscriber.wallet_address:
            wallet_display = f"`{subscriber.wallet_address[:6]}\\.\\.\\.{subscriber.wallet_address[-4:]}`"

        threshold_str = f"{subscriber.high_apr_threshold}".replace('.', '\\.')

        return f"""*Notification Settings* ⚙️

*Current Preferences:*
Voting Reminders \\(24h\\): {'✅' if subscriber.notify_24h_before else '❌'}
Final Warning: {'✅' if subscriber.notify_final_warning else '❌'}
Epoch Start: {'✅' if subscriber.notify_epoch_start else '❌'}
High APR Alerts: {'✅' if subscriber.notify_high_apr else '❌'}

*Wallet:* {wallet_display}
*High APR Threshold:* {threshold_str}%

To update settings, contact support or use individual commands"""

    @staticmethod
    def notification_24h_reminder_broadcast(
        epoch_number: int,
        close_time: str,
        top_pools: List[Dict[str, Any]]
    ) -> str:
        """24h voting reminder \\(broadcast\\)."""
        pools_text = ""
        for pool in top_pools[:3]:
            name = pool.get('pool_name', 'Unknown')
            apr = pool.get('apr_total', 0)
            apr_str = format_apr(apr).replace('.', '\\.')
            pools_text += f"• {name} \\({apr_str} APR\\)\n"

        close_time_escaped = close_time.replace('.', '\\.')

        return f"""*Voting Reminder* ⏰

The voting window for *Epoch {epoch_number}* closes in *24 hours\\!*

*Vote before:* {close_time_escaped} UTC

*Top Pools:*
{pools_text}

View all: /pools"""

    @staticmethod
    def notification_24h_reminder_personalized(
        username: str,
        epoch_number: int,
        close_time: str,
        voting_power: float,
        top_pools: List[Dict[str, Any]],
        has_voted: bool
    ) -> str:
        """24h voting reminder \\(personalized\\)."""
        if has_voted:
            return f"""*You're all set\\!* ✅

You already voted in *Epoch {epoch_number}*\\.

Next epoch starts soon\\!
View your votes: /myvotes"""

        pools_text = ""
        for pool in top_pools[:3]:
            name = pool.get('pool_name', 'Unknown')
            apr = pool.get('apr_total', 0)
            apr_str = format_apr(apr).replace('.', '\\.')
            pools_text += f"• {name} \\({apr_str} APR\\)\n"

        vp_str = f"{voting_power:.2f}".replace('.', '\\.')

        return f"""*Voting Reminder* ⏰

Hi @{username}\\! You haven't voted yet in *Epoch {epoch_number}*\\.

*Voting closes in:* 24 hours
*Your voting power:* {vp_str} veBTC

*Top Pools:*
{pools_text}

Don't miss out\\! 🗳️"""

    @staticmethod
    def notification_final_warning(
        username: str,
        epoch_number: int,
        close_time: str,
        voting_power: float
    ) -> str:
        """Final warning notification \\(2\\-4h before close\\)."""
        close_time_escaped = close_time.replace('.', '\\.')
        vp_str = f"{voting_power:.2f}".replace('.', '\\.')

        return f"""*FINAL CALL \\- 3 Hours Left\\!* 🚨

Hi @{username}\\! *Epoch {epoch_number}* voting closes at *{close_time_escaped} UTC*\\.

You have *{vp_str} veBTC* ready to vote\\!

Top pools: /pools
Vote now\\!"""

    @staticmethod
    def notification_epoch_start(
        epoch_number: int,
        close_date: str,
        voting_duration: str,
        top_pools: List[Dict[str, Any]]
    ) -> str:
        """Epoch start announcement."""
        pools_text = ""
        for pool in top_pools[:3]:
            name = pool.get('pool_name', 'Unknown')
            apr = pool.get('apr_total', 0)
            apr_str = format_apr(apr).replace('.', '\\.')
            pools_text += f"• {name} \\({apr_str} APR\\)\n"

        close_date_escaped = close_date.replace('.', '\\.')

        return f"""*New Epoch Started\\!* 🎉

*Epoch {epoch_number}* has begun\\!

*Voting Window:*
Opens: Now
Closes: {close_date_escaped} \\({voting_duration}\\)

*Top Pools:*
{pools_text}

View all: /pools"""

    @staticmethod
    def notification_high_apr(
        pool_name: str,
        apr: float,
        bribes_usd: float,
        current_votes: float
    ) -> str:
        """High APR pool alert."""
        apr_str = format_apr(apr).replace('.', '\\.')
        bribes_str = format_usd(bribes_usd).replace('.', '\\.')
        votes_str = f"{current_votes:.2f}".replace('.', '\\.')

        return f"""*High APR Alert\\!* 🔥

New pool with exceptional returns:

*Pool:* {pool_name}
*APR:* {apr_str} 🚀
*Bribes:* {bribes_str}
*Current Votes:* {votes_str} veBTC

This could be a great opportunity\\!
View details: /pools"""

    @staticmethod
    def error_message(error: str = "An error occurred") -> str:
        """Generic error message."""
        return f"""*Error* ❌

{error}

Please try again or contact support\\."""
