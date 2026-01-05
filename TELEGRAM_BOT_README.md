# veBTC Telegram Notification Bot

A full-featured Telegram bot for the veBTC voting system that provides personalized notifications, voting reminders, and pool incentive alerts.

## Features

- **Personalized Notifications**: Optional wallet linking for customized voting reminders
- **Voting Reminders**: 24h advance notice and final warnings before voting closes
- **Epoch Announcements**: Notifications when new epochs begin
- **High APR Alerts**: Notifications when pools exceed APR thresholds
- **Interactive Commands**: Query epoch status, voting history, and pool APRs
- **Privacy-Friendly**: Users can receive broadcast messages without linking wallets

## Commands

| Command | Description |
|---------|-------------|
| `/start` | Subscribe to notifications |
| `/subscribe` | Re-subscribe if unsubscribed |
| `/unsubscribe` | Stop all notifications |
| `/link <address>` | Link your wallet address for personalized notifications |
| `/unlink` | Remove wallet linking |
| `/status` | View current epoch and voting status |
| `/myvotes` | View your voting history (requires linked wallet) |
| `/pools` | View top pools by APR |
| `/settings` | View your notification preferences |
| `/help` | Show all available commands |

## Requirements

- Python 3.8+
- Telegram Bot Token (from @BotFather)
- Deployment platform (Render.com FREE tier recommended)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Bot Token

Create a `.env` file in the project root:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

Get your bot token from [@BotFather](https://t.me/botfather) on Telegram.

### 3. Verify Configuration

The bot reads from `config.json`. The telegram section should look like:

```json
{
  "telegram": {
    "bot_token": "${TELEGRAM_BOT_TOKEN}",
    "admin_chat_ids": [],
    "notification_check_interval": 300,
    "high_apr_threshold_default": 50.0,
    "rate_limit": {
      "messages_per_second": 30,
      "messages_per_minute": 20
    }
  }
}
```

## Running the Bot

### Locally (for testing)

```bash
python3 telegram_bot.py
```

Press Ctrl-C to stop.

### Production Deployment (FREE)

**Recommended: Render.com**

Deploy for free in 5 minutes: See [RENDER_DEPLOY.md](./RENDER_DEPLOY.md)

- ✅ Free tier (750 hours/month)
- ✅ No credit card required
- ✅ Auto-deploy from GitHub
- ✅ Easy setup

**Alternative Options:**
- VPS deployment: See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- Railway.app: See [RAILWAY_DEPLOY.md](./RAILWAY_DEPLOY.md)

## Architecture

```
telegram_bot.py              # Main entry point
├── lib/notifications/
│   ├── subscriber_manager.py   # Database operations
│   ├── notification_engine.py  # Notification logic
│   ├── bot_commands.py         # Command handlers
│   └── message_templates.py    # Message formatting
└── subscribers.db              # SQLite database (auto-created)
```

## Database

The bot uses SQLite (`subscribers.db`) to store:
- Subscriber information (chat_id, username)
- Wallet linkings
- Notification preferences
- Notification history (for deduplication)

The database is created automatically on first run.

## Notifications

### 24-Hour Reminder
- Sent 24 hours before voting closes
- Broadcast: All subscribers receive pool APRs and voting deadline
- Personalized: Linked users get customized messages based on voting status

### Final Warning
- Sent 2-4 hours before voting closes
- Only sent to linked users who haven't voted yet

### Epoch Start Announcement
- Sent 1 hour after new epoch begins (when voting opens)
- Broadcast to all subscribers

### High APR Alerts
- Sent when pools exceed user's APR threshold (default: 50%)
- Configurable per user

## Logs

Logs are written to:
- Console (stdout)
- `telegram_bot.log` file

Log rotation is recommended for production.

## Troubleshooting

### Bot doesn't respond
- Check that `TELEGRAM_BOT_TOKEN` is set correctly
- Verify bot is running: `ps aux | grep telegram_bot`
- Check logs: `tail -f telegram_bot.log`

### Notifications not sending
- Verify `vebtc_data.json` is being updated (GitHub Actions)
- Check notification check interval in config (default: 5 minutes)
- Review logs for errors

### Database errors
- Ensure `subscribers.db` file has write permissions
- Try deleting `subscribers.db` to recreate (subscribers will need to re-subscribe)

## Security

- **Never commit** `.env` or bot tokens to version control
- Bot token is loaded from environment variables only
- Wallet addresses are validated using Web3.py
- User data is stored locally in SQLite (not shared)

## Development

### Adding New Commands

1. Add handler method to `lib/notifications/bot_commands.py`
2. Register handler in `telegram_bot.py` `_register_handlers()`
3. Add message template to `lib/notifications/message_templates.py`

### Adding New Notifications

1. Add notification check logic to `notification_engine.py`
2. Add message template to `message_templates.py`
3. Add notification job to `telegram_bot.py` `_check_notifications()`

## Support

For issues or questions:
- Check logs: `telegram_bot.log`
- Review documentation: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- GitHub Issues: [Report a bug](https://github.com/your-username/mezo-vebtc/issues)

## License

MIT License - See LICENSE file for details
