# Deployment Guide

## Telegram Bot Deployment on Railway

The Telegram bot is deployed on Railway.app which provides $5/month free credit.

### Prerequisites

1. Create a Telegram bot via [@BotFather](https://t.me/botfather)
2. Save your bot token securely (never commit it to git)
3. Create a Railway account at https://railway.app

### Deployment Steps

1. **Fork this repository** to your GitHub account

2. **Create Railway Project:**
   - Go to https://railway.app
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your forked repository

3. **Configure Environment Variables:**
   - In Railway dashboard, go to your service
   - Click "Variables" tab
   - Add the following:
     ```
     TELEGRAM_BOT_TOKEN=<your_bot_token_from_botfather>
     GITHUB_DATA_URL=https://raw.githubusercontent.com/<your_username>/mezo-vebtc/main/vebtc_data.json
     ```

4. **Verify Deployment:**
   - Railway will automatically detect `Procfile` and deploy
   - Check logs to ensure bot started successfully
   - Test bot in Telegram with `/start` command

### Data Updates

The GitHub Actions workflow automatically updates `vebtc_data.json` every 10 minutes.
The bot fetches this file from GitHub to get the latest veBTC data.

### Bot Commands

- `/start` - Subscribe to notifications
- `/link <address>` - Link your wallet
- `/status` - Check current epoch
- `/pools` - View top pools by APR
- `/help` - Show all commands

## Security Notes

**IMPORTANT:**
- Never commit your bot token to git
- Always use environment variables for secrets
- Keep `.env` file in `.gitignore`
- If your token is exposed, revoke it via @BotFather and create a new one

## Troubleshooting

**Bot not responding:**
- Check Railway logs for errors
- Verify `TELEGRAM_BOT_TOKEN` is set correctly
- Ensure `GITHUB_DATA_URL` is accessible

**Data not updating:**
- Check GitHub Actions workflows are running
- Verify workflow has permissions to push to main branch

## Costs

Railway provides $5 free credit per month, which is sufficient for this bot.
