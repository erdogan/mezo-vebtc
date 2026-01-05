# Deploy Telegram Bot in 5 Minutes (FREE)

## Step 1: Push to GitHub

```bash
git add .
git commit -m "Add Telegram bot with Render deployment"
git push origin main
```

Make sure your repo is **public**.

## Step 2: Get Bot Token

1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` or use existing bot
3. Copy your bot token (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

## Step 3: Deploy to Render

1. **Sign up:** https://render.com (no credit card needed)

2. **Create service:**
   - Click "New +" → "Background Worker"
   - Connect GitHub
   - Select your `mezo-vebtc` repo

3. **Configure:**
   ```
   Name: vebtc-bot
   Build Command: pip install -r requirements.txt
   Start Command: python3 telegram_bot.py
   Plan: Free
   ```

4. **Add environment variables:**

   Click "Advanced" → "Add Environment Variable":

   ```
   TELEGRAM_BOT_TOKEN = your_token_from_step_2
   GITHUB_DATA_URL = https://raw.githubusercontent.com/YOUR_USERNAME/mezo-vebtc/main/vebtc_data.json
   ```

   Replace `YOUR_USERNAME` with your GitHub username.

5. **Click "Create Background Worker"**

## Step 4: Wait & Test

1. Wait 2-3 minutes for deployment
2. Check logs in Render dashboard for "Bot started successfully"
3. Open Telegram and message your bot
4. Send `/start` command

**Done!** Your bot is now running 24/7 for free.

## Commands to Try

- `/start` - Subscribe
- `/status` - Check current epoch
- `/pools` - View top pools
- `/link YOUR_WALLET_ADDRESS` - Link your wallet
- `/myvotes` - See your votes (after linking)

## Troubleshooting

**Bot not responding?**
- Check Render logs for errors
- Verify TELEGRAM_BOT_TOKEN is correct
- Make sure repo is public

**"No data" errors?**
- Check GITHUB_DATA_URL is correct
- Test URL in browser (should show JSON)
- Wait 10 minutes for GitHub Actions to run

## Need Help?

See [RENDER_DEPLOY.md](./RENDER_DEPLOY.md) for detailed docs.
