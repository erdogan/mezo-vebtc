# Deploy Telegram Bot to Railway.app (FREE)

Railway.app offers $5 free credits per month, which is enough to run the bot 24/7.

## Prerequisites

1. GitHub account with your repo pushed
2. Telegram Bot Token (from @BotFather)
3. Your GitHub repo must be public OR you need Railway Pro ($5/month for private repos)

## Deployment Steps

### 1. Prepare Environment Variable

Your bot needs to fetch data from GitHub since Railway doesn't have persistent storage.

Get your GitHub raw data URL (replace with your username/repo):
```
https://raw.githubusercontent.com/YOUR_USERNAME/mezo-vebtc/main/vebtc_data.json
```

### 2. Deploy to Railway

1. Go to https://railway.app and sign up (free)

2. Create new project:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Authenticate with GitHub
   - Select your `mezo-vebtc` repository
   - Click "Deploy Now"

3. Add environment variables:
   - Click on your service
   - Go to "Variables" tab
   - Add these variables:
     ```
     TELEGRAM_BOT_TOKEN=your_bot_token_here
     GITHUB_DATA_URL=https://raw.githubusercontent.com/YOUR_USERNAME/mezo-vebtc/main/vebtc_data.json
     ```

4. Railway will automatically:
   - Install dependencies from `requirements.txt`
   - Run the bot using `Procfile`
   - Keep it running 24/7

### 3. Verify Deployment

1. Check logs in Railway dashboard:
   - Go to "Deployments" tab
   - Click latest deployment
   - View logs to ensure bot started successfully

2. Test your bot:
   - Open Telegram
   - Send `/start` to your bot
   - Try other commands

### 4. Monitor Usage

Railway gives you $5 free credits/month:
- View usage in dashboard
- Bot should use minimal resources (~$1-2/month)
- You'll get email if you approach limit

## Troubleshooting

### Bot not responding
- Check Railway logs for errors
- Verify `TELEGRAM_BOT_TOKEN` is correct
- Check `GITHUB_DATA_URL` is accessible

### "No data available" errors
- Verify `vebtc_data.json` exists in your GitHub repo
- Check GitHub Actions is running (updates every 10 min)
- Ensure repo is public OR make data URL accessible

### Database issues
Railway uses ephemeral storage, so:
- Subscriber data persists during deployment
- Data is reset if you redeploy
- For production, consider Railway's persistent volumes ($)

## Cost Breakdown

**Free Tier ($5/month credits):**
- Telegram bot: ~$1-2/month
- Lightweight service, minimal compute
- Should stay within free tier

**If you exceed:**
- Railway charges $0.000231/GB-minute
- You can set spending limits
- Bot will pause if credits run out

## Alternative: Render.com

If Railway doesn't work, try Render.com (also free):

1. Go to https://render.com
2. Create "Background Worker"
3. Connect GitHub repo
4. Set start command: `python3 telegram_bot.py`
5. Add same environment variables

## Data Sync

Your bot fetches data from GitHub every time it needs it:
- GitHub Actions updates `vebtc_data.json` every 10 min
- Bot fetches latest data on each command
- No manual sync needed!

## Support

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Check bot logs in Railway dashboard for errors
