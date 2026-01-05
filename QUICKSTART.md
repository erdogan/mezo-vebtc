# veBTC Dashboard + Bot - Quick Start

Everything is set up! Follow these steps to deploy.

## 🎯 What You Have Now

```
mezo-vebtc/
├── .github/workflows/
│   └── update-dashboard.yml      ✅ Auto-updates dashboard every 5 min
├── telegram_bot/
│   ├── bot.py                    ✅ Telegram bot code
│   ├── requirements.txt          ✅ Dependencies
│   ├── deploy.sh                 ✅ One-click Railway deployment
│   ├── railway.json              ✅ Railway configuration
│   └── README.md                 ✅ Bot documentation
├── vebtc_dashboard.py            ✅ Dashboard generator
├── DEPLOYMENT.md                 ✅ Complete deployment guide
└── QUICKSTART.md                 ✅ This file
```

## 🚀 Deploy in 3 Steps

### Step 1: Deploy Dashboard (2 minutes)

```bash
# Commit and push
git add .
git commit -m "Add GitHub Actions and Telegram bot"
git push origin main

# Enable GitHub Pages
# 1. Go to: https://github.com/erdogan/mezo-vebtc/settings/pages
# 2. Source: "Deploy from a branch"
# 3. Branch: gh-pages / (root)
# 4. Save

# Trigger first deployment
# Go to: https://github.com/erdogan/mezo-vebtc/actions
# Click: "Update veBTC Dashboard" → "Run workflow"

# Wait 2-3 minutes, then visit:
# https://erdogan.github.io/mezo-vebtc/
```

### Step 2: Create Telegram Bot (1 minute)

```
1. Open Telegram
2. Search for: @BotFather
3. Send: /newbot
4. Name: veBTC Dashboard Bot
5. Username: vebtc_dashboard_bot (or your choice)
6. Save the token (123456789:ABC...)
```

### Step 3: Deploy Bot (2 minutes)

```bash
# Option A: Automated script
cd telegram_bot
./deploy.sh
# Follow the prompts

# Option B: Manual
cd telegram_bot
railway login
railway init
railway variables set TELEGRAM_BOT_TOKEN="your_token"
railway variables set DATA_URL="https://erdogan.github.io/mezo-vebtc/vebtc_data.json"
railway up
```

## ✅ Verify Everything Works

### Dashboard
- Visit: https://erdogan.github.io/mezo-vebtc/
- Should see: Real-time veBTC stats with tabs

### Data API
```bash
curl https://erdogan.github.io/mezo-vebtc/vebtc_data.json | jq '.last_updated'
```
Should show recent timestamp.

### Bot
1. Open Telegram
2. Search for your bot
3. Send: `/start`
4. Should respond with welcome message

## 📊 What Happens Now

1. **GitHub Actions** runs every 5 minutes:
   - Fetches latest locks/votes from Mezo
   - Calculates incentives & APRs
   - Generates HTML dashboard
   - Updates `vebtc_data.json`
   - Deploys to GitHub Pages

2. **Telegram Bot** (running on Railway):
   - Fetches data from GitHub Pages every 2 minutes
   - Responds to user commands instantly
   - Caches data to avoid rate limits

3. **Users** can:
   - View dashboard: https://erdogan.github.io/mezo-vebtc/
   - Query bot: Search for bot on Telegram

## 🎮 Try These Bot Commands

```
/start        - Welcome message
/stats        - Overall veBTC statistics
/incentives   - Pool bribes & APRs
/leaderboard  - Top lockers & voters
/search <addr> - Lookup any participant
```

## 📝 Monitoring

### GitHub Actions
```bash
# Check if workflow is running
open https://github.com/erdogan/mezo-vebtc/actions
```

### Railway Bot
```bash
cd telegram_bot
railway logs         # View logs
railway status       # Check status
railway restart      # Restart if needed
```

### Quick Health Check
```bash
# Dashboard
curl -I https://erdogan.github.io/mezo-vebtc/

# Data JSON
curl https://erdogan.github.io/mezo-vebtc/vebtc_data.json | head -20

# Bot (send /stats in Telegram)
```

## 💰 Costs

| Service | Free Tier | Cost |
|---------|-----------|------|
| GitHub Actions | 2,000 min/month | $0 |
| GitHub Pages | 100GB bandwidth | $0 |
| Railway | 500 hours/month | $0 |
| **Total** | | **$0/month** |

**Note**: Railway free tier = ~20 days uptime. Upgrade to Hobby ($5/mo) for 24/7.

## 🔧 Customization

### Change Update Frequency

Edit `.github/workflows/update-dashboard.yml`:
```yaml
schedule:
  - cron: '*/10 * * * *'  # Every 10 minutes
```

### Add Bot Commands

Edit `telegram_bot/bot.py` and redeploy:
```bash
cd telegram_bot
railway up
```

### Change Dashboard Theme

Edit CSS in `lib/generators/html_*.py` and push:
```bash
git push origin main
# GitHub Actions will auto-update
```

## 🆘 Troubleshooting

### Dashboard not updating
```bash
# Check workflow
open https://github.com/erdogan/mezo-vebtc/actions

# Manually trigger
# Actions tab → Run workflow
```

### Bot not responding
```bash
cd telegram_bot
railway logs --tail 50
railway restart
```

### Data not fresh
```bash
# Check last update
curl https://erdogan.github.io/mezo-vebtc/vebtc_data.json | jq '.last_updated'

# If old, check GitHub Actions
```

## 📚 Full Documentation

- **Complete Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Bot README**: [telegram_bot/README.md](telegram_bot/README.md)
- **Architecture Details**: See DEPLOYMENT.md

## 🎉 You're Done!

Your veBTC dashboard and bot are now live!

**Dashboard**: https://erdogan.github.io/mezo-vebtc/
**Bot**: Search on Telegram

Updates automatically every 5 minutes. No maintenance required.
