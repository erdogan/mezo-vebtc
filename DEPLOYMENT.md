# veBTC Dashboard & Bot Deployment Guide

Complete guide to deploying the veBTC dashboard and Telegram bot.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Repository                        │
│  ┌──────────────────────┐    ┌─────────────────────────┐   │
│  │  Dashboard Code      │    │   Telegram Bot Code     │   │
│  │  vebtc_dashboard.py  │    │   telegram_bot/bot.py   │   │
│  └──────────────────────┘    └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                   │                           │
                   │ (GitHub Actions)          │ (Push to Railway)
                   │ Every 5 minutes           │
                   ↓                           ↓
        ┌──────────────────────┐    ┌──────────────────────┐
        │   GitHub Pages       │    │   Railway.app        │
        │   erdogan.github.io  │    │   (Bot Server)       │
        │                      │    │                      │
        │  • index.html        │←───│  Reads data every    │
        │  • vebtc_data.json   │    │  2 minutes           │
        └──────────────────────┘    └──────────────────────┘
                   │                           │
                   │ (HTTP)                    │ (Telegram API)
                   ↓                           ↓
        ┌──────────────────────┐    ┌──────────────────────┐
        │   Web Users          │    │  Telegram Users      │
        │   View Dashboard     │    │  Query Bot           │
        └──────────────────────┘    └──────────────────────┘
```

## Part 1: Dashboard Deployment (GitHub Pages)

### Step 1: Enable GitHub Pages

1. **Push code to GitHub**
   ```bash
   cd /Users/engin/Projects/mezo-vebtc
   git add .
   git commit -m "Add GitHub Actions workflow and bot"
   git push origin main
   ```

2. **Enable GitHub Pages**
   - Go to: https://github.com/erdogan/mezo-vebtc/settings/pages
   - Source: "Deploy from a branch"
   - Branch: `gh-pages` (will be created automatically by Actions)
   - Folder: `/ (root)`
   - Click "Save"

### Step 2: Configure GitHub Actions

The workflow is already created at `.github/workflows/update-dashboard.yml`.

**What it does:**
- Runs every 5 minutes
- Executes `vebtc_dashboard.py`
- Generates `index.html` and `vebtc_data.json`
- Deploys to `gh-pages` branch

### Step 3: Trigger First Deployment

**Option A: Automatic (wait 5 minutes)**
The workflow will run automatically.

**Option B: Manual trigger**
1. Go to: https://github.com/erdogan/mezo-vebtc/actions
2. Click "Update veBTC Dashboard"
3. Click "Run workflow" → "Run workflow"

### Step 4: Verify Dashboard

After 2-3 minutes, visit:
- **Dashboard**: https://erdogan.github.io/mezo-vebtc/
- **Data JSON**: https://erdogan.github.io/mezo-vebtc/vebtc_data.json

## Part 2: Telegram Bot Deployment (Railway)

### Step 1: Create Telegram Bot

1. **Message @BotFather on Telegram**
   ```
   /newbot
   ```

2. **Follow prompts:**
   - Bot name: "veBTC Dashboard Bot" (or your choice)
   - Username: "vebtc_dashboard_bot" (must end in 'bot')

3. **Save the token** (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Step 2: Install Railway CLI

```bash
# macOS
brew install railway

# Or via npm
npm i -g @railway/cli
```

### Step 3: Deploy Bot to Railway

```bash
# Navigate to bot directory
cd /Users/engin/Projects/mezo-vebtc/telegram_bot

# Login to Railway
railway login

# Create new project
railway init

# Set environment variables
railway variables set TELEGRAM_BOT_TOKEN="your_bot_token_from_botfather"
railway variables set DATA_URL="https://erdogan.github.io/mezo-vebtc/vebtc_data.json"

# Deploy
railway up
```

### Step 4: Verify Bot Deployment

```bash
# Check logs
railway logs

# You should see:
# "Starting veBTC Telegram Bot..."
# "Application started successfully"
```

### Step 5: Test Bot

1. Open Telegram
2. Search for your bot username
3. Send `/start`
4. Bot should respond with welcome message

## Monitoring & Maintenance

### GitHub Actions Status

Check workflow status:
- https://github.com/erdogan/mezo-vebtc/actions

**If failing:**
1. Click on the failed run
2. Check error logs
3. Common issues:
   - Missing dependencies in `requirements.txt`
   - RPC endpoint rate limiting
   - Syntax errors in Python code

### Railway Bot Status

```bash
# View logs
railway logs

# Check service status
railway status

# Restart service
railway restart
```

### Data Flow Verification

```bash
# 1. Check if data is being generated
curl -I https://erdogan.github.io/mezo-vebtc/vebtc_data.json
# Should return: HTTP/2 200

# 2. Check data freshness
curl https://erdogan.github.io/mezo-vebtc/vebtc_data.json | jq '.last_updated'
# Should show recent timestamp

# 3. Test bot manually
# Send /stats to bot in Telegram
```

## Cost Breakdown

| Service | Cost | Usage |
|---------|------|-------|
| GitHub Actions | Free | 2,000 minutes/month (enough for ~41k workflow runs) |
| GitHub Pages | Free | 100GB bandwidth, 1GB storage |
| Railway (Free Tier) | $0 | 500 hours/month (~20 days) |
| Railway (Hobby) | $5/month | Unlimited hours |

**Recommended**: Start with free tier, upgrade Railway to Hobby if bot needs 24/7 uptime.

## Updating

### Update Dashboard Code

```bash
# Make changes to vebtc_dashboard.py or lib/ files
git add .
git commit -m "Update dashboard"
git push origin main

# GitHub Actions will automatically redeploy
```

### Update Bot Code

```bash
# Make changes to telegram_bot/bot.py
cd telegram_bot
git add .
git commit -m "Update bot"
git push origin main

# Redeploy to Railway
railway up
```

## Troubleshooting

### Dashboard not updating

**Check:**
1. GitHub Actions status (should be green)
2. Workflow logs for errors
3. GitHub Pages is enabled

**Fix:**
```bash
# Manually trigger workflow
# Go to Actions tab → Run workflow

# Or commit a change
git commit --allow-empty -m "Trigger workflow"
git push
```

### Bot not responding

**Check:**
1. Railway logs: `railway logs`
2. Environment variables are set: `railway variables`
3. Data URL is accessible: `curl https://erdogan.github.io/mezo-vebtc/vebtc_data.json`

**Fix:**
```bash
# Restart bot
railway restart

# Check logs
railway logs --tail 100
```

### "Unable to fetch data" in bot

**Causes:**
- GitHub Actions workflow failed
- GitHub Pages not deployed yet
- `vebtc_data.json` not generated

**Fix:**
1. Check GitHub Actions: https://github.com/erdogan/mezo-vebtc/actions
2. Verify JSON exists: https://erdogan.github.io/mezo-vebtc/vebtc_data.json
3. Manually trigger workflow if needed

## Security Notes

### Secrets Management

**GitHub:**
- No secrets needed (uses `GITHUB_TOKEN` automatically)
- RPC endpoint is public (Mezo mainnet)

**Railway:**
- Store `TELEGRAM_BOT_TOKEN` as environment variable
- Never commit `.env` files

### Rate Limiting

**GitHub Actions:**
- Runs every 5 minutes = 288 runs/day
- Well within free tier (2,000 minutes/month)

**Mezo RPC:**
- Rate limit: Unknown (likely 100+ req/min)
- Bot caches data for 2 minutes
- Dashboard caches token prices for 5 minutes

**GitHub Pages:**
- Bandwidth: 100GB/month
- Soft limit: ~1GB/day
- Bot fetches ~1MB every 2 min = ~720MB/day (within limits)

## Advanced Configuration

### Change Update Frequency

Edit `.github/workflows/update-dashboard.yml`:

```yaml
schedule:
  # Every 1 minute (max)
  - cron: '* * * * *'

  # Every 10 minutes
  - cron: '*/10 * * * *'

  # Every hour
  - cron: '0 * * * *'
```

### Add Notification on Workflow Failure

Add to workflow:

```yaml
- name: Notify on failure
  if: failure()
  run: |
    curl -X POST https://api.telegram.org/bot${{ secrets.TELEGRAM_BOT_TOKEN }}/sendMessage \
      -d chat_id=${{ secrets.TELEGRAM_CHAT_ID }} \
      -d text="⚠️ Dashboard update failed!"
```

### Use Custom Domain

1. **Add CNAME record:**
   ```
   dashboard.yourdomain.com -> erdogan.github.io
   ```

2. **Update GitHub Pages settings:**
   - Custom domain: `dashboard.yourdomain.com`
   - Enforce HTTPS: ✓

3. **Update bot DATA_URL:**
   ```bash
   railway variables set DATA_URL="https://dashboard.yourdomain.com/vebtc_data.json"
   ```

## Support

**Issues:**
- GitHub: https://github.com/erdogan/mezo-vebtc/issues

**Telegram:**
- Message your deployed bot with `/help`

**Logs:**
- GitHub Actions: https://github.com/erdogan/mezo-vebtc/actions
- Railway: `railway logs`
