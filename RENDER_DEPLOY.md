# Deploy Telegram Bot to Render.com (FREE)

Render.com offers free tier for background workers - perfect for running the bot 24/7.

## Quick Start (5 minutes)

### 1. Push to GitHub

```bash
git push origin main
```

Make sure your repo is **public** (Render free tier requires public repos).

### 2. Get Your Data URL

Your bot needs to fetch data from GitHub. Use this URL format:

```
https://raw.githubusercontent.com/YOUR_USERNAME/mezo-vebtc/main/vebtc_data.json
```

Replace `YOUR_USERNAME` with your GitHub username.

### 3. Deploy to Render

1. **Sign up:** Go to https://render.com (free, no credit card required)

2. **Create Background Worker:**
   - Click "New +" → "Background Worker"
   - Connect your GitHub account
   - Select `mezo-vebtc` repository
   - Click "Connect"

3. **Configure Service:**
   ```
   Name: vebtc-telegram-bot
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: python3 telegram_bot.py
   ```

4. **Add Environment Variables:**

   In the "Environment" section, add:

   | Key | Value |
   |-----|-------|
   | `TELEGRAM_BOT_TOKEN` | `your_bot_token_from_botfather` |
   | `GITHUB_DATA_URL` | `https://raw.githubusercontent.com/YOUR_USERNAME/mezo-vebtc/main/vebtc_data.json` |

5. **Select Free Plan:**
   - Instance Type: Free
   - Click "Create Background Worker"

6. **Wait for Deploy:**
   - Render will install dependencies
   - Start your bot
   - First deploy takes ~2-3 minutes

### 4. Verify It's Working

1. **Check Logs:**
   - Go to your service dashboard
   - Click "Logs" tab
   - Look for: `Bot started successfully`

2. **Test Bot:**
   - Open Telegram
   - Send `/start` to your bot
   - Try `/status`, `/pools`, etc.

## How It Works

```
GitHub Actions (every 10 min)
    ↓
Updates vebtc_data.json
    ↓
Render Bot fetches from GitHub
    ↓
Responds to Telegram users
```

## Free Tier Limits

**Render Free Tier includes:**
- ✅ 750 hours/month (enough for 24/7)
- ✅ Background workers supported
- ✅ Auto-deploys from GitHub
- ✅ No credit card required
- ❌ Service sleeps after 15 min inactivity

**About Sleep:**
- Bot "spins down" after 15 min of no activity
- Wakes up in ~30 seconds when someone messages it
- Scheduled notifications might be delayed
- **Workaround:** Use a cron job to ping bot every 10 min (optional)

## Keep Bot Awake (Optional)

If you need 24/7 uptime for scheduled notifications:

### Option A: Self-Ping with Cron

Add this to your GitHub Actions workflow (`.github/workflows/deploy.yml`):

```yaml
- name: Keep bot awake
  run: |
    # Trigger any bot endpoint to keep it alive
    echo "Ping bot to prevent sleep"
  if: always()
```

### Option B: UptimeRobot (Free)

1. Sign up at https://uptimerobot.com (free)
2. Add HTTP monitor
3. URL: Your Render service URL
4. Interval: 5 minutes
5. This keeps bot awake 24/7

### Option C: Upgrade ($7/month)

Render Starter plan ($7/month):
- No sleep
- Always-on 24/7
- More compute resources

## Troubleshooting

### Bot not responding

Check logs in Render dashboard:
```bash
# Common issues:
- Invalid TELEGRAM_BOT_TOKEN
- GITHUB_DATA_URL not accessible
- Bot sleeping (15 min inactivity)
```

### "Fetch data from GitHub" errors

Verify:
1. `vebtc_data.json` exists in your repo
2. Repo is public
3. GitHub Actions workflow is running
4. URL is correct (no typos)

Test the URL in browser:
```
https://raw.githubusercontent.com/YOUR_USERNAME/mezo-vebtc/main/vebtc_data.json
```

### Database resets on redeploy

Render's free tier has ephemeral disk:
- Subscriber data (`subscribers.db`) persists during runtime
- Resets when you redeploy
- For permanent storage: Use Render Postgres (free tier available)

### Bot stops responding after 15 min

This is normal on free tier. Options:
1. Accept 30-second wake time (most users won't notice)
2. Set up UptimeRobot ping (free, keeps alive)
3. Upgrade to paid plan ($7/month, no sleep)

## Manual Deploy Trigger

To manually redeploy (after pushing changes):

1. Go to Render dashboard
2. Click your service
3. Click "Manual Deploy" → "Clear build cache & deploy"

Or configure auto-deploy:
- Render auto-deploys on every `git push` by default
- No action needed!

## Auto-Deploy on Git Push

Already configured! Every time you push to `main`:
1. Render detects change
2. Rebuilds bot
3. Restarts service
4. Takes ~2-3 minutes

## Monitoring

**Check bot health:**
- Render Dashboard → Logs
- Look for errors or crashes
- Monitor CPU/Memory usage

**Check data updates:**
- View GitHub Actions runs
- Verify `vebtc_data.json` timestamp
- Should update every 10 minutes

## Cost Comparison

| Platform | Free Tier | Limitations | Best For |
|----------|-----------|-------------|----------|
| **Render** | 750 hrs/month | Sleeps after 15 min | Light usage, testing |
| **Railway** | $5 credit/month | Usage-based billing | Consistent uptime |
| **Fly.io** | 3 shared VMs | Complex setup | Advanced users |

## Next Steps

After deploying:

1. **Test all commands:** `/start`, `/link`, `/status`, `/myvotes`, `/pools`
2. **Link your wallet:** `/link YOUR_ADDRESS`
3. **Check notifications work:** Wait for next epoch event
4. **Monitor logs:** Watch for errors in first 24 hours
5. **Set up UptimeRobot:** If you need 24/7 uptime

## Support

- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
- Bot Issues: Check GitHub repo issues

## Alternative: Self-Hosting

If Render's sleep is a problem:

1. **Oracle Cloud:** Always-free VM (no sleep, complex setup)
2. **AWS Free Tier:** 12 months free (requires credit card)
3. **DigitalOcean:** $4/month (always-on, simple)

For now, Render is the easiest free option!
