# Railway Volume Setup - CRITICAL FIX

## Problem
Your Telegram bot loses all linked wallets and subscribers on every deployment because the SQLite database (`subscribers.db`) is stored on the container filesystem, which gets wiped with each deploy.

## Solution
Configure Railway to use a persistent volume for the database.

## Setup Steps

### 1. Create a Volume in Railway

1. Go to your Railway project: https://railway.app/project/your-project
2. Click on your bot service
3. Go to the **"Volumes"** tab
4. Click **"+ New Volume"**
5. Configure:
   - **Mount Path:** `/data`
   - Click **"Add"**

### 2. Update Environment Variables

1. Still in your bot service, go to the **"Variables"** tab
2. Add a new variable:
   - **Key:** `DATABASE_PATH`
   - **Value:** `/data/subscribers.db`
3. Your variables should now look like:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   GITHUB_DATA_URL=https://raw.githubusercontent.com/erdogan/mezo-vebtc/main/vebtc_data.json
   DATABASE_PATH=/data/subscribers.db
   ```

### 3. Redeploy

Railway will automatically redeploy after you add the volume and variable.

## Verification

After the bot redeploys:

1. **Test in Telegram:**
   - Send `/start` to your bot
   - Send `/link <your_wallet_address>`
   - Verify it works

2. **Make a test deployment:**
   - Push a minor code change to trigger a deploy
   - After deploy, send `/status` in Telegram
   - Your wallet should still be linked!

## Why This Works

- **Before:** Database stored in `/app/subscribers.db` (ephemeral, wiped on deploy)
- **After:** Database stored in `/data/subscribers.db` (persistent volume, survives deploys)

Railway volumes persist data across deployments, restarts, and even if you change regions.

## Important Notes

1. **Free Tier Limit:** Railway free tier includes 1GB of volume storage (plenty for SQLite)
2. **Backup:** The database is automatically backed up by Railway
3. **No Code Changes Needed:** The code now reads `DATABASE_PATH` from environment variables

## If You Already Lost Data

Unfortunately, any subscribers/links before this fix are lost. Users will need to:
1. Send `/start` again to re-subscribe
2. Send `/link <address>` again to re-link their wallet

This is a one-time issue - after this fix, data will persist forever.
