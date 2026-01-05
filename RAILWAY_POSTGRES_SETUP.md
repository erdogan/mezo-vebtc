# Railway PostgreSQL Setup - PERMANENT FIX

## Problem
Your Telegram bot loses all linked wallets on every deployment because SQLite database gets wiped.

## Solution
Use Railway's PostgreSQL database - free, persistent, and reliable.

## Setup Steps (5 minutes)

### 1. Add PostgreSQL to Your Project

1. Go to your Railway project: https://railway.app
2. Click **"+ New"** button (top right)
3. Select **"Database"**
4. Choose **"Add PostgreSQL"**
5. Railway will create a new PostgreSQL database service
6. Wait ~30 seconds for it to provision

### 2. Link Database to Bot Service

Railway automatically creates a `DATABASE_URL` environment variable in **all services** in your project. Your bot will automatically detect and use it - **no manual configuration needed!**

### 3. Verify Setup

1. Go to your **bot service** (not the database)
2. Click **"Variables"** tab
3. You should see `DATABASE_URL` (added automatically by Railway)
4. It will look like: `postgresql://postgres:...@...railway.app:5432/railway`

Your existing variables should be:
- `TELEGRAM_BOT_TOKEN` (your bot token)
- `GITHUB_DATA_URL` (data source)
- `DATABASE_URL` (auto-created by Railway)

### 4. Redeploy

Railway will automatically redeploy your bot. The deployment will:
- Install PostgreSQL driver (`psycopg2-binary`)
- Detect `DATABASE_URL` and use PostgreSQL
- Create tables automatically on first run

## Verification

After the bot redeploys (check logs for "Using PostgreSQL database"):

1. **Test in Telegram:**
   ```
   /start
   /link 0xYourWalletAddress
   /status
   ```

2. **Trigger a test deploy:**
   - Push any minor code change (add a comment somewhere)
   - Wait for redeploy to complete
   - Send `/status` again
   - **Your wallet should still be linked!** ✅

## What Happens Now

- **Before:** SQLite database lost on every deploy
- **After:** PostgreSQL database persists forever
- **Cost:** $0 (included in Railway free tier)
- **Storage:** Up to 1GB on free tier (plenty for subscriber data)

## Migration Notes

Unfortunately, any wallets linked before this fix are already lost. Users will need to re-link **one last time**:
1. `/start` - Subscribe
2. `/link <address>` - Link wallet

After that, their data will persist forever, even through deployments.

## Database Management

### View Your Data

1. In Railway, click on the **PostgreSQL** service
2. Go to **"Data"** tab
3. You can see the `subscribers` table and all linked wallets

### Backup (Optional)

Railway automatically backs up your PostgreSQL database. But you can also:
1. Click on PostgreSQL service
2. Go to **"Settings"**
3. Find connection details
4. Use any PostgreSQL client to export data

## Troubleshooting

### Bot still losing data?

Check logs:
1. Go to bot service
2. Click **"Deployments"**
3. Click latest deployment
4. Check logs for: `"Using PostgreSQL database (persistent)"`
5. If you see `"Using SQLite database"` instead, `DATABASE_URL` isn't set

### Database connection errors?

1. Verify PostgreSQL service is running (should show green dot)
2. Check that both services are in the same project
3. Try redeploying the bot service

## Why This is Better

| Feature | SQLite (before) | PostgreSQL (now) |
|---------|----------------|------------------|
| Persistence | Lost on deploy | Persists forever |
| Reliability | Low | High |
| Backups | None | Automatic |
| Scalability | Single file | Production-ready |
| Cost | Free | Free (Railway tier) |

## What Changed in the Code

The bot now:
1. Checks for `DATABASE_URL` environment variable
2. If found → Uses PostgreSQL (Railway production)
3. If not found → Uses SQLite (local development)

**No more data loss!** 🎉
