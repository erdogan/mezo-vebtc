# Deploy Telegram Bot on Google Cloud (FREE Forever)

Google Cloud offers a **free e2-micro VM** that runs 24/7 forever - perfect for the bot.

## Free Tier Details

- **Instance:** e2-micro (0.25-1 vCPU, 1GB RAM)
- **Disk:** 30GB standard persistent disk
- **Egress:** 1GB network egress per month (enough for bot)
- **Always Free:** Not a trial, free forever
- **Regions:** us-west1, us-central1, or us-east1

## Prerequisites

- Google account
- Credit card for verification (won't be charged on free tier)
- Your bot token: `8596819881:AAHj8cpwRNW4Zc4Ryc-MAb_Ucb4RmWqe9fo`

## Step 1: Create GCP Account

1. Go to https://console.cloud.google.com
2. Sign up with Google account
3. Add credit card for verification (required but not charged)
4. Accept free trial ($300 credit for 90 days, separate from always-free tier)

## Step 2: Create VM Instance

1. **Go to Compute Engine:**
   - In GCP Console, click hamburger menu (☰)
   - Navigate to: Compute Engine → VM instances
   - Click "Create Instance"

2. **Configure Instance (Important - for free tier):**
   ```
   Name: vebtc-telegram-bot
   Region: us-west1 (Oregon) or us-central1 or us-east1
   Zone: Any zone in selected region

   Machine configuration:
   - Series: E2
   - Machine type: e2-micro (2 vCPU, 1GB memory) ✓ Free tier eligible

   Boot disk:
   - Click "Change"
   - Operating System: Ubuntu
   - Version: Ubuntu 22.04 LTS
   - Boot disk type: Standard persistent disk
   - Size: 30 GB (max for free tier)
   - Click "Select"

   Firewall:
   - ☐ Allow HTTP traffic (not needed)
   - ☐ Allow HTTPS traffic (not needed)
   ```

3. **Click "Create"** (takes 1-2 minutes)

## Step 3: Connect to VM

1. **SSH into instance:**
   - In VM instances list, click "SSH" button next to your instance
   - A browser SSH window will open

2. **Update system:**
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

3. **Install Python and Git:**
   ```bash
   sudo apt install -y python3 python3-pip git
   ```

## Step 4: Deploy Bot

1. **Clone repository:**
   ```bash
   cd ~
   git clone https://github.com/erdogan/mezo-vebtc.git
   cd mezo-vebtc
   ```

2. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Create environment file:**
   ```bash
   cat > .env << 'EOF'
   TELEGRAM_BOT_TOKEN=8596819881:AAHj8cpwRNW4Zc4Ryc-MAb_Ucb4RmWqe9fo
   GITHUB_DATA_URL=https://raw.githubusercontent.com/erdogan/mezo-vebtc/main/vebtc_data.json
   EOF
   ```

4. **Test bot:**
   ```bash
   python3 telegram_bot.py
   ```

   Press Ctrl+C after verifying it starts successfully.

## Step 5: Run Bot as Service (Auto-start)

1. **Create systemd service:**
   ```bash
   sudo tee /etc/systemd/system/vebtc-bot.service > /dev/null << 'EOF'
   [Unit]
   Description=veBTC Telegram Bot
   After=network.target

   [Service]
   Type=simple
   User=YOUR_USERNAME
   WorkingDirectory=/home/YOUR_USERNAME/mezo-vebtc
   ExecStart=/usr/bin/python3 /home/YOUR_USERNAME/mezo-vebtc/telegram_bot.py
   Restart=always
   RestartSec=10
   Environment="PYTHONUNBUFFERED=1"

   [Install]
   WantedBy=multi-user.target
   EOF
   ```

2. **Replace YOUR_USERNAME with your actual username:**
   ```bash
   # Get your username
   whoami

   # Edit the service file
   sudo nano /etc/systemd/system/vebtc-bot.service
   # Replace YOUR_USERNAME with output from whoami
   # Save with Ctrl+X, Y, Enter
   ```

3. **Start and enable service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start vebtc-bot
   sudo systemctl enable vebtc-bot
   ```

4. **Check status:**
   ```bash
   sudo systemctl status vebtc-bot
   ```

## Step 6: Verify Bot is Running

1. **Check logs:**
   ```bash
   sudo journalctl -u vebtc-bot -f
   ```

   Should see: "Bot started successfully"

2. **Test in Telegram:**
   - Send `/start` to your bot
   - Try `/status`, `/pools`

## Auto-Update Data

The bot automatically fetches from GitHub, so no cron job needed!

Your GitHub Actions updates `vebtc_data.json` every 10 minutes, and the bot fetches it on each command.

## Management Commands

```bash
# View logs
sudo journalctl -u vebtc-bot -f

# Stop bot
sudo systemctl stop vebtc-bot

# Start bot
sudo systemctl start vebtc-bot

# Restart bot
sudo systemctl restart vebtc-bot

# Update bot code
cd ~/mezo-vebtc
git pull
sudo systemctl restart vebtc-bot
```

## Monitor Free Tier Usage

1. Go to: https://console.cloud.google.com/billing
2. View "Cost breakdown"
3. Ensure you're staying within free tier

The bot should use:
- Compute: Free (e2-micro)
- Network egress: <1GB/month (free)
- Storage: 30GB (free)

## Security Best Practices

1. **Set up firewall rules:**
   ```bash
   # Only allow SSH from your IP
   gcloud compute firewall-rules create allow-ssh-from-my-ip \
     --allow tcp:22 \
     --source-ranges YOUR_IP_ADDRESS/32
   ```

2. **Regular updates:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

3. **Monitor bot logs:**
   ```bash
   sudo journalctl -u vebtc-bot --since "1 hour ago"
   ```

## Troubleshooting

### Bot not starting

```bash
# Check logs
sudo journalctl -u vebtc-bot -n 50

# Check if process is running
ps aux | grep telegram_bot

# Restart service
sudo systemctl restart vebtc-bot
```

### Out of memory

e2-micro has 1GB RAM. If bot crashes:
```bash
# Add swap space
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Data fetch errors

```bash
# Test GitHub URL
curl https://raw.githubusercontent.com/erdogan/mezo-vebtc/main/vebtc_data.json

# Check bot can access internet
ping -c 3 google.com
```

## Cost Alerts (Safety)

Set up billing alert to avoid surprise charges:

1. Go to: https://console.cloud.google.com/billing
2. Click "Budgets & alerts"
3. Create budget: $1
4. Get email if exceeded

## Backup and Restore

Subscriber database is stored in `~/mezo-vebtc/subscribers.db`

**Backup:**
```bash
cp ~/mezo-vebtc/subscribers.db ~/subscribers.db.backup
```

**Restore:**
```bash
cp ~/subscribers.db.backup ~/mezo-vebtc/subscribers.db
sudo systemctl restart vebtc-bot
```

## Advantages vs Other Platforms

| Feature | GCP e2-micro | Railway | Render |
|---------|--------------|---------|--------|
| Cost | Free forever | $5 credit/month | $7/month |
| Sleep | No | No | Yes (15 min) |
| Control | Full VM | Limited | Limited |
| Setup | 15 min | 5 min | 5 min |
| Best for | Production | Quick start | Paid plans |

## Next Steps

After deployment:
1. Test all bot commands
2. Link your wallet: `/link YOUR_ADDRESS`
3. Check notifications work
4. Monitor for 24 hours
5. Set up billing alerts

## Support

- GCP Documentation: https://cloud.google.com/compute/docs
- GCP Free Tier: https://cloud.google.com/free
- Check bot logs: `sudo journalctl -u vebtc-bot -f`
