# Telegram Bot Deployment Guide

Complete guide for deploying the veBTC Telegram notification bot to production.

## Prerequisites

- Ubuntu 20.04+ / Debian 10+ VPS (DigitalOcean, Hetzner, AWS EC2, etc.)
- Root or sudo access
- Telegram Bot Token from [@BotFather](https://t.me/botfather)
- Domain name (optional, but recommended for webhook mode)

## Quick Start (VPS Deployment)

### Step 1: Provision VPS

Recommended specifications:
- **RAM**: 1GB minimum (2GB recommended)
- **CPU**: 1 core minimum
- **Storage**: 10GB
- **Cost**: $5-10/month

Providers:
- [DigitalOcean](https://digitalocean.com) - $6/month Droplet
- [Hetzner Cloud](https://hetzner.com) - €3.79/month CX11
- [Linode](https://linode.com) - $5/month Nanode
- [AWS EC2](https://aws.amazon.com/ec2/) - t2.micro (free tier)

### Step 2: Initial Server Setup

SSH into your VPS:

```bash
ssh root@your-vps-ip
```

Update system:

```bash
apt update && apt upgrade -y
```

Install required packages:

```bash
apt install -y python3 python3-pip git supervisor curl
```

Create a dedicated user (optional but recommended):

```bash
adduser vebtc
usermod -aG sudo vebtc
su - vebtc
```

### Step 3: Clone Repository

```bash
cd /opt
sudo git clone https://github.com/your-username/mezo-vebtc.git
cd mezo-vebtc
```

Set permissions:

```bash
sudo chown -R vebtc:vebtc /opt/mezo-vebtc
```

### Step 4: Install Python Dependencies

```bash
pip3 install -r requirements.txt
```

Verify installation:

```bash
python3 -c "import telegram; print('Telegram library installed')"
```

### Step 5: Configure Bot Token

Create `.env` file:

```bash
nano .env
```

Add your bot token:

```
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz1234567890
```

Save and exit (Ctrl+X, Y, Enter).

**Important**: Ensure `.env` is gitignored:

```bash
echo ".env" >> .gitignore
```

### Step 6: Test Bot Locally

Run bot manually to verify it works:

```bash
python3 telegram_bot.py
```

You should see:
```
INFO - VeBTC Bot initialized successfully
INFO - Command handlers registered
INFO - Scheduler configured (check interval: 300s)
INFO - Scheduler started
INFO - Bot is running. Press Ctrl-C to stop.
```

Test by messaging your bot `/start` on Telegram.

Press Ctrl-C to stop the test run.

### Step 7: Setup Data Sync

The bot needs access to `vebtc_data.json` which is updated by GitHub Actions.

**Option A: Git Pull (Recommended)**

Create a cron job to pull updates every 10 minutes:

```bash
crontab -e
```

Add:

```bash
*/10 * * * * cd /opt/mezo-vebtc && git pull origin main >> /var/log/vebtc-sync.log 2>&1
```

**Option B: Direct Download**

If repository is public, download JSON directly:

Create sync script:

```bash
nano /opt/mezo-vebtc/sync_data.sh
```

Add:

```bash
#!/bin/bash
cd /opt/mezo-vebtc
curl -s https://raw.githubusercontent.com/your-username/mezo-vebtc/main/vebtc_data.json -o vebtc_data.json.tmp
if [ $? -eq 0 ]; then
    mv vebtc_data.json.tmp vebtc_data.json
    echo "$(date): Data synced successfully"
else
    echo "$(date): Failed to sync data"
fi
```

Make executable:

```bash
chmod +x /opt/mezo-vebtc/sync_data.sh
```

Add to cron:

```bash
*/10 * * * * /opt/mezo-vebtc/sync_data.sh >> /var/log/vebtc-sync.log 2>&1
```

### Step 8: Setup Supervisor (Process Manager)

Create supervisor config:

```bash
sudo nano /etc/supervisor/conf.d/vebtc-telegram-bot.conf
```

Add:

```ini
[program:vebtc-telegram-bot]
command=/usr/bin/python3 /opt/mezo-vebtc/telegram_bot.py
directory=/opt/mezo-vebtc
autostart=true
autorestart=true
stderr_logfile=/var/log/vebtc-telegram-bot.err.log
stdout_logfile=/var/log/vebtc-telegram-bot.out.log
user=vebtc
environment=TELEGRAM_BOT_TOKEN="%(ENV_TELEGRAM_BOT_TOKEN)s"
```

Load bot token into supervisor environment:

```bash
sudo nano /etc/supervisor/supervisord.conf
```

Add in `[supervisord]` section:

```ini
environment=TELEGRAM_BOT_TOKEN="your_bot_token_here"
```

Or use systemd environment file (more secure):

```bash
sudo nano /etc/supervisor/conf.d/vebtc-telegram-bot.conf
```

Change environment line to:

```ini
environment=PATH="/usr/bin:/usr/local/bin"
```

Then load from `.env`:

```bash
source /opt/mezo-vebtc/.env && sudo -E supervisorctl reread
```

### Step 9: Start Bot with Supervisor

Reload supervisor configuration:

```bash
sudo supervisorctl reread
sudo supervisorctl update
```

Start the bot:

```bash
sudo supervisorctl start vebtc-telegram-bot
```

Check status:

```bash
sudo supervisorctl status vebtc-telegram-bot
```

You should see:
```
vebtc-telegram-bot               RUNNING   pid 12345, uptime 0:00:05
```

### Step 10: Verify Bot is Running

Check logs:

```bash
sudo tail -f /var/log/vebtc-telegram-bot.out.log
```

Test bot on Telegram:
1. Open Telegram
2. Search for your bot (`@YourBotName`)
3. Send `/start`
4. You should receive the welcome message

## Useful Commands

### Supervisor Commands

```bash
# View all processes
sudo supervisorctl status

# Start bot
sudo supervisorctl start vebtc-telegram-bot

# Stop bot
sudo supervisorctl stop vebtc-telegram-bot

# Restart bot
sudo supervisorctl restart vebtc-telegram-bot

# View logs (last 1000 lines)
sudo supervisorctl tail -1000 vebtc-telegram-bot

# Follow logs in real-time
sudo supervisorctl tail -f vebtc-telegram-bot
```

### Log Management

```bash
# View error logs
sudo tail -f /var/log/vebtc-telegram-bot.err.log

# View output logs
sudo tail -f /var/log/vebtc-telegram-bot.out.log

# View bot's internal logs
tail -f /opt/mezo-vebtc/telegram_bot.log

# View data sync logs
tail -f /var/log/vebtc-sync.log
```

### Database Management

```bash
# View database
sqlite3 /opt/mezo-vebtc/subscribers.db

# Count subscribers
sqlite3 /opt/mezo-vebtc/subscribers.db "SELECT COUNT(*) FROM subscribers;"

# View all subscribers
sqlite3 /opt/mezo-vebtc/subscribers.db "SELECT * FROM subscribers;"

# Exit sqlite
.exit
```

## Log Rotation

Setup log rotation to prevent disk space issues:

```bash
sudo nano /etc/logrotate.d/vebtc-telegram-bot
```

Add:

```
/var/log/vebtc-telegram-bot.*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 vebtc vebtc
    sharedscripts
    postrotate
        supervisorctl restart vebtc-telegram-bot > /dev/null
    endscript
}

/opt/mezo-vebtc/telegram_bot.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 vebtc vebtc
}
```

## Monitoring

### Setup Health Check Script

Create health check:

```bash
nano /opt/mezo-vebtc/healthcheck.sh
```

Add:

```bash
#!/bin/bash

BOT_STATUS=$(supervisorctl status vebtc-telegram-bot | grep RUNNING)

if [ -z "$BOT_STATUS" ]; then
    echo "Bot is DOWN! Restarting..."
    supervisorctl restart vebtc-telegram-bot
    # Optional: Send alert to admin
else
    echo "Bot is UP"
fi
```

Make executable:

```bash
chmod +x /opt/mezo-vebtc/healthcheck.sh
```

Add to cron (check every 5 minutes):

```bash
crontab -e
```

Add:

```bash
*/5 * * * * /opt/mezo-vebtc/healthcheck.sh >> /var/log/vebtc-health.log 2>&1
```

### Monitoring Tools

**Option 1: UptimeRobot** (Free)
- Monitor bot HTTP endpoint (if you add one)
- Get email/SMS alerts on downtime
- https://uptimerobot.com

**Option 2: Prometheus + Grafana**
- Advanced metrics and dashboards
- Requires additional setup

## Updating the Bot

Pull latest changes:

```bash
cd /opt/mezo-vebtc
git pull origin main
```

Install new dependencies (if any):

```bash
pip3 install -r requirements.txt
```

Restart bot:

```bash
sudo supervisorctl restart vebtc-telegram-bot
```

## Security Best Practices

1. **Firewall Setup**

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTPS (if using webhooks)
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw enable
```

2. **Restrict SSH Access**

Edit SSH config:

```bash
sudo nano /etc/ssh/sshd_config
```

Change:
- `PermitRootLogin no`
- `PasswordAuthentication no` (use SSH keys)

Restart SSH:

```bash
sudo systemctl restart sshd
```

3. **Keep System Updated**

```bash
# Enable automatic security updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

4. **Database Backups**

Create backup script:

```bash
nano /opt/mezo-vebtc/backup_db.sh
```

Add:

```bash
#!/bin/bash
BACKUP_DIR="/opt/mezo-vebtc/backups"
mkdir -p $BACKUP_DIR
cp /opt/mezo-vebtc/subscribers.db $BACKUP_DIR/subscribers-$(date +%Y%m%d-%H%M%S).db

# Keep only last 7 days
find $BACKUP_DIR -name "subscribers-*.db" -mtime +7 -delete
```

Make executable and add to cron:

```bash
chmod +x /opt/mezo-vebtc/backup_db.sh
crontab -e
```

Add (daily at 2 AM):

```bash
0 2 * * * /opt/mezo-vebtc/backup_db.sh
```

## Troubleshooting

### Bot Not Starting

Check supervisor logs:

```bash
sudo supervisorctl tail vebtc-telegram-bot
```

Common issues:
- **Missing bot token**: Verify `.env` file exists and has correct token
- **Permission errors**: Check file ownership: `ls -la /opt/mezo-vebtc`
- **Port conflicts**: Ensure no other process is using bot's resources

### Notifications Not Sending

1. Check if data file is updating:

```bash
ls -lh /opt/mezo-vebtc/vebtc_data.json
```

2. Verify sync cron job is running:

```bash
sudo grep CRON /var/log/syslog | grep vebtc
```

3. Check bot logs for errors:

```bash
tail -50 /opt/mezo-vebtc/telegram_bot.log
```

### High Memory Usage

Monitor memory:

```bash
free -h
top -p $(pgrep -f telegram_bot.py)
```

If memory is high:
- Check for memory leaks in logs
- Restart bot: `sudo supervisorctl restart vebtc-telegram-bot`
- Upgrade VPS if consistently high

## Cost Breakdown

- **VPS Hosting**: $5-10/month
- **Domain** (optional): $1/month
- **Total**: $5-10/month

## Alternative: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "telegram_bot.py"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  telegram-bot:
    build: .
    restart: always
    volumes:
      - ./subscribers.db:/app/subscribers.db
      - ./vebtc_data.json:/app/vebtc_data.json
      - ./telegram_bot.log:/app/telegram_bot.log
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
```

Run:

```bash
docker-compose up -d
```

## Support

For deployment issues:
- Check logs first
- Review this guide
- GitHub Issues: [Report deployment problem](https://github.com/your-username/mezo-vebtc/issues)

## Next Steps

After successful deployment:
1. Monitor bot for 24-48 hours
2. Test all notification types
3. Add monitoring/alerting
4. Setup regular database backups
5. Document any customizations

Congratulations! Your veBTC Telegram bot is now live. 🎉
