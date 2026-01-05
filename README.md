# veBTC Voting Analytics & Telegram Bot

A comprehensive analytics platform and notification system for veBTC (Bitcoin governance token) voting on the Mezo network. This project provides real-time voting data, pool analytics, and automated Telegram notifications to keep the community engaged with governance decisions.

## Overview

This repository contains two main components:

1. **Web Dashboard** - Interactive analytics dashboard displaying voting power, pool performance, incentives, and leaderboards
2. **Telegram Bot** - Automated notification system that sends voting reminders, epoch updates, and high APR alerts

## Dashboard Features

The dashboard (`index.html`) provides comprehensive voting analytics:

### Epoch Information
- Current epoch number with date range
- Real-time countdown timer until voting closes
- Voting status (open/closed) indicator
- Total voting power and unique voter count

### Pool Analytics
- Complete list of all voting pools
- Current voting power allocated to each pool
- Bribes and fees incentives per pool
- APR calculations (bribe-based and fee-based)
- USD value per vote metrics

### Voting Statistics
- Interactive charts showing historical locks and votes (powered by Plotly.js)
- Participant leaderboards ranking top voters and lock contributors
- Searchable interface for finding specific wallets or participants

### Incentives Tracking
- Total bribes offered across all pools
- Total fees available for distribution
- APR metrics for investment decisions
- Pool-by-pool incentive breakdown

## Telegram Bot Features

The bot provides automated notifications and on-demand information about veBTC voting.

### Available Commands

| Command | Description |
|---------|-------------|
| `/start` | Subscribe to bot notifications |
| `/subscribe` | Re-subscribe if you've unsubscribed |
| `/unsubscribe` | Stop receiving all notifications |
| `/link <address>` | Link your Ethereum wallet for personalized notifications |
| `/unlink` | Remove wallet association |
| `/epoch` | View current epoch status and your voting participation |
| `/myvotes` | See your voting history for all pools in current epoch (requires linked wallet) |
| `/pools` | Display top 5 pools by APR with incentive details |
| `/settings` | View your notification preferences |
| `/botstats` | Show bot usage statistics |
| `/help` | Display all available commands |

### Automated Notifications

The bot sends three types of automatic notifications:

#### 24-Hour Voting Reminders
Sent 24 hours before voting closes:
- **General broadcast** - Sent to all subscribers with top 3 pools
- **Personalized for non-voters** - Shows your voting power if you haven't voted
- **Personalized for voters** - Displays your current votes

#### Final Warnings
Sent 1 hour before voting closes:
- Only sent to linked wallet holders who haven't voted yet
- Includes your voting power and time remaining

#### Epoch Start Announcements
Sent when a new epoch begins:
- Announces voting window duration
- Shows top pools by APR
- Includes epoch date range

#### High APR Alerts
Continuously monitored alerts:
- Real-time notifications when pools exceed APR threshold (default: 50%)
- Configurable alert intervals
- Helps users identify profitable voting opportunities

## Technical Architecture

### Data Sources
- **Mezo Explorer API** (`api.explorer.mezo.org`)
  - Lock transactions from veBTC contract
  - Vote logs from Voter contract
- **GitHub-hosted data** for dashboard generation
- **Direct RPC calls** for real-time blockchain data

### Smart Contracts (Mezo Network)

| Contract | Address | Purpose |
|----------|---------|---------|
| **veBTC** | `0x3D4b1b884A7a1E59fE8589a3296EC8f8cBB6f279` | Main governance token |
| **Voter** | `0x48233cCC97B87Ba93bCA212cbEe48e3210211f03` | Voting mechanism |
