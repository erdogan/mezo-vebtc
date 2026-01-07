"""Lock analytics calculator for wallet-level lock statistics."""
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class WalletLockProfile:
    """Lock profile for a wallet address."""
    address: str
    total_btc_locked: float
    lock_count: int
    max_lock_count: int  # Number of max-duration locks
    max_lock_percentage: float  # % of locks at max duration
    locks: List[Dict[str, Any]]  # Individual lock details
    unique_token_ids: List[int]

    @property
    def display_address(self) -> str:
        """Get shortened display address."""
        return f"{self.address[:6]}...{self.address[-4:]}"


class LockAnalyzer:
    """Analyze lock data to produce wallet-level statistics."""

    def __init__(self, deposits: List[Dict[str, Any]]):
        """Initialize with parsed deposit data.

        Args:
            deposits: List of parsed deposit events
        """
        self.deposits = deposits
        self._profiles_cache = None

    def get_wallet_profiles(self) -> Dict[str, WalletLockProfile]:
        """Calculate lock profiles for all wallets.

        Returns:
            Dict mapping address to WalletLockProfile
        """
        if self._profiles_cache is not None:
            return self._profiles_cache

        # Step 1: Build token_id -> owner mapping from CREATE_LOCK events
        token_owners = {}
        for deposit in self.deposits:
            if deposit.get('deposit_type') == 0:  # CREATE_LOCK
                token_id = deposit.get('token_id')
                provider = deposit.get('provider')
                if token_id and provider and provider != 'unknown':
                    token_owners[token_id] = provider

        # Step 2: Aggregate all deposits by token_id
        token_data = defaultdict(lambda: {
            'total_locked': 0.0,
            'deposits': [],
            'create_lock': None
        })

        for deposit in self.deposits:
            token_id = deposit.get('token_id')
            if not token_id:
                continue

            deposit_type = deposit.get('deposit_type')
            value = deposit.get('value', 0)

            # Track all deposits for this token
            # CREATE_LOCK (0): Initial lock creation
            # DEPOSIT_FOR (1): Someone deposits for this token
            # INCREASE_AMOUNT (2): Owner increases amount
            # INCREASE_UNLOCK_TIME (3): Owner extends lock
            token_data[token_id]['deposits'].append(deposit)
            token_data[token_id]['total_locked'] += value

            # Store CREATE_LOCK separately for lock duration info
            if deposit_type == 0:
                token_data[token_id]['create_lock'] = deposit

        # Step 3: Aggregate tokens by owner wallet
        wallet_data = defaultdict(lambda: {
            'total_locked': 0.0,
            'locks': [],  # CREATE_LOCK events only
            'token_ids': set(),
            'all_deposits': []  # All deposit events for this wallet's tokens
        })

        for token_id, data in token_data.items():
            owner = token_owners.get(token_id)
            if not owner:
                continue

            wallet_data[owner]['total_locked'] += data['total_locked']
            wallet_data[owner]['token_ids'].add(token_id)
            wallet_data[owner]['all_deposits'].extend(data['deposits'])

            # Add CREATE_LOCK event if exists
            if data['create_lock']:
                wallet_data[owner]['locks'].append(data['create_lock'])

        # Build profiles
        profiles = {}

        for address, data in wallet_data.items():
            locks = data['locks']
            lock_count = len(locks)
            max_lock_count = sum(1 for lock in locks if lock.get('is_max_lock'))
            max_lock_pct = (max_lock_count / lock_count * 100) if lock_count > 0 else 0

            profile = WalletLockProfile(
                address=address,
                total_btc_locked=data['total_locked'],
                lock_count=lock_count,
                max_lock_count=max_lock_count,
                max_lock_percentage=max_lock_pct,
                locks=sorted(locks, key=lambda x: x.get('timestamp', 0), reverse=True),
                unique_token_ids=sorted(list(data['token_ids']))
            )

            profiles[address] = profile

        self._profiles_cache = profiles
        return profiles

    def get_top_by_max_locks(self, limit: int = 50) -> List[WalletLockProfile]:
        """Get wallets sorted by number of max-duration locks.

        Args:
            limit: Maximum number of profiles to return

        Returns:
            List of WalletLockProfile objects
        """
        profiles = self.get_wallet_profiles()
        sorted_profiles = sorted(
            profiles.values(),
            key=lambda p: p.max_lock_count,
            reverse=True
        )
        return sorted_profiles[:limit]

    def get_top_by_max_percentage(self, limit: int = 50, min_locks: int = 3) -> List[WalletLockProfile]:
        """Get wallets sorted by percentage of max-duration locks.

        Args:
            limit: Maximum number of profiles to return
            min_locks: Minimum number of locks required to be included

        Returns:
            List of WalletLockProfile objects
        """
        profiles = self.get_wallet_profiles()
        # Filter to wallets with at least min_locks
        filtered = [p for p in profiles.values() if p.lock_count >= min_locks]
        sorted_profiles = sorted(
            filtered,
            key=lambda p: p.max_lock_percentage,
            reverse=True
        )
        return sorted_profiles[:limit]

    def get_top_by_total_btc(self, limit: int = 50) -> List[WalletLockProfile]:
        """Get wallets sorted by total BTC locked.

        Args:
            limit: Maximum number of profiles to return

        Returns:
            List of WalletLockProfile objects
        """
        profiles = self.get_wallet_profiles()
        sorted_profiles = sorted(
            profiles.values(),
            key=lambda p: p.total_btc_locked,
            reverse=True
        )
        return sorted_profiles[:limit]

    def get_top_by_lock_count(self, limit: int = 50) -> List[WalletLockProfile]:
        """Get wallets sorted by total number of locks.

        Args:
            limit: Maximum number of profiles to return

        Returns:
            List of WalletLockProfile objects
        """
        profiles = self.get_wallet_profiles()
        sorted_profiles = sorted(
            profiles.values(),
            key=lambda p: p.lock_count,
            reverse=True
        )
        return sorted_profiles[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics.

        Returns:
            Dictionary with aggregate stats
        """
        profiles = self.get_wallet_profiles()

        total_wallets = len(profiles)
        total_locks = sum(p.lock_count for p in profiles.values())
        total_max_locks = sum(p.max_lock_count for p in profiles.values())
        total_btc = sum(p.total_btc_locked for p in profiles.values())

        avg_locks_per_wallet = total_locks / total_wallets if total_wallets > 0 else 0
        max_lock_rate = (total_max_locks / total_locks * 100) if total_locks > 0 else 0

        return {
            'total_wallets': total_wallets,
            'total_locks': total_locks,
            'total_max_locks': total_max_locks,
            'total_btc_locked': total_btc,
            'avg_locks_per_wallet': avg_locks_per_wallet,
            'max_lock_rate': max_lock_rate
        }
