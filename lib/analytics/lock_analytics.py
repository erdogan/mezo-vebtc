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

        # Aggregate by wallet
        wallet_data = defaultdict(lambda: {
            'total_locked': 0.0,
            'locks': [],
            'token_ids': set()
        })

        for deposit in self.deposits:
            # Only include CREATE_LOCK deposits for analytics
            if deposit.get('deposit_type') != 0:
                continue

            provider = deposit.get('provider')
            if not provider or provider == 'unknown':
                continue

            value = deposit.get('value', 0)
            wallet_data[provider]['total_locked'] += value
            wallet_data[provider]['locks'].append(deposit)

            token_id = deposit.get('token_id')
            if token_id:
                wallet_data[provider]['token_ids'].add(token_id)

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
