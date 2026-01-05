"""Participant analytics and profiling."""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime


@dataclass
class ParticipantProfile:
    """Profile information for a participant."""
    address: str

    # Lock data
    total_locked: float
    num_locks: int
    first_lock_date: Optional[str]
    last_lock_date: Optional[str]

    # Vote data
    total_votes_cast: int
    current_voting_power: float
    token_ids: List[int]
    pools_voted: List[str]
    first_vote_date: Optional[str]
    last_vote_date: Optional[str]

    # Rankings
    lock_rank: Optional[int] = None
    vote_rank: Optional[int] = None

    @property
    def display_address(self) -> str:
        """Get shortened display address."""
        return f"{self.address[:6]}...{self.address[-4:]}"


class ParticipantAnalyzer:
    """Analyze participant data from locks and votes."""

    def __init__(self, locks_data: List[Dict[str, Any]], votes_data: List[Dict[str, Any]]):
        """Initialize analyzer with raw data.

        Args:
            locks_data: List of lock records
            votes_data: List of vote records
        """
        self.locks_data = locks_data
        self.votes_data = votes_data
        self._participants_cache = None

    def get_all_participants(self) -> Dict[str, ParticipantProfile]:
        """Get profiles for all participants.

        Returns:
            Dictionary mapping address to ParticipantProfile
        """
        if self._participants_cache is not None:
            return self._participants_cache

        # Aggregate lock data by address
        lock_agg = defaultdict(lambda: {
            'total': 0.0,
            'count': 0,
            'dates': []
        })

        for lock in self.locks_data:
            sender = lock.get('sender', '').lower()
            if sender and sender != 'unknown':
                lock_agg[sender]['total'] += lock.get('amount', 0)
                lock_agg[sender]['count'] += 1
                if 'date' in lock:
                    lock_agg[sender]['dates'].append(lock['date'])

        # Aggregate vote data by address
        vote_agg = defaultdict(lambda: {
            'count': 0,
            'voting_power': 0.0,
            'token_ids': set(),
            'pools': set(),
            'dates': []
        })

        for vote in self.votes_data:
            voter = vote.get('voter', '').lower()
            if voter and voter != 'unknown':
                vote_agg[voter]['count'] += 1
                vote_agg[voter]['voting_power'] = vote.get('voting_power', 0)  # Latest voting power

                token_id = vote.get('token_id')
                if token_id is not None:
                    vote_agg[voter]['token_ids'].add(token_id)

                pool = vote.get('pool')
                if pool and pool != 'Unknown':
                    vote_agg[voter]['pools'].add(pool)

                if 'date' in vote:
                    vote_agg[voter]['dates'].append(vote['date'])

        # Combine into profiles
        all_addresses = set(lock_agg.keys()) | set(vote_agg.keys())
        participants = {}

        for address in all_addresses:
            locks = lock_agg.get(address, {})
            votes = vote_agg.get(address, {})

            # Get date ranges
            lock_dates = sorted(locks.get('dates', []))
            vote_dates = sorted(votes.get('dates', []))

            profile = ParticipantProfile(
                address=address,
                total_locked=locks.get('total', 0.0),
                num_locks=locks.get('count', 0),
                first_lock_date=lock_dates[0] if lock_dates else None,
                last_lock_date=lock_dates[-1] if lock_dates else None,
                total_votes_cast=votes.get('count', 0),
                current_voting_power=votes.get('voting_power', 0.0),
                token_ids=sorted(list(votes.get('token_ids', set()))),
                pools_voted=sorted(list(votes.get('pools', set()))),
                first_vote_date=vote_dates[0] if vote_dates else None,
                last_vote_date=vote_dates[-1] if vote_dates else None
            )

            participants[address] = profile

        self._participants_cache = participants
        return participants

    def get_participant(self, address: str) -> Optional[ParticipantProfile]:
        """Get profile for a specific participant.

        Args:
            address: Participant address (case-insensitive)

        Returns:
            ParticipantProfile or None if not found
        """
        participants = self.get_all_participants()
        return participants.get(address.lower())

    def search_by_token_id(self, token_id: int) -> Optional[ParticipantProfile]:
        """Find participant by their token ID.

        Args:
            token_id: veBTC token ID

        Returns:
            ParticipantProfile or None if not found
        """
        participants = self.get_all_participants()
        for profile in participants.values():
            if token_id in profile.token_ids:
                return profile
        return None

    def get_top_lockers(self, limit: int = 20) -> List[ParticipantProfile]:
        """Get top participants by total BTC locked.

        Args:
            limit: Maximum number of results

        Returns:
            List of ParticipantProfile sorted by total_locked (descending)
        """
        participants = self.get_all_participants()

        # Filter to only those with locks
        lockers = [p for p in participants.values() if p.total_locked > 0]

        # Sort by total locked (descending)
        sorted_lockers = sorted(lockers, key=lambda p: p.total_locked, reverse=True)

        # Assign ranks
        for rank, profile in enumerate(sorted_lockers[:limit], start=1):
            profile.lock_rank = rank

        return sorted_lockers[:limit]

    def get_top_voters(self, limit: int = 20) -> List[ParticipantProfile]:
        """Get top participants by current voting power.

        Args:
            limit: Maximum number of results

        Returns:
            List of ParticipantProfile sorted by current_voting_power (descending)
        """
        participants = self.get_all_participants()

        # Filter to only those with voting power
        voters = [p for p in participants.values() if p.current_voting_power > 0]

        # Sort by current voting power (descending)
        sorted_voters = sorted(voters, key=lambda p: p.current_voting_power, reverse=True)

        # Assign ranks
        for rank, profile in enumerate(sorted_voters[:limit], start=1):
            profile.vote_rank = rank

        return sorted_voters[:limit]

    def get_pool_voters(self, pool_address: str, limit: int = 20) -> List[ParticipantProfile]:
        """Get top voters for a specific pool.

        Args:
            pool_address: Pool address
            limit: Maximum number of results

        Returns:
            List of ParticipantProfile who voted on this pool
        """
        participants = self.get_all_participants()

        # Filter to those who voted on this pool
        pool_voters = [
            p for p in participants.values()
            if pool_address.lower() in [pool.lower() for pool in p.pools_voted]
        ]

        # Sort by voting power
        sorted_voters = sorted(pool_voters, key=lambda p: p.current_voting_power, reverse=True)

        return sorted_voters[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics across all participants.

        Returns:
            Dictionary with statistics
        """
        participants = self.get_all_participants()

        total_participants = len(participants)
        participants_with_locks = sum(1 for p in participants.values() if p.num_locks > 0)
        participants_with_votes = sum(1 for p in participants.values() if p.total_votes_cast > 0)

        total_btc_locked = sum(p.total_locked for p in participants.values())
        total_voting_power = sum(p.current_voting_power for p in participants.values())

        avg_locks_per_participant = (
            sum(p.num_locks for p in participants.values()) / participants_with_locks
            if participants_with_locks > 0 else 0
        )

        avg_votes_per_participant = (
            sum(p.total_votes_cast for p in participants.values()) / participants_with_votes
            if participants_with_votes > 0 else 0
        )

        return {
            'total_participants': total_participants,
            'participants_with_locks': participants_with_locks,
            'participants_with_votes': participants_with_votes,
            'total_btc_locked': total_btc_locked,
            'total_voting_power': total_voting_power,
            'avg_locks_per_participant': avg_locks_per_participant,
            'avg_votes_per_participant': avg_votes_per_participant
        }
