"""Epoch data aggregation for historical analysis."""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from lib.analytics.epoch_tracker import EpochTracker


@dataclass
class EpochMetrics:
    """Aggregated metrics for an epoch."""
    epoch_number: int
    start_ts: int
    end_ts: int
    start_date: str
    end_date: str

    # Vote metrics
    total_voted: float
    unique_voters: int
    vote_tx_count: int

    # Incentive metrics
    total_bribes_usd: float
    total_fees_usd: float
    average_apr: float
    pool_count: int

    # Pool details
    pools: List[Dict[str, Any]]


class EpochAggregator:
    """Aggregate historical epoch data."""

    def __init__(self, votes_data: List[Dict[str, Any]]):
        """Initialize aggregator.

        Args:
            votes_data: List of parsed vote records
        """
        self.votes_data = votes_data
        self.epoch_tracker = EpochTracker()

    def aggregate_votes_by_epoch(self, epoch_number: int) -> Dict[str, Any]:
        """Aggregate vote data for a specific epoch.

        Args:
            epoch_number: Epoch number to aggregate

        Returns:
            Dictionary with vote metrics
        """
        epoch_info = self.epoch_tracker.get_epoch_by_number(epoch_number)

        # Filter votes within this epoch's voting window
        epoch_votes = []
        for vote in self.votes_data:
            vote_ts = self._parse_timestamp(vote.get('ts'))
            if vote_ts and epoch_info.vote_start_ts <= vote_ts <= epoch_info.vote_end_ts:
                epoch_votes.append(vote)

        # Calculate metrics
        unique_voters = len(set(v.get('voter') for v in epoch_votes if v.get('voter') and v.get('voter') != 'Unknown'))
        vote_tx_count = len(epoch_votes)

        # Get latest total_weight for each pool (total voted per pool)
        pool_votes = {}
        for vote in sorted(epoch_votes, key=lambda v: self._parse_timestamp(v.get('ts', 0)) or 0):
            pool = vote.get('pool')
            if pool and pool != 'Unknown':
                pool_votes[pool] = vote.get('total_weight', 0)

        total_voted = sum(pool_votes.values())

        return {
            "total_voted": total_voted,
            "unique_voters": unique_voters,
            "vote_tx_count": vote_tx_count,
            "pool_votes": pool_votes  # For pool-level aggregation
        }

    def calculate_epoch_metrics(self,
                                epoch_number: int,
                                vote_metrics: Dict[str, Any],
                                incentives_data: List[Dict[str, Any]]) -> EpochMetrics:
        """Calculate complete metrics for an epoch.

        Args:
            epoch_number: Epoch number
            vote_metrics: Vote aggregation results
            incentives_data: Pool incentives data

        Returns:
            EpochMetrics object
        """
        epoch_info = self.epoch_tracker.get_epoch_by_number(epoch_number)

        # Aggregate incentive data
        total_bribes_usd = sum(p.get('bribes_usd', 0) for p in incentives_data)
        total_fees_usd = sum(p.get('fees_usd', 0) for p in incentives_data)

        # Calculate average APR (weighted by votes)
        total_votes = sum(p.get('current_votes', 0) for p in incentives_data)
        if total_votes > 0:
            weighted_apr = sum(
                p.get('apr_total', 0) * p.get('current_votes', 0)
                for p in incentives_data
            )
            average_apr = weighted_apr / total_votes
        else:
            average_apr = 0.0

        # Build pool list
        pools = []
        for pool in incentives_data:
            pools.append({
                "pool_address": pool.get('pool_address'),
                "pool_name": pool.get('pool_name'),
                "votes": pool.get('current_votes', 0),
                "bribes_usd": pool.get('bribes_usd', 0),
                "fees_usd": pool.get('fees_usd', 0),
                "apr_total": pool.get('apr_total', 0)
            })

        return EpochMetrics(
            epoch_number=epoch_number,
            start_ts=epoch_info.start_ts,
            end_ts=epoch_info.end_ts,
            start_date=epoch_info.start_date.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=epoch_info.end_date.strftime("%Y-%m-%d %H:%M:%S"),
            total_voted=vote_metrics['total_voted'],
            unique_voters=vote_metrics['unique_voters'],
            vote_tx_count=vote_metrics['vote_tx_count'],
            total_bribes_usd=total_bribes_usd,
            total_fees_usd=total_fees_usd,
            average_apr=average_apr,
            pool_count=len(pools),
            pools=pools
        )

    @staticmethod
    def _parse_timestamp(ts) -> Optional[int]:
        """Parse timestamp from various formats."""
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                return int(dt.timestamp())
            except (ValueError, AttributeError):
                return None
        elif hasattr(ts, 'timestamp'):
            return int(ts.timestamp())
        elif isinstance(ts, (int, float)):
            return int(ts)
        return None
