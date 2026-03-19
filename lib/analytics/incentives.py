"""Incentives and APR/ROI calculation module."""
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PoolIncentives:
    """Incentive information for a pool."""
    pool_address: str
    pool_name: str
    current_votes: float
    bribes: Dict[str, float]  # token_symbol -> amount
    bribes_usd: float
    fees: Dict[str, float]  # token_symbol -> amount
    fees_usd: float
    apr_bribes: float
    apr_fees: float
    apr_total: float
    usd_per_vote: float


class IncentivesCalculator:
    """Calculate APR and ROI for pool incentives."""

    WEEKS_PER_YEAR = 52
    DEFAULT_BTC_PRICE = 100000  # Fallback BTC price in USD

    def __init__(self, token_prices: Optional[Dict[str, float]] = None):
        """Initialize calculator with token prices.

        Args:
            token_prices: Dictionary of token symbols to USD prices
        """
        self.token_prices = token_prices or {"BTC": self.DEFAULT_BTC_PRICE}

    def calculate_bribes_usd(self, bribes: Dict[str, float]) -> float:
        """Calculate total USD value of bribes.

        Args:
            bribes: Dictionary of token symbols to amounts

        Returns:
            Total USD value
        """
        total_usd = 0.0
        for token, amount in bribes.items():
            price = self.token_prices.get(token, 0)
            total_usd += amount * price
        return total_usd

    def calculate_bribe_apr(self,
                           bribes_usd: float,
                           pool_votes: float,
                           btc_price: Optional[float] = None) -> float:
        """Calculate APR from bribes.

        Formula: (Bribes USD / Pool Votes) * (52 weeks) / BTC Price * 100

        Args:
            bribes_usd: Total USD value of bribes for current epoch
            pool_votes: Total voting power for the pool
            btc_price: BTC price in USD (optional, uses default if not provided)

        Returns:
            APR percentage
        """
        if pool_votes == 0:
            return 0.0

        btc_price = btc_price or self.token_prices.get("BTC", self.DEFAULT_BTC_PRICE)

        # $/vote for this epoch
        usd_per_vote = bribes_usd / pool_votes

        # Annualize (52 epochs)
        annual_return_per_vote = usd_per_vote * self.WEEKS_PER_YEAR

        # APR = (Annual Return / Principal) * 100
        # Assume 1 veBTC ≈ 1 BTC locked (conservative)
        apr = (annual_return_per_vote / btc_price) * 100

        return apr

    def calculate_fee_apr(self,
                         historical_fees_usd: List[float],
                         pool_votes: float,
                         btc_price: Optional[float] = None) -> float:
        """Calculate APR from historical fees.

        Formula: (Avg Weekly Fees USD / Pool Votes) * (52 weeks) / BTC Price * 100

        Args:
            historical_fees_usd: List of weekly fee totals in USD (last 4-10 weeks)
            pool_votes: Average voting power for the pool
            btc_price: BTC price in USD

        Returns:
            APR percentage
        """
        if pool_votes == 0 or not historical_fees_usd:
            return 0.0

        btc_price = btc_price or self.token_prices.get("BTC", self.DEFAULT_BTC_PRICE)

        # Average weekly fees
        avg_weekly_fees = sum(historical_fees_usd) / len(historical_fees_usd)

        # $/vote per week
        usd_per_vote = avg_weekly_fees / pool_votes

        # Annualize
        annual_return_per_vote = usd_per_vote * self.WEEKS_PER_YEAR

        # APR
        apr = (annual_return_per_vote / btc_price) * 100

        return apr

    def calculate_pool_incentives(self,
                                  pool_address: str,
                                  pool_name: str,
                                  current_votes: float,
                                  current_epoch_bribes: Dict[str, float],
                                  current_epoch_fees: Optional[Dict[str, float]] = None,
                                  historical_fees: Optional[List[Dict[str, float]]] = None) -> PoolIncentives:
        """Calculate complete incentive information for a pool.

        Args:
            pool_address: Pool contract address
            pool_name: Human-readable pool name
            current_votes: Current voting power allocated to pool
            current_epoch_bribes: Current epoch bribes {token: amount}
            current_epoch_fees: Current epoch fees {token: amount}
            historical_fees: Last 4-10 epochs fees [{token: amount}, ...]

        Returns:
            PoolIncentives object with all calculations
        """
        # Calculate bribe values
        bribes_usd = self.calculate_bribes_usd(current_epoch_bribes)
        bribe_apr = self.calculate_bribe_apr(bribes_usd, current_votes)

        # Calculate fee values and APR
        fees_usd = 0.0
        fee_apr = 0.0
        fees_dict = current_epoch_fees or {}

        # If we have current epoch fees, calculate USD value
        if current_epoch_fees:
            fees_usd = self.calculate_bribes_usd(current_epoch_fees)  # Same USD calculation

        # If we have historical fees, use them for APR calculation
        if historical_fees:
            # Sum up recent fees
            fees_totals = {}
            fees_usd_list = []

            for epoch_fees in historical_fees[-10:]:  # Last 10 epochs max
                epoch_usd = self.calculate_bribes_usd(epoch_fees)  # Same calculation
                fees_usd_list.append(epoch_usd)

                for token, amount in epoch_fees.items():
                    fees_totals[token] = fees_totals.get(token, 0) + amount

            # Average fees per token
            num_epochs = len(historical_fees[-10:])
            fees_dict = {token: amount / num_epochs for token, amount in fees_totals.items()}
            fees_usd = sum(fees_usd_list) / len(fees_usd_list)

            # Calculate fee APR
            fee_apr = self.calculate_fee_apr(fees_usd_list, current_votes)

        # Calculate USD per vote (for this epoch)
        usd_per_vote = (bribes_usd + fees_usd) / current_votes if current_votes > 0 else 0

        return PoolIncentives(
            pool_address=pool_address,
            pool_name=pool_name,
            current_votes=current_votes,
            bribes=current_epoch_bribes,
            bribes_usd=bribes_usd,
            fees=fees_dict,
            fees_usd=fees_usd,
            apr_bribes=bribe_apr,
            apr_fees=fee_apr,
            apr_total=bribe_apr + fee_apr,
            usd_per_vote=usd_per_vote
        )

    def calculate_roi_projection(self,
                                 voting_power: float,
                                 pool_incentives: PoolIncentives,
                                 epochs: int = 52) -> Dict[str, float]:
        """Calculate projected ROI for voting on a pool.

        Args:
            voting_power: User's voting power in veBTC
            pool_incentives: Pool incentive information
            epochs: Number of epochs to project (default 52 = 1 year)

        Returns:
            Dictionary with ROI projections
        """
        # Calculate user's share of pool
        if pool_incentives.current_votes == 0:
            user_share = 0
        else:
            user_share = voting_power / (pool_incentives.current_votes + voting_power)

        # Projected rewards per epoch
        epoch_rewards_usd = (pool_incentives.bribes_usd + pool_incentives.fees_usd) * user_share

        # Total projected rewards
        total_rewards_usd = epoch_rewards_usd * epochs

        # Calculate ROI
        btc_price = self.token_prices.get("BTC", self.DEFAULT_BTC_PRICE)
        principal_usd = voting_power * btc_price
        roi_percentage = (total_rewards_usd / principal_usd * 100) if principal_usd > 0 else 0

        return {
            "user_share_percentage": user_share * 100,
            "epoch_rewards_usd": epoch_rewards_usd,
            "total_rewards_usd": total_rewards_usd,
            "principal_usd": principal_usd,
            "roi_percentage": roi_percentage,
            "epochs": epochs
        }


def format_apr(apr: float) -> str:
    """Format APR for display.

    Args:
        apr: APR percentage

    Returns:
        Formatted string
    """
    if apr >= 1000:
        return f"{apr:,.0f}%"
    elif apr >= 100:
        return f"{apr:.1f}%"
    elif apr >= 10:
        return f"{apr:.2f}%"
    else:
        return f"{apr:.3f}%"


def format_usd(amount: float) -> str:
    """Format USD amount for display.

    Args:
        amount: USD amount

    Returns:
        Formatted string
    """
    if amount >= 1000000:
        return f"${amount/1000000:.2f}M"
    elif amount >= 1000:
        return f"${amount/1000:.1f}K"
    else:
        return f"${amount:.2f}"
