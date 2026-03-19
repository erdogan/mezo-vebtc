"""Contract fetcher for querying on-chain bribe and fee data."""
import json
from typing import Dict, List, Optional, Tuple, Any
from web3 import Web3
from web3.contract import Contract

from lib.fetchers.rpc_fetcher import RPCFetcher
from lib.analytics.epoch_tracker import EpochTracker


class ContractFetcher:
    """Fetch bribe and fee data from Voter and reward contracts."""

    def __init__(self, rpc_fetcher: RPCFetcher, abis_dir: str = "abis"):
        """Initialize contract fetcher.

        Args:
            rpc_fetcher: RPC fetcher instance
            abis_dir: Directory containing contract ABIs
        """
        self.rpc_fetcher = rpc_fetcher
        self.abis_dir = abis_dir
        self.epoch_tracker = EpochTracker()

        # Load ABIs
        self.voter_abi = self._load_abi("Voter.json")
        self.gauge_abi = self._load_abi("Gauge.json")
        self.bribe_abi = self._load_abi("BribeVotingReward.json")
        self.fees_abi = self._load_abi("FeesVotingReward.json")

    def _load_abi(self, filename: str) -> List[Dict[str, Any]]:
        """Load ABI from JSON file.

        Args:
            filename: Name of ABI file

        Returns:
            ABI as list of dictionaries
        """
        with open(f"{self.abis_dir}/{filename}", "r") as f:
            return json.load(f)

    def _retry(self, fn, *args, default=None, label=""):
        """Shorthand for RPC call with retry."""
        return self.rpc_fetcher.call_with_retry(fn, *args, default=default, label=label)

    def get_all_pools(self, voter_address: str) -> List[Tuple[str, str, str, str]]:
        """Get all pools with their gauge, bribe, and fees contracts.

        Args:
            voter_address: Voter contract address

        Returns:
            List of tuples: (pool_address, gauge_address, bribe_address, fees_address)
        """
        voter = self.rpc_fetcher.get_contract(voter_address, self.voter_abi)

        pools = []
        index = 0

        while True:
            try:
                pool_address = self._retry(
                    lambda idx=index: voter.functions.pools(idx).call(),
                    label=f"pools({index})"
                )
                if pool_address is None or pool_address == "0x0000000000000000000000000000000000000000":
                    break

                gauge_address = self._retry(
                    lambda pa=pool_address: voter.functions.gauges(pa).call(),
                    label=f"gauges({pool_address[:10]})"
                )
                if gauge_address is None:
                    index += 1
                    continue

                bribe_address = self._retry(
                    lambda ga=gauge_address: voter.functions.gaugeToBribe(ga).call(),
                    label=f"gaugeToBribe({gauge_address[:10]})"
                )
                fees_address = self._retry(
                    lambda ga=gauge_address: voter.functions.gaugeToFees(ga).call(),
                    label=f"gaugeToFees({gauge_address[:10]})"
                )

                pools.append((pool_address, gauge_address, bribe_address or "", fees_address or ""))
                index += 1

            except Exception as e:
                if "execution reverted" in str(e).lower() or "invalid" in str(e).lower():
                    break
                else:
                    print(f"Error fetching pool {index}: {e}")
                    break

        return pools

    def get_pool_weights(self, voter_address: str, pool_addresses: List[str]) -> Dict[str, float]:
        """Get voting weights for pools.

        Args:
            voter_address: Voter contract address
            pool_addresses: List of pool addresses

        Returns:
            Dictionary mapping pool address to voting weight
        """
        voter = self.rpc_fetcher.get_contract(voter_address, self.voter_abi)
        weights = {}

        for pool_address in pool_addresses:
            weight = self._retry(
                lambda pa=pool_address: voter.functions.weights(pa).call(),
                default=0,
                label=f"weights({pool_address[:10]})"
            )
            weights[pool_address] = float(weight) / 1e18 if weight else 0.0

        return weights

    def get_bribe_data(self, bribe_address: str, epoch_timestamp: int) -> Dict[str, Any]:
        """Get bribe data for a specific epoch.

        Args:
            bribe_address: Bribe contract address
            epoch_timestamp: Epoch start timestamp

        Returns:
            Dictionary with bribe data: {
                "tokens": ["0xToken1", "0xToken2"],
                "amounts": {"0xToken1": 1000.0, "0xToken2": 2000.0},
                "total_supply": 85.5
            }
        """
        if bribe_address == "0x0000000000000000000000000000000000000000":
            return {"tokens": [], "amounts": {}, "total_supply": 0}

        bribe = self.rpc_fetcher.get_contract(bribe_address, self.bribe_abi)

        # Get list of reward tokens
        reward_tokens = []
        rewards_length = self._retry(
            lambda: bribe.functions.rewardsListLength().call(),
            default=0,
            label=f"bribe rewardsListLength({bribe_address[:10]})"
        )
        for i in range(rewards_length or 0):
            token_address = self._retry(
                lambda idx=i: bribe.functions.rewards(idx).call(),
                label=f"bribe rewards({i})"
            )
            if token_address:
                reward_tokens.append(token_address)

        # Get amounts per token for this epoch
        amounts = {}
        for token in reward_tokens:
            amount_wei = self._retry(
                lambda t=token: bribe.functions.tokenRewardsPerEpoch(t, epoch_timestamp).call(),
                default=0,
                label=f"bribe tokenRewardsPerEpoch({token[:10]})"
            )
            amounts[token] = float(amount_wei) / 1e18 if amount_wei else 0.0

        # Get total supply (voting power in this bribe contract)
        supply_wei = self._retry(
            lambda: bribe.functions.totalSupply().call(),
            default=0,
            label=f"bribe totalSupply({bribe_address[:10]})"
        )
        total_supply = float(supply_wei) / 1e18 if supply_wei else 0.0

        return {
            "tokens": reward_tokens,
            "amounts": amounts,
            "total_supply": total_supply
        }

    def get_fees_data(self, fees_address: str, epoch_timestamp: int) -> Dict[str, Any]:
        """Get fees data for a specific epoch.

        Args:
            fees_address: Fees contract address
            epoch_timestamp: Epoch start timestamp

        Returns:
            Dictionary with fees data (same structure as bribe_data)
        """
        if fees_address == "0x0000000000000000000000000000000000000000":
            return {"tokens": [], "amounts": {}, "total_supply": 0}

        fees = self.rpc_fetcher.get_contract(fees_address, self.fees_abi)

        # Get list of reward tokens
        reward_tokens = []
        rewards_length = self._retry(
            lambda: fees.functions.rewardsListLength().call(),
            default=0,
            label=f"fees rewardsListLength({fees_address[:10]})"
        )
        for i in range(rewards_length or 0):
            token_address = self._retry(
                lambda idx=i: fees.functions.rewards(idx).call(),
                label=f"fees rewards({i})"
            )
            if token_address:
                reward_tokens.append(token_address)

        # Get amounts per token for this epoch
        amounts = {}
        for token in reward_tokens:
            amount_wei = self._retry(
                lambda t=token: fees.functions.tokenRewardsPerEpoch(t, epoch_timestamp).call(),
                default=0,
                label=f"fees tokenRewardsPerEpoch({token[:10]})"
            )
            amounts[token] = float(amount_wei) / 1e18 if amount_wei else 0.0

        # Get total supply
        supply_wei = self._retry(
            lambda: fees.functions.totalSupply().call(),
            default=0,
            label=f"fees totalSupply({fees_address[:10]})"
        )
        total_supply = float(supply_wei) / 1e18 if supply_wei else 0.0

        return {
            "tokens": reward_tokens,
            "amounts": amounts,
            "total_supply": total_supply
        }

    def get_all_pool_incentives(self, voter_address: str,
                                current_timestamp: int) -> List[Dict[str, Any]]:
        """Get incentive data for all pools.

        Args:
            voter_address: Voter contract address
            current_timestamp: Current timestamp for epoch calculation

        Returns:
            List of pool incentive dictionaries
        """
        # Get current epoch start
        epoch_info = self.epoch_tracker.get_current_epoch(current_timestamp)
        epoch_start = epoch_info.start_ts

        # Get all pools
        pools_data = self.get_all_pools(voter_address)
        print(f"Found {len(pools_data)} pools")

        # Get pool weights
        pool_addresses = [pool[0] for pool in pools_data]
        weights = self.get_pool_weights(voter_address, pool_addresses)

        # Fetch incentive data for each pool
        incentives = []
        for pool_address, gauge_address, bribe_address, fees_address in pools_data:
            pool_weight = weights.get(pool_address, 0.0)

            # Get bribe data
            bribe_data = self.get_bribe_data(bribe_address, epoch_start)

            # Get fees data
            fees_data = self.get_fees_data(fees_address, epoch_start)

            incentives.append({
                "pool_address": pool_address,
                "gauge_address": gauge_address,
                "bribe_address": bribe_address,
                "fees_address": fees_address,
                "voting_weight": pool_weight,
                "bribes": bribe_data,
                "fees": fees_data
            })

        return incentives

    def get_token_symbol(self, token_address: str) -> str:
        """Get ERC20 token symbol.

        Args:
            token_address: Token contract address

        Returns:
            Token symbol (e.g., "USDC", "WBTC")
        """
        erc20_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "symbol",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            }
        ]

        token = self.rpc_fetcher.get_contract(token_address, erc20_abi)
        symbol = self._retry(
            lambda: token.functions.symbol().call(),
            default=token_address[:8],
            label=f"symbol({token_address[:10]})"
        )
        return symbol

    def get_token_decimals(self, token_address: str) -> int:
        """Get ERC20 token decimals.

        Args:
            token_address: Token contract address

        Returns:
            Number of decimals (default 18)
        """
        erc20_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            }
        ]

        token = self.rpc_fetcher.get_contract(token_address, erc20_abi)
        decimals = self._retry(
            lambda: token.functions.decimals().call(),
            default=18,
            label=f"decimals({token_address[:10]})"
        )
        return decimals

    def get_pool_name(self, pool_address: str) -> str:
        """Get pool name as token pair (e.g., "MUSDT/USDC").

        Args:
            pool_address: Pool contract address

        Returns:
            Pool name as "TOKEN0/TOKEN1" or shortened address if query fails
        """
        pool_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "token0",
                "outputs": [{"name": "", "type": "address"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "token1",
                "outputs": [{"name": "", "type": "address"}],
                "type": "function"
            }
        ]

        pool = self.rpc_fetcher.get_contract(pool_address, pool_abi)

        token0_address = self._retry(
            lambda: pool.functions.token0().call(),
            label=f"token0({pool_address[:10]})"
        )
        token1_address = self._retry(
            lambda: pool.functions.token1().call(),
            label=f"token1({pool_address[:10]})"
        )

        if not token0_address or not token1_address:
            return f"{pool_address[:10]}..."

        symbol0 = self.get_token_symbol(token0_address)
        symbol1 = self.get_token_symbol(token1_address)

        return f"{symbol0}/{symbol1}"
