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

    def get_all_pools(self, voter_address: str) -> List[Tuple[str, str, str, str]]:
        """Get all pools with their gauge, bribe, and fees contracts.

        Args:
            voter_address: Voter contract address

        Returns:
            List of tuples: (pool_address, gauge_address, bribe_address, fees_address)
        """
        voter = self.rpc_fetcher.get_contract(voter_address, self.voter_abi)

        # Get all pools from the pools array
        pools = []
        index = 0

        try:
            while True:
                try:
                    pool_address = voter.functions.pools(index).call()
                    if pool_address == "0x0000000000000000000000000000000000000000":
                        break

                    # Get gauge address for this pool
                    gauge_address = voter.functions.gauges(pool_address).call()

                    # Get associated contracts
                    bribe_address = voter.functions.gaugeToBribe(gauge_address).call()
                    fees_address = voter.functions.gaugeToFees(gauge_address).call()

                    pools.append((pool_address, gauge_address, bribe_address, fees_address))
                    index += 1

                except Exception as e:
                    # Reached end of array or error
                    if "execution reverted" in str(e).lower() or "invalid" in str(e).lower():
                        break  # End of array
                    else:
                        print(f"Error fetching pool {index}: {e}")
                        break

        except Exception as e:
            print(f"Error fetching pools: {e}")

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
            try:
                weight = voter.functions.weights(pool_address).call()
                # Convert from wei to readable units (18 decimals)
                weights[pool_address] = float(weight) / 1e18
            except Exception as e:
                print(f"Error fetching weight for {pool_address}: {e}")
                weights[pool_address] = 0.0

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
        try:
            rewards_length = bribe.functions.rewardsListLength().call()
            for i in range(rewards_length):
                token_address = bribe.functions.rewards(i).call()
                reward_tokens.append(token_address)
        except Exception as e:
            print(f"Error fetching bribe tokens for {bribe_address}: {e}")

        # Get amounts per token for this epoch
        amounts = {}
        for token in reward_tokens:
            try:
                amount_wei = bribe.functions.tokenRewardsPerEpoch(token, epoch_timestamp).call()
                # Assume 18 decimals (adjust if needed based on token)
                amounts[token] = float(amount_wei) / 1e18
            except Exception as e:
                print(f"Error fetching bribe amount for token {token}: {e}")
                amounts[token] = 0.0

        # Get total supply (voting power in this bribe contract)
        total_supply = 0.0
        try:
            supply_wei = bribe.functions.totalSupply().call()
            total_supply = float(supply_wei) / 1e18
        except Exception as e:
            print(f"Error fetching bribe total supply: {e}")

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
        try:
            rewards_length = fees.functions.rewardsListLength().call()
            for i in range(rewards_length):
                token_address = fees.functions.rewards(i).call()
                reward_tokens.append(token_address)
        except Exception as e:
            print(f"Error fetching fee tokens for {fees_address}: {e}")

        # Get amounts per token for this epoch
        amounts = {}
        for token in reward_tokens:
            try:
                amount_wei = fees.functions.tokenRewardsPerEpoch(token, epoch_timestamp).call()
                # Assume 18 decimals (adjust if needed based on token)
                amounts[token] = float(amount_wei) / 1e18
            except Exception as e:
                print(f"Error fetching fee amount for token {token}: {e}")
                amounts[token] = 0.0

        # Get total supply
        total_supply = 0.0
        try:
            supply_wei = fees.functions.totalSupply().call()
            total_supply = float(supply_wei) / 1e18
        except Exception as e:
            print(f"Error fetching fees total supply: {e}")

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
        # Minimal ERC20 ABI for symbol()
        erc20_abi = [
            {
                "constant": True,
                "inputs": [],
                "name": "symbol",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            }
        ]

        try:
            token = self.rpc_fetcher.get_contract(token_address, erc20_abi)
            symbol = token.functions.symbol().call()
            return symbol
        except Exception as e:
            print(f"Error fetching symbol for {token_address}: {e}")
            return token_address[:8]  # Return short address as fallback

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

        try:
            token = self.rpc_fetcher.get_contract(token_address, erc20_abi)
            decimals = token.functions.decimals().call()
            return decimals
        except Exception as e:
            print(f"Error fetching decimals for {token_address}: {e}")
            return 18  # Default to 18 decimals

    def get_pool_name(self, pool_address: str) -> str:
        """Get pool name as token pair (e.g., "MUSDT/USDC").

        Args:
            pool_address: Pool contract address

        Returns:
            Pool name as "TOKEN0/TOKEN1" or shortened address if query fails
        """
        # Pool ABI for token0() and token1()
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

        try:
            pool = self.rpc_fetcher.get_contract(pool_address, pool_abi)

            # Get token addresses
            token0_address = pool.functions.token0().call()
            token1_address = pool.functions.token1().call()

            # Get token symbols
            symbol0 = self.get_token_symbol(token0_address)
            symbol1 = self.get_token_symbol(token1_address)

            return f"{symbol0}/{symbol1}"

        except Exception as e:
            print(f"Error fetching pool name for {pool_address}: {e}")
            # Return shortened address as fallback
            return f"{pool_address[:10]}..."
