"""RPC fetcher for on-chain contract queries."""
import random
import time
from typing import Any, Callable, Optional, Tuple
from web3 import Web3
from web3.exceptions import Web3Exception


# Errors indicating rate limiting
_RATE_LIMIT_PHRASES = ("429", "too many requests", "rate limit")


def _is_rate_limited(exc: Exception) -> bool:
    """Check if an exception is a rate-limit (429) error."""
    msg = str(exc).lower()
    return any(phrase in msg for phrase in _RATE_LIMIT_PHRASES)


class RPCFetcher:
    """Handles RPC calls to blockchain with retry logic and caching."""

    def __init__(self, rpc_url: str, fallback_rpcs: list = None, retry_count: int = 5, timeout: int = 15):
        """Initialize RPC fetcher.

        Args:
            rpc_url: Primary RPC endpoint URL
            fallback_rpcs: List of fallback RPC URLs
            retry_count: Number of retry attempts per RPC
            timeout: Request timeout in seconds
        """
        self.primary_rpc = rpc_url
        self.fallback_rpcs = fallback_rpcs or []
        self.retry_count = retry_count
        self.timeout = timeout
        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': timeout}))
        self._last_call_time = 0.0
        self._min_call_interval = 0.25  # Minimum seconds between RPC calls

    def _throttle(self):
        """Enforce minimum interval between RPC calls to avoid rate limiting."""
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._min_call_interval:
            time.sleep(self._min_call_interval - elapsed)
        self._last_call_time = time.time()

    def is_connected(self) -> bool:
        """Check if connected to RPC.

        Returns:
            True if connected
        """
        try:
            return self.w3.is_connected()
        except Exception:
            return False

    def call_with_retry(self, fn: Callable, *args, default: Any = None, label: str = "") -> Any:
        """Call any callable with retry logic and 429-aware exponential backoff.

        This is the primary method for making RPC calls. All contract_fetcher
        calls should use this to get automatic rate-limit handling.

        Args:
            fn: Callable to execute (e.g., contract.functions.pools(0).call)
            *args: Arguments to pass to fn
            default: Value to return if all retries fail
            label: Description for log messages

        Returns:
            Result of fn(*args) on success, or default on failure
        """
        for attempt in range(self.retry_count):
            try:
                self._throttle()
                result = fn(*args)
                return result
            except Exception as e:
                if _is_rate_limited(e):
                    # 429: use longer backoff with jitter
                    delay = min(2 ** (attempt + 1) + random.uniform(0, 1), 30)
                    if label:
                        print(f"Rate limited on {label}, retry {attempt + 1}/{self.retry_count} in {delay:.1f}s")
                    else:
                        print(f"Rate limited, retry {attempt + 1}/{self.retry_count} in {delay:.1f}s")
                    time.sleep(delay)
                elif "execution reverted" in str(e).lower():
                    # Contract revert — no point retrying
                    raise
                else:
                    # Other errors: shorter backoff
                    delay = 2 ** attempt + random.uniform(0, 0.5)
                    if attempt < self.retry_count - 1:
                        if label:
                            print(f"Error on {label}: {e}, retry {attempt + 1}/{self.retry_count} in {delay:.1f}s")
                        time.sleep(delay)
                    else:
                        if label:
                            print(f"Failed {label} after {self.retry_count} attempts: {e}")

        return default

    def call_contract_function(self, contract_function, *args, default: Any = None) -> Tuple[Any, bool]:
        """Call a contract function with retry logic and fallbacks.

        Args:
            contract_function: Web3 contract function
            *args: Function arguments
            default: Default value on failure

        Returns:
            Tuple of (result, is_fresh) where is_fresh indicates if data is current
        """
        result = self.call_with_retry(
            lambda: contract_function(*args).call(block_identifier='latest', timeout=self.timeout),
            default=None,
            label=str(contract_function)
        )
        if result is not None:
            return result, True
        return default, False

    def get_block_number(self) -> Optional[int]:
        """Get current block number.

        Returns:
            Block number or None on failure
        """
        try:
            return self.w3.eth.block_number
        except Exception as e:
            print(f"Error getting block number: {e}")
            return None

    def get_block_timestamp(self, block_number: Optional[int] = None) -> Optional[int]:
        """Get block timestamp.

        Args:
            block_number: Block number (None for latest)

        Returns:
            Block timestamp or None on failure
        """
        try:
            block = self.w3.eth.get_block(block_number or 'latest')
            return block['timestamp']
        except Exception as e:
            print(f"Error getting block timestamp: {e}")
            return None

    def get_contract(self, address: str, abi: list):
        """Get a Web3 contract instance.

        Args:
            address: Contract address
            abi: Contract ABI as list

        Returns:
            Web3 Contract instance
        """
        checksum_address = Web3.to_checksum_address(address)
        return self.w3.eth.contract(address=checksum_address, abi=abi)
