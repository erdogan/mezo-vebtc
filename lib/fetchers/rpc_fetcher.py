"""RPC fetcher for on-chain contract queries."""
import time
from typing import Any, Optional, Tuple
from web3 import Web3
from web3.exceptions import Web3Exception


class RPCFetcher:
    """Handles RPC calls to blockchain with retry logic and caching."""

    def __init__(self, rpc_url: str, fallback_rpcs: list = None, retry_count: int = 3, timeout: int = 10):
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

    def is_connected(self) -> bool:
        """Check if connected to RPC.

        Returns:
            True if connected
        """
        try:
            return self.w3.is_connected()
        except Exception:
            return False

    def call_contract_function(self, contract_function, *args, default: Any = None) -> Tuple[Any, bool]:
        """Call a contract function with retry logic and fallbacks.

        Args:
            contract_function: Web3 contract function
            *args: Function arguments
            default: Default value on failure

        Returns:
            Tuple of (result, is_fresh) where is_fresh indicates if data is current
        """
        rpcs = [self.primary_rpc] + self.fallback_rpcs

        for rpc_url in rpcs:
            # Switch to different RPC if needed
            if self.w3.provider.endpoint_uri != rpc_url:
                self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': self.timeout}))

            # Try calling with retries
            for attempt in range(self.retry_count):
                try:
                    result = contract_function(*args).call(
                        block_identifier='latest',
                        timeout=self.timeout
                    )
                    return result, True  # Success, fresh data

                except Web3Exception as e:
                    print(f"RPC {rpc_url} attempt {attempt + 1} failed: {e}")
                    if attempt < self.retry_count - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    continue

                except Exception as e:
                    print(f"Unexpected error calling {rpc_url}: {e}")
                    break  # Don't retry on unexpected errors

        # All RPCs failed
        print(f"All RPC calls failed, returning default value")
        return default, False  # Stale or default data

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
