"""Token price fetcher using CoinGecko API."""
import time
from typing import Dict, Optional, List
import requests

from lib.fetchers.cache_manager import CacheManager


class PriceFetcher:
    """Fetch token prices from CoinGecko with caching."""

    # CoinGecko API endpoints
    COINGECKO_API = "https://api.coingecko.com/api/v3"
    SIMPLE_PRICE_ENDPOINT = f"{COINGECKO_API}/simple/price"

    # Token ID mappings (CoinGecko ID -> symbol)
    TOKEN_IDS = {
        "bitcoin": "BTC",
        "usd-coin": "USDC",
        "tether": "USDT",
        "wrapped-bitcoin": "WBTC",
        "ethereum": "ETH",
        "mezo": "MEZO"
    }

    # Reverse mapping (symbol -> CoinGecko ID)
    SYMBOL_TO_ID = {v: k for k, v in TOKEN_IDS.items()}

    # Cache TTL: 5 minutes (prices don't change that frequently)
    CACHE_TTL = 300

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize price fetcher.

        Args:
            cache_manager: Optional cache manager (creates new one if not provided)
        """
        self.cache_manager = cache_manager or CacheManager()
        self.last_request_time = 0
        self.min_request_interval = 1.5  # Rate limit: max 1 request per 1.5s

    def get_prices(self, symbols: List[str], vs_currency: str = "usd") -> Dict[str, float]:
        """Get current prices for multiple tokens.

        Args:
            symbols: List of token symbols (e.g., ["BTC", "USDC", "ETH"])
            vs_currency: Currency to price against (default: "usd")

        Returns:
            Dictionary mapping symbol to price
            Example: {"BTC": 42000.0, "USDC": 1.0, "ETH": 2200.0}
        """
        # Try cache first
        cache_key = f"prices_{vs_currency}_{'_'.join(sorted(symbols))}"
        cached_prices = self.cache_manager.get(cache_key, ttl=self.CACHE_TTL)

        if cached_prices is not None:
            return cached_prices

        # Convert symbols to CoinGecko IDs
        token_ids = []
        for symbol in symbols:
            token_id = self.SYMBOL_TO_ID.get(symbol.upper())
            if token_id:
                token_ids.append(token_id)
            else:
                print(f"Warning: Unknown token symbol '{symbol}', skipping")

        if not token_ids:
            return {}

        # Fetch from API
        try:
            prices = self._fetch_prices_from_api(token_ids, vs_currency)

            # Convert back to symbol keys
            result = {}
            for token_id, price in prices.items():
                symbol = self.TOKEN_IDS.get(token_id)
                if symbol:
                    result[symbol] = price

            # Cache the result
            if result:
                self.cache_manager.set(cache_key, result)

            return result

        except Exception as e:
            print(f"Error fetching prices: {e}")
            # Return cached data even if expired, if available
            cached_prices = self.cache_manager.get(cache_key, ttl=None)
            if cached_prices is not None:
                print("Using expired cache data")
                return cached_prices
            return {}

    def get_price(self, symbol: str, vs_currency: str = "usd") -> Optional[float]:
        """Get price for a single token.

        Args:
            symbol: Token symbol (e.g., "BTC", "USDC")
            vs_currency: Currency to price against (default: "usd")

        Returns:
            Price as float, or None if not available
        """
        prices = self.get_prices([symbol], vs_currency)
        return prices.get(symbol.upper())

    def _fetch_prices_from_api(self, token_ids: List[str],
                               vs_currency: str = "usd") -> Dict[str, float]:
        """Fetch prices from CoinGecko API.

        Args:
            token_ids: List of CoinGecko token IDs
            vs_currency: Currency to price against

        Returns:
            Dictionary mapping token_id to price
        """
        # Rate limiting
        now = time.time()
        time_since_last_request = now - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)

        # Build API request
        params = {
            "ids": ",".join(token_ids),
            "vs_currencies": vs_currency
        }

        # Make request
        self.last_request_time = time.time()
        response = requests.get(self.SIMPLE_PRICE_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Extract prices
        prices = {}
        for token_id in token_ids:
            if token_id in data and vs_currency in data[token_id]:
                prices[token_id] = float(data[token_id][vs_currency])

        return prices

    def get_all_supported_tokens(self) -> Dict[str, str]:
        """Get mapping of all supported token symbols to CoinGecko IDs.

        Returns:
            Dictionary mapping symbol to CoinGecko ID
            Example: {"BTC": "bitcoin", "USDC": "usd-coin"}
        """
        return self.SYMBOL_TO_ID.copy()

    def add_token(self, symbol: str, coingecko_id: str) -> None:
        """Add a new token to the supported list.

        Args:
            symbol: Token symbol (e.g., "MEZO")
            coingecko_id: CoinGecko token ID (e.g., "mezo")
        """
        self.SYMBOL_TO_ID[symbol.upper()] = coingecko_id
        self.TOKEN_IDS[coingecko_id] = symbol.upper()
        print(f"Added token: {symbol} -> {coingecko_id}")


def get_token_prices(symbols: List[str], cache_manager: Optional[CacheManager] = None) -> Dict[str, float]:
    """Convenience function to get token prices.

    Args:
        symbols: List of token symbols
        cache_manager: Optional cache manager

    Returns:
        Dictionary mapping symbol to USD price
    """
    fetcher = PriceFetcher(cache_manager)
    return fetcher.get_prices(symbols)


def get_token_price(symbol: str, cache_manager: Optional[CacheManager] = None) -> Optional[float]:
    """Convenience function to get a single token price.

    Args:
        symbol: Token symbol
        cache_manager: Optional cache manager

    Returns:
        Price in USD, or None if not available
    """
    fetcher = PriceFetcher(cache_manager)
    return fetcher.get_price(symbol)
