"""Cache manager for file-based caching with TTL."""
import json
import os
import time
from typing import Any, Optional


class CacheManager:
    """Manages file-based cache with TTL expiration."""

    def __init__(self, cache_dir: str = "cache"):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str) -> str:
        """Get file path for cache key.

        Args:
            key: Cache key

        Returns:
            Path to cache file
        """
        # Sanitize key for filename
        safe_key = key.replace("/", "_").replace(":", "_")
        return os.path.join(self.cache_dir, f"{safe_key}.json")

    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """Get value from cache if not expired.

        Args:
            key: Cache key
            ttl: Time to live in seconds (None = no expiration check)

        Returns:
            Cached value or None if expired/missing
        """
        cache_path = self._get_cache_path(key)

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)

            timestamp = data.get("timestamp", 0)
            value = data.get("value")

            # Check TTL if specified
            if ttl is not None:
                age = time.time() - timestamp
                if age > ttl:
                    # Expired
                    return None

            return value

        except Exception as e:
            print(f"Error reading cache {key}: {e}")
            return None

    def set(self, key: str, value: Any) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)

        Returns:
            True if successful
        """
        cache_path = self._get_cache_path(key)

        try:
            data = {
                "timestamp": time.time(),
                "value": value
            }

            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            print(f"Error writing cache {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete cache entry.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        cache_path = self._get_cache_path(key)

        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return True
        except Exception as e:
            print(f"Error deleting cache {key}: {e}")
            return False

    def clear_expired(self, ttl: int) -> int:
        """Clear all expired cache entries.

        Args:
            ttl: Time to live in seconds

        Returns:
            Number of entries cleared
        """
        cleared = 0

        try:
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue

                cache_path = os.path.join(self.cache_dir, filename)

                with open(cache_path, 'r') as f:
                    data = json.load(f)

                timestamp = data.get("timestamp", 0)
                age = time.time() - timestamp

                if age > ttl:
                    os.remove(cache_path)
                    cleared += 1

        except Exception as e:
            print(f"Error clearing expired cache: {e}")

        return cleared

    def clear_all(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        cleared = 0

        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
                    cleared += 1
        except Exception as e:
            print(f"Error clearing cache: {e}")

        return cleared
