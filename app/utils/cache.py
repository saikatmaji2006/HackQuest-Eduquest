<<<<<<< HEAD
import time
from typing import Any, Dict, Optional, Tuple
from threading import Lock
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Cache:
    """
    Thread-safe cache implementation with TTL support and statistics tracking.
    """
    
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        """
        Initialize cache with default TTL and max size.
        
        Args:
            default_ttl: Default time-to-live in seconds for cache items
            max_size: Maximum number of items to store in cache
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}  # (value, expiry_time)
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._lock = Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._last_cleanup = time.time()
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if it exists and hasn't expired.
        
        Args:
            key: Cache key to lookup
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key in self._cache:
                value, expiry_time = self._cache[key]
                if time.time() < expiry_time:
                    self._hits += 1
                    return value
                else:
                    # Remove expired item
                    del self._cache[key]
                    
            self._misses += 1
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            # Check if we need to evict items
            if len(self._cache) >= self._max_size:
                self._evict()
                
            expiry_time = time.time() + (ttl if ttl is not None else self._default_ttl)
            self._cache[key] = (value, expiry_time)
            
    def delete(self, key: str) -> bool:
        """
        Delete item from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if item was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
            
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            
    def _evict(self) -> None:
        """Evict expired items and if needed, oldest items to maintain max size."""
        now = time.time()
        # Remove expired items
        expired = [k for k, (_, exp) in self._cache.items() if exp <= now]
        for key in expired:
            del self._cache[key]
            self._evictions += 1
            
        # If still too big, remove oldest items
        if len(self._cache) >= self._max_size:
            sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
            to_remove = len(self._cache) - self._max_size + 1
            for key, _ in sorted_items[:to_remove]:
                del self._cache[key]
                self._evictions += 1
                
    def cleanup(self) -> int:
        """
        Remove all expired items from cache.
        
        Returns:
            Number of items removed
        """
        with self._lock:
            initial_size = len(self._cache)
            now = time.time()
            expired = [k for k, (_, exp) in self._cache.items() if exp <= now]
            for key in expired:
                del self._cache[key]
            return initial_size - len(self._cache)
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': f"{hit_rate:.1f}%",
                'evictions': self._evictions,
                'default_ttl': self._default_ttl
            }
            
    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self._cache)
        
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache and hasn't expired."""
        with self._lock:
            if key in self._cache:
                _, expiry_time = self._cache[key]
                return time.time() < expiry_time
            return False
            
    def __str__(self) -> str:
        """Return string representation of cache stats."""
        stats = self.get_stats()
        return f"Cache(size={stats['size']}/{stats['max_size']}, hit_rate={stats['hit_rate']})"

# Example usage
if __name__ == "__main__":
    # Create cache with 5 second TTL
    cache = Cache(default_ttl=5, max_size=100)
    
    # Set some values
    cache.set("key1", "value1")
    cache.set("key2", "value2", ttl=10)  # Custom TTL
    
    # Get values
    print(cache.get("key1"))  # -> "value1"
    print(cache.get("key2"))  # -> "value2"
    print(cache.get("key3"))  # -> None
    
    # Wait for expiration
    time.sleep(6)
    print(cache.get("key1"))  # -> None (expired)
    print(cache.get("key2"))  # -> "value2" (still valid)
    
    # Print stats
=======
import time
from typing import Any, Dict, Optional, Tuple
from threading import Lock
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Cache:
    """
    Thread-safe cache implementation with TTL support and statistics tracking.
    """
    
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        """
        Initialize cache with default TTL and max size.
        
        Args:
            default_ttl: Default time-to-live in seconds for cache items
            max_size: Maximum number of items to store in cache
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}  # (value, expiry_time)
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._lock = Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._last_cleanup = time.time()
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if it exists and hasn't expired.
        
        Args:
            key: Cache key to lookup
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key in self._cache:
                value, expiry_time = self._cache[key]
                if time.time() < expiry_time:
                    self._hits += 1
                    return value
                else:
                    # Remove expired item
                    del self._cache[key]
                    
            self._misses += 1
            return None
            
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            # Check if we need to evict items
            if len(self._cache) >= self._max_size:
                self._evict()
                
            expiry_time = time.time() + (ttl if ttl is not None else self._default_ttl)
            self._cache[key] = (value, expiry_time)
            
    def delete(self, key: str) -> bool:
        """
        Delete item from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if item was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
            
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            
    def _evict(self) -> None:
        """Evict expired items and if needed, oldest items to maintain max size."""
        now = time.time()
        # Remove expired items
        expired = [k for k, (_, exp) in self._cache.items() if exp <= now]
        for key in expired:
            del self._cache[key]
            self._evictions += 1
            
        # If still too big, remove oldest items
        if len(self._cache) >= self._max_size:
            sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
            to_remove = len(self._cache) - self._max_size + 1
            for key, _ in sorted_items[:to_remove]:
                del self._cache[key]
                self._evictions += 1
                
    def cleanup(self) -> int:
        """
        Remove all expired items from cache.
        
        Returns:
            Number of items removed
        """
        with self._lock:
            initial_size = len(self._cache)
            now = time.time()
            expired = [k for k, (_, exp) in self._cache.items() if exp <= now]
            for key in expired:
                del self._cache[key]
            return initial_size - len(self._cache)
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': f"{hit_rate:.1f}%",
                'evictions': self._evictions,
                'default_ttl': self._default_ttl
            }
            
    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self._cache)
        
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache and hasn't expired."""
        with self._lock:
            if key in self._cache:
                _, expiry_time = self._cache[key]
                return time.time() < expiry_time
            return False
            
    def __str__(self) -> str:
        """Return string representation of cache stats."""
        stats = self.get_stats()
        return f"Cache(size={stats['size']}/{stats['max_size']}, hit_rate={stats['hit_rate']})"

# Example usage
if __name__ == "__main__":
    # Create cache with 5 second TTL
    cache = Cache(default_ttl=5, max_size=100)
    
    # Set some values
    cache.set("key1", "value1")
    cache.set("key2", "value2", ttl=10)  # Custom TTL
    
    # Get values
    print(cache.get("key1"))  # -> "value1"
    print(cache.get("key2"))  # -> "value2"
    print(cache.get("key3"))  # -> None
    
    # Wait for expiration
    time.sleep(6)
    print(cache.get("key1"))  # -> None (expired)
    print(cache.get("key2"))  # -> "value2" (still valid)
    
    # Print stats
>>>>>>> f41f548 (frontend)
    print(cache.get_stats())