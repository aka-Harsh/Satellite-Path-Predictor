import redis
import json
import pickle
from typing import Any, Optional, Union
from datetime import datetime, timedelta
import logging

from config.settings import Config

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis-based cache manager for satellite data"""
    
    def __init__(self):
        try:
            self.redis_client = redis.from_url(Config.REDIS_URL)
            # Test connection
            self.redis_client.ping()
            self.connected = True
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using memory cache.")
            self.connected = False
            self.memory_cache = {}
    
    def set(self, key: str, value: Any, timeout: int = None) -> bool:
        """
        Set a value in the cache
        
        Args:
            key: Cache key
            value: Value to cache
            timeout: Expiration time in seconds
            
        Returns:
            True if successful
        """
        if timeout is None:
            timeout = Config.CACHE_TIMEOUT
            
        try:
            if self.connected:
                # Use Redis
                serialized_value = json.dumps(value) if isinstance(value, (dict, list)) else pickle.dumps(value)
                return self.redis_client.setex(key, timeout, serialized_value)
            else:
                # Use memory cache
                expiry = datetime.utcnow() + timedelta(seconds=timeout)
                self.memory_cache[key] = {
                    'value': value,
                    'expiry': expiry
                }
                return True
                
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            if self.connected:
                # Use Redis
                cached_value = self.redis_client.get(key)
                if cached_value:
                    try:
                        return json.loads(cached_value)
                    except json.JSONDecodeError:
                        return pickle.loads(cached_value)
                return None
            else:
                # Use memory cache
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    if datetime.utcnow() < cache_entry['expiry']:
                        return cache_entry['value']
                    else:
                        # Remove expired entry
                        del self.memory_cache[key]
                return None
                
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful
        """
        try:
            if self.connected:
                return bool(self.redis_client.delete(key))
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists and is not expired
        """
        try:
            if self.connected:
                return bool(self.redis_client.exists(key))
            else:
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    if datetime.utcnow() < cache_entry['expiry']:
                        return True
                    else:
                        del self.memory_cache[key]
                return False
                
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern
        
        Args:
            pattern: Pattern to match (e.g., "tle_*")
            
        Returns:
            Number of keys deleted
        """
        try:
            if self.connected:
                keys = self.redis_client.keys(pattern)
                if keys:
                    return self.redis_client.delete(*keys)
                return 0
            else:
                deleted_count = 0
                import fnmatch
                keys_to_delete = []
                
                for key in self.memory_cache.keys():
                    if fnmatch.fnmatch(key, pattern):
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    del self.memory_cache[key]
                    deleted_count += 1
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error clearing pattern {pattern}: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary containing cache statistics
        """
        try:
            if self.connected:
                info = self.redis_client.info()
                return {
                    'connected': True,
                    'used_memory': info.get('used_memory_human'),
                    'total_keys': info.get('db0', {}).get('keys', 0),
                    'hits': info.get('keyspace_hits', 0),
                    'misses': info.get('keyspace_misses', 0)
                }
            else:
                # Clean expired entries first
                self._cleanup_memory_cache()
                return {
                    'connected': False,
                    'type': 'memory',
                    'total_keys': len(self.memory_cache)
                }
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'connected': False, 'error': str(e)}
    
    def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache"""
        try:
            current_time = datetime.utcnow()
            expired_keys = []
            
            for key, cache_entry in self.memory_cache.items():
                if current_time >= cache_entry['expiry']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
                
        except Exception as e:
            logger.error(f"Error cleaning memory cache: {e}")
    
    def set_json(self, key: str, value: dict, timeout: int = None) -> bool:
        """
        Set a JSON value in the cache
        
        Args:
            key: Cache key
            value: Dictionary to cache as JSON
            timeout: Expiration time in seconds
            
        Returns:
            True if successful
        """
        if not isinstance(value, dict):
            raise ValueError("Value must be a dictionary for JSON caching")
        
        return self.set(key, value, timeout)
    
    def get_json(self, key: str) -> Optional[dict]:
        """
        Get a JSON value from the cache
        
        Args:
            key: Cache key
            
        Returns:
            Dictionary or None if not found
        """
        value = self.get(key)
        if value and isinstance(value, dict):
            return value
        return None
    
    def cache_tle_data(self, norad_id: str, tle_data: dict) -> bool:
        """
        Cache TLE data for a satellite
        
        Args:
            norad_id: NORAD catalog number
            tle_data: TLE data dictionary
            
        Returns:
            True if successful
        """
        key = f"tle_{norad_id}"
        return self.set_json(key, tle_data, Config.TLE_CACHE_TIMEOUT)
    
    def get_cached_tle(self, norad_id: str) -> Optional[dict]:
        """
        Get cached TLE data for a satellite
        
        Args:
            norad_id: NORAD catalog number
            
        Returns:
            TLE data dictionary or None
        """
        key = f"tle_{norad_id}"
        return self.get_json(key)
    
    def cache_trajectory(self, cache_key: str, trajectory_data: dict, 
                        custom_timeout: int = None) -> bool:
        """
        Cache trajectory calculation results
        
        Args:
            cache_key: Unique cache key for the trajectory
            trajectory_data: Trajectory data dictionary
            custom_timeout: Custom timeout in seconds
            
        Returns:
            True if successful
        """
        timeout = custom_timeout or Config.CACHE_TIMEOUT
        return self.set_json(f"trajectory_{cache_key}", trajectory_data, timeout)
    
    def get_cached_trajectory(self, cache_key: str) -> Optional[dict]:
        """
        Get cached trajectory data
        
        Args:
            cache_key: Unique cache key for the trajectory
            
        Returns:
            Trajectory data dictionary or None
        """
        return self.get_json(f"trajectory_{cache_key}")
    
    def cache_prediction(self, prediction_key: str, prediction_data: dict) -> bool:
        """
        Cache prediction results
        
        Args:
            prediction_key: Unique prediction identifier
            prediction_data: Prediction data dictionary
            
        Returns:
            True if successful
        """
        # Predictions are cached for shorter time due to time sensitivity
        timeout = Config.CACHE_TIMEOUT // 2
        return self.set_json(f"prediction_{prediction_key}", prediction_data, timeout)
    
    def get_cached_prediction(self, prediction_key: str) -> Optional[dict]:
        """
        Get cached prediction data
        
        Args:
            prediction_key: Unique prediction identifier
            
        Returns:
            Prediction data dictionary or None
        """
        return self.get_json(f"prediction_{prediction_key}")
    
    def clear_all_cache(self) -> bool:
        """
        Clear all cached data
        
        Returns:
            True if successful
        """
        try:
            if self.connected:
                return bool(self.redis_client.flushdb())
            else:
                self.memory_cache.clear()
                return True
                
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
            return False
    
    def generate_cache_key(self, *args) -> str:
        """
        Generate a cache key from multiple arguments
        
        Args:
            *args: Arguments to include in the key
            
        Returns:
            Generated cache key
        """
        key_parts = []
        for arg in args:
            if isinstance(arg, (dict, list)):
                key_parts.append(str(hash(json.dumps(arg, sort_keys=True))))
            else:
                key_parts.append(str(arg))
        
        return "_".join(key_parts)