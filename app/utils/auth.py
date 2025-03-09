<<<<<<< HEAD
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from threading import Lock
import hashlib
import os

logger = logging.getLogger(__name__)

class APIKeyAuth:
    """Handles API key authentication and management with rate limiting and permissions."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the API key authentication system.
        
        Args:
            config_path: Optional path to JSON config file with API keys
        """
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._last_cleanup = time.time()
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        Load API keys and settings from JSON config file.
        
        Args:
            config_path: Path to JSON config file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            with self._lock:
                for key, details in config.get('api_keys', {}).items():
                    self._keys[key] = {
                        'owner': details.get('owner', 'unknown'),
                        'permissions': details.get('permissions', []),
                        'rate_limit': details.get('rate_limit', 0),
                        'calls': 0,
                        'last_call': None,
                        'created': datetime.now(),
                        'enabled': details.get('enabled', True)
                    }
            logger.info(f"Loaded {len(self._keys)} API keys from config")
            
        except Exception as e:
            logger.error(f"Error loading API key config: {str(e)}")
    
    def create_key(self, owner: str, permissions: List[str] = None, 
                  rate_limit: int = 0) -> str:
        """
        Create a new API key.
        
        Args:
            owner: Name of key owner
            permissions: List of granted permissions
            rate_limit: Maximum calls per minute (0 for unlimited)
            
        Returns:
            Newly created API key
        """
        # Generate random key
        key = hashlib.sha256(os.urandom(32)).hexdigest()[:32]
        
        with self._lock:
            self._keys[key] = {
                'owner': owner,
                'permissions': permissions or [],
                'rate_limit': rate_limit,
                'calls': 0,
                'last_call': None,
                'created': datetime.now(),
                'enabled': True
            }
        
        return key
    
    def validate_key(self, api_key: str, required_permission: Optional[str] = None) -> bool:
        """
        Validate an API key and optionally check for specific permission.
        
        Args:
            api_key: API key to validate
            required_permission: Optional permission to check for
            
        Returns:
            True if key is valid and has required permission (if specified)
        """
        with self._lock:
            if api_key not in self._keys:
                return False
                
            key_info = self._keys[api_key]
            
            # Check if key is enabled
            if not key_info['enabled']:
                return False
                
            # Check rate limit
            if key_info['rate_limit'] > 0:
                now = datetime.now()
                if key_info['last_call']:
                    # Reset calls if minute has passed
                    if (now - key_info['last_call']).total_seconds() >= 60:
                        key_info['calls'] = 0
                    # Check rate limit
                    elif key_info['calls'] >= key_info['rate_limit']:
                        return False
                
                key_info['calls'] += 1
                key_info['last_call'] = now
            
            # Check permission if required
            if required_permission:
                return required_permission in key_info['permissions']
            
            return True
    
    def get_key_info(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an API key.
        
        Args:
            api_key: API key to look up
            
        Returns:
            Dictionary with key information or None if key not found
        """
        with self._lock:
            if api_key in self._keys:
                info = dict(self._keys[api_key])
                # Convert datetime to string
                info['created'] = info['created'].isoformat()
                if info['last_call']:
                    info['last_call'] = info['last_call'].isoformat()
                return info
            return None
    
    def disable_key(self, api_key: str) -> bool:
        """
        Disable an API key.
        
        Args:
            api_key: API key to disable
            
        Returns:
            True if key was disabled, False if not found
        """
        with self._lock:
            if api_key in self._keys:
                self._keys[api_key]['enabled'] = False
                return True
            return False
    
    def enable_key(self, api_key: str) -> bool:
        """
        Enable an API key.
        
        Args:
            api_key: API key to enable
            
        Returns:
            True if key was enabled, False if not found
        """
        with self._lock:
            if api_key in self._keys:
                self._keys[api_key]['enabled'] = True
                return True
            return False
    
    def delete_key(self, api_key: str) -> bool:
        """
        Delete an API key.
        
        Args:
            api_key: API key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if api_key in self._keys:
                del self._keys[api_key]
                return True
            return False
    
    def cleanup(self) -> None:
        """Remove disabled keys that haven't been used in 30 days."""
        with self._lock:
            now = datetime.now()
            to_delete = []
            
            for key, info in self._keys.items():
                if not info['enabled']:
                    last_used = info['last_call'] or info['created']
                    if (now - last_used).days >= 30:
                        to_delete.append(key)
            
            for key in to_delete:
                del self._keys[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get authentication system statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        with self._lock:
            total_keys = len(self._keys)
            active_keys = sum(1 for info in self._keys.values() if info['enabled'])
            rate_limited_keys = sum(1 for info in self._keys.values() 
                                  if info['rate_limit'] > 0)
            
            return {
                'total_keys': total_keys,
                'active_keys': active_keys,
                'disabled_keys': total_keys - active_keys,
                'rate_limited_keys': rate_limited_keys
            }

# Example usage
if __name__ == "__main__":
    # Create auth instance
    auth = APIKeyAuth()
    
    # Create some test keys
    key1 = auth.create_key("user1", ["read", "write"], rate_limit=60)
    key2 = auth.create_key("user2", ["read"])
    
    # Validate keys
    print(auth.validate_key(key1))  # True
    print(auth.validate_key(key1, "write"))  # True
    print(auth.validate_key(key2, "write"))  # False
    
    # Get key info
    print(auth.get_key_info(key1))
    
    # Disable key
    auth.disable_key(key1)
    print(auth.validate_key(key1))  # False
    
    # Print stats
=======
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from threading import Lock
import hashlib
import os

logger = logging.getLogger(__name__)

class APIKeyAuth:
    """Handles API key authentication and management with rate limiting and permissions."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the API key authentication system.
        
        Args:
            config_path: Optional path to JSON config file with API keys
        """
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._last_cleanup = time.time()
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        Load API keys and settings from JSON config file.
        
        Args:
            config_path: Path to JSON config file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            with self._lock:
                for key, details in config.get('api_keys', {}).items():
                    self._keys[key] = {
                        'owner': details.get('owner', 'unknown'),
                        'permissions': details.get('permissions', []),
                        'rate_limit': details.get('rate_limit', 0),
                        'calls': 0,
                        'last_call': None,
                        'created': datetime.now(),
                        'enabled': details.get('enabled', True)
                    }
            logger.info(f"Loaded {len(self._keys)} API keys from config")
            
        except Exception as e:
            logger.error(f"Error loading API key config: {str(e)}")
    
    def create_key(self, owner: str, permissions: List[str] = None, 
                  rate_limit: int = 0) -> str:
        """
        Create a new API key.
        
        Args:
            owner: Name of key owner
            permissions: List of granted permissions
            rate_limit: Maximum calls per minute (0 for unlimited)
            
        Returns:
            Newly created API key
        """
        # Generate random key
        key = hashlib.sha256(os.urandom(32)).hexdigest()[:32]
        
        with self._lock:
            self._keys[key] = {
                'owner': owner,
                'permissions': permissions or [],
                'rate_limit': rate_limit,
                'calls': 0,
                'last_call': None,
                'created': datetime.now(),
                'enabled': True
            }
        
        return key
    
    def validate_key(self, api_key: str, required_permission: Optional[str] = None) -> bool:
        """
        Validate an API key and optionally check for specific permission.
        
        Args:
            api_key: API key to validate
            required_permission: Optional permission to check for
            
        Returns:
            True if key is valid and has required permission (if specified)
        """
        with self._lock:
            if api_key not in self._keys:
                return False
                
            key_info = self._keys[api_key]
            
            # Check if key is enabled
            if not key_info['enabled']:
                return False
                
            # Check rate limit
            if key_info['rate_limit'] > 0:
                now = datetime.now()
                if key_info['last_call']:
                    # Reset calls if minute has passed
                    if (now - key_info['last_call']).total_seconds() >= 60:
                        key_info['calls'] = 0
                    # Check rate limit
                    elif key_info['calls'] >= key_info['rate_limit']:
                        return False
                
                key_info['calls'] += 1
                key_info['last_call'] = now
            
            # Check permission if required
            if required_permission:
                return required_permission in key_info['permissions']
            
            return True
    
    def get_key_info(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an API key.
        
        Args:
            api_key: API key to look up
            
        Returns:
            Dictionary with key information or None if key not found
        """
        with self._lock:
            if api_key in self._keys:
                info = dict(self._keys[api_key])
                # Convert datetime to string
                info['created'] = info['created'].isoformat()
                if info['last_call']:
                    info['last_call'] = info['last_call'].isoformat()
                return info
            return None
    
    def disable_key(self, api_key: str) -> bool:
        """
        Disable an API key.
        
        Args:
            api_key: API key to disable
            
        Returns:
            True if key was disabled, False if not found
        """
        with self._lock:
            if api_key in self._keys:
                self._keys[api_key]['enabled'] = False
                return True
            return False
    
    def enable_key(self, api_key: str) -> bool:
        """
        Enable an API key.
        
        Args:
            api_key: API key to enable
            
        Returns:
            True if key was enabled, False if not found
        """
        with self._lock:
            if api_key in self._keys:
                self._keys[api_key]['enabled'] = True
                return True
            return False
    
    def delete_key(self, api_key: str) -> bool:
        """
        Delete an API key.
        
        Args:
            api_key: API key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if api_key in self._keys:
                del self._keys[api_key]
                return True
            return False
    
    def cleanup(self) -> None:
        """Remove disabled keys that haven't been used in 30 days."""
        with self._lock:
            now = datetime.now()
            to_delete = []
            
            for key, info in self._keys.items():
                if not info['enabled']:
                    last_used = info['last_call'] or info['created']
                    if (now - last_used).days >= 30:
                        to_delete.append(key)
            
            for key in to_delete:
                del self._keys[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get authentication system statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        with self._lock:
            total_keys = len(self._keys)
            active_keys = sum(1 for info in self._keys.values() if info['enabled'])
            rate_limited_keys = sum(1 for info in self._keys.values() 
                                  if info['rate_limit'] > 0)
            
            return {
                'total_keys': total_keys,
                'active_keys': active_keys,
                'disabled_keys': total_keys - active_keys,
                'rate_limited_keys': rate_limited_keys
            }

# Example usage
if __name__ == "__main__":
    # Create auth instance
    auth = APIKeyAuth()
    
    # Create some test keys
    key1 = auth.create_key("user1", ["read", "write"], rate_limit=60)
    key2 = auth.create_key("user2", ["read"])
    
    # Validate keys
    print(auth.validate_key(key1))  # True
    print(auth.validate_key(key1, "write"))  # True
    print(auth.validate_key(key2, "write"))  # False
    
    # Get key info
    print(auth.get_key_info(key1))
    
    # Disable key
    auth.disable_key(key1)
    print(auth.validate_key(key1))  # False
    
    # Print stats
>>>>>>> f41f548 (frontend)
    print(auth.get_stats())