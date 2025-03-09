<<<<<<< HEAD
"""
Configuration module for Eduquest API.
Handles API keys, model settings, and other configurations.
"""

from typing import List, Dict, Any

class APIConfig:
    """API Configuration settings."""
    
    # Gemini API Keys
    GEMINI_API_KEYS: List[str] = [
        "AIzaSyAvdYKVRzTkbI-TxdmgcpQt3z4nHis1Ifg",
        "AIzaSyBxespyCjGQ1M73KQRDk0om_khk1M_z2pc",
        "AIzaSyCEP6y4h-kHFXRtsLHWcoLbYW7X1RZFe74",
        "AIzaSyBczbG7_7JJEEfrbh8QbGHTgGYpMKTzgA4"
    ]
    
    # Model settings
    DEFAULT_MODEL: str = "gemini-pro"
    MODEL_CONFIG: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 40,
        "max_output_tokens": 2048
    }
    
    # Cache settings
    CACHE_ENABLED: bool = True
    CACHE_DIR: str = ".cache"
    CACHE_TTL: int = 3600  # 1 hour
    
    # API settings
    KEY_STRATEGY: str = "adaptive"
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "logs/gemini.log"
    
    # Rate limiting
    RATE_LIMIT: Dict[str, Any] = {
        "requests_per_minute": 60,
        "tokens_per_minute": 60000
    }

    @classmethod
    def get_model_config(cls, **overrides) -> Dict[str, Any]:
        """
        Get model configuration with optional overrides.
        
        Args:
            **overrides: Key-value pairs to override default config
            
        Returns:
            Dictionary with model configuration
        """
        config = cls.MODEL_CONFIG.copy()
        config.update(overrides)
        return config

# Version info
__version__ = "1.0.0"
__author__ = "Sohom"
__description__ = "Eduquest API Configuration"

# Export APIConfig
=======
"""
Configuration module for Eduquest API.
Handles API keys, model settings, and other configurations.
"""

from typing import List, Dict, Any

class APIConfig:
    """API Configuration settings."""
    
    # Gemini API Keys
    GEMINI_API_KEYS: List[str] = [
        "AIzaSyAvdYKVRzTkbI-TxdmgcpQt3z4nHis1Ifg",
        "AIzaSyBxespyCjGQ1M73KQRDk0om_khk1M_z2pc",
        "AIzaSyCEP6y4h-kHFXRtsLHWcoLbYW7X1RZFe74",
        "AIzaSyBczbG7_7JJEEfrbh8QbGHTgGYpMKTzgA4"
    ]
    
    # Model settings
    DEFAULT_MODEL: str = "gemini-pro"
    MODEL_CONFIG: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 40,
        "max_output_tokens": 2048
    }
    
    # Cache settings
    CACHE_ENABLED: bool = True
    CACHE_DIR: str = ".cache"
    CACHE_TTL: int = 3600  # 1 hour
    
    # API settings
    KEY_STRATEGY: str = "adaptive"
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "logs/gemini.log"
    
    # Rate limiting
    RATE_LIMIT: Dict[str, Any] = {
        "requests_per_minute": 60,
        "tokens_per_minute": 60000
    }

    @classmethod
    def get_model_config(cls, **overrides) -> Dict[str, Any]:
        """
        Get model configuration with optional overrides.
        
        Args:
            **overrides: Key-value pairs to override default config
            
        Returns:
            Dictionary with model configuration
        """
        config = cls.MODEL_CONFIG.copy()
        config.update(overrides)
        return config

# Version info
__version__ = "1.0.0"
__author__ = "Sohom"
__description__ = "Eduquest API Configuration"

# Export APIConfig
>>>>>>> f41f548 (frontend)
__all__ = ["APIConfig"]