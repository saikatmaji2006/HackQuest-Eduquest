<<<<<<< HEAD
import os
import json
import logging
from datetime import timedelta
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Base configuration."""
    API_VERSION = 'v1'
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    # API Settings
    API_KEYS = {
        'test': os.environ.get('TEST_API_KEY', 'test-key'),
        'default': os.environ.get('DEFAULT_API_KEY', 'your-api-key-here'),
        'admin': os.environ.get('ADMIN_API_KEY', 'your-admin-key-here')
    }
    
    # CORS Settings
    CORS_ORIGINS = [
        'http://localhost:3000',
        'http://127.0.0.1:3000'
    ]
    
    # MongoDB settings
    MONGODB_SETTINGS = {
        'host': 'mongodb://localhost:27017/eduquest',
        'db': 'eduquest',
        'connect': False  # Defer connection until first use
    }
    
    # API settings
    API_PREFIX = '/api/v1'
    API_KEY_HEADER = 'X-API-Key'
    API_RATE_LIMIT = '100/hour'
    
    # JWT settings
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'your-jwt-secret-key-here')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # Cache settings
    CACHE_TYPE = "simple"
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Logging settings
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'logs/eduquest.log'
    LOG_LEVEL = logging.INFO
    
    # Gemini API keys
    GEMINI_API_KEYS = [
        "AIzaSyAvdYKVRzTkbI-TxdmgcpQt3z4nHis1Ifg",
        "AIzaSyBxespyCjGQ1M73KQRDk0om_khk1M_z2pc", 
        "AIzaSyCEP6y4h-kHFXRtsLHWcoLbYW7X1RZFe74",
        "AIzaSyBczbG7_7JJEEfrbh8QbGHTgGYpMKTzgA4"
    ]
    
    @classmethod
    def load_config(cls, config_file: Optional[str] = None) -> None:
        """Load configuration from JSON file."""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {str(e)}")
    
    @classmethod
    def init_app(cls, app):
        """Initialize application with this configuration."""
        pass

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    DEVELOPMENT = True
    ENV = 'development'
    
    # Development Server
    HOST = '0.0.0.0'
    PORT = 5000

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    MONGODB_SETTINGS = {
        'host': 'mongodb://localhost:27017/eduquest_test',
        'db': 'eduquest_test',
        'connect': False
    }

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    MONGODB_SETTINGS = {
        'host': 'mongodb://user:pass@your-mongodb-host:27017/eduquest',
        'db': 'eduquest',
        'connect': False
    }
    
    @classmethod
    def init_app(cls, app):
        # Production-specific initialization
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = logging.FileHandler('logs/eduquest.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(cls.LOG_FORMAT))
        app.logger.addHandler(file_handler)

# Configuration dictionary
config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}

def get_config(config_name: str = "default") -> Config:
    """Get configuration class by name."""
    return config.get(config_name, config["default"])

# Initialize configuration
=======
import os
import json
import logging
from datetime import timedelta
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Base configuration."""
    API_VERSION = 'v1'
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    # API Settings
    API_KEYS = {
        'test': os.environ.get('TEST_API_KEY', 'test-key'),
        'default': os.environ.get('DEFAULT_API_KEY', 'your-api-key-here'),
        'admin': os.environ.get('ADMIN_API_KEY', 'your-admin-key-here')
    }
    
    # CORS Settings
    CORS_ORIGINS = [
        'http://localhost:3000',
        'http://127.0.0.1:3000'
    ]
    
    # MongoDB settings
    MONGODB_SETTINGS = {
        'host': 'mongodb://localhost:27017/eduquest',
        'db': 'eduquest',
        'connect': False  # Defer connection until first use
    }
    
    # API settings
    API_PREFIX = '/api/v1'
    API_KEY_HEADER = 'X-API-Key'
    API_RATE_LIMIT = '100/hour'
    
    # JWT settings
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'your-jwt-secret-key-here')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # Cache settings
    CACHE_TYPE = "simple"
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Logging settings
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'logs/eduquest.log'
    LOG_LEVEL = logging.INFO
    
    # Gemini API keys
    GEMINI_API_KEYS = [
        "AIzaSyAvdYKVRzTkbI-TxdmgcpQt3z4nHis1Ifg",
        "AIzaSyBxespyCjGQ1M73KQRDk0om_khk1M_z2pc", 
        "AIzaSyCEP6y4h-kHFXRtsLHWcoLbYW7X1RZFe74",
        "AIzaSyBczbG7_7JJEEfrbh8QbGHTgGYpMKTzgA4"
    ]
    
    @classmethod
    def load_config(cls, config_file: Optional[str] = None) -> None:
        """Load configuration from JSON file."""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {str(e)}")
    
    @classmethod
    def init_app(cls, app):
        """Initialize application with this configuration."""
        pass

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    DEVELOPMENT = True
    ENV = 'development'
    
    # Development Server
    HOST = '0.0.0.0'
    PORT = 5000

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    MONGODB_SETTINGS = {
        'host': 'mongodb://localhost:27017/eduquest_test',
        'db': 'eduquest_test',
        'connect': False
    }

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    MONGODB_SETTINGS = {
        'host': 'mongodb://user:pass@your-mongodb-host:27017/eduquest',
        'db': 'eduquest',
        'connect': False
    }
    
    @classmethod
    def init_app(cls, app):
        # Production-specific initialization
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = logging.FileHandler('logs/eduquest.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(cls.LOG_FORMAT))
        app.logger.addHandler(file_handler)

# Configuration dictionary
config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}

def get_config(config_name: str = "default") -> Config:
    """Get configuration class by name."""
    return config.get(config_name, config["default"])

# Initialize configuration
>>>>>>> f41f548 (frontend)
current_config = get_config()