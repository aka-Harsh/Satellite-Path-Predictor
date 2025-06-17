import os
from datetime import timedelta

class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'orbital-tracker-secret-key-2024'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database Configuration
    SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'satellites.db')
    
    # Redis Configuration
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    CACHE_TIMEOUT = 3600  # 1 hour
    TLE_CACHE_TIMEOUT = 1800  # 30 minutes
    
    # API Configuration
    CELESTRAK_BASE_URL = 'https://celestrak.org'
    SPACETRACK_BASE_URL = 'https://www.space-track.org'
    
    # Space-Track.org credentials (optional)
    SPACETRACK_USERNAME = os.environ.get('SPACETRACK_USERNAME')
    SPACETRACK_PASSWORD = os.environ.get('SPACETRACK_PASSWORD')
    
    # Prediction Configuration
    DEFAULT_PREDICTION_HOURS = 24
    MAX_PREDICTION_HOURS = 168  # 7 days
    ORBITAL_CALCULATION_STEP_MINUTES = 10
    
    # Model Configuration
    LSTM_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'trained_models')
    SGP4_FALLBACK_ENABLED = True
    
    # Popular Satellites Configuration
    POPULAR_SATELLITES_UPDATE_INTERVAL = timedelta(hours=6)
    
    # File Paths
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    SATELLITES_JSON_PATH = os.path.join(DATA_DIR, 'satellites.json')
    TLE_CACHE_DIR = os.path.join(DATA_DIR, 'tle_cache')
    
    # Rate Limiting
    API_RATE_LIMIT = 100  # requests per minute
    
    # Error Handling
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'orbital_tracker.log')

class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    
class TestingConfig(Config):
    TESTING = True
    SQLITE_DB_PATH = ':memory:'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}