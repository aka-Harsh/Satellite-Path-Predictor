import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class APIKeys:
    """Centralized API key management"""
    
    # Space-Track.org API credentials (optional but recommended)
    SPACETRACK_USERNAME = os.environ.get('SPACETRACK_USERNAME', '')
    SPACETRACK_PASSWORD = os.environ.get('SPACETRACK_PASSWORD', '')
    
    # N2YO API key (optional, for enhanced tracking)
    N2YO_API_KEY = os.environ.get('N2YO_API_KEY', '')
    
    # Future API integrations
    CELESTRAK_API_KEY = os.environ.get('CELESTRAK_API_KEY', '')  # Free, no key needed currently
    
    # Redis credentials (if using cloud Redis)
    REDIS_USERNAME = os.environ.get('REDIS_USERNAME', '')
    REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', '')
    
    # Database credentials (if using external database)
    DATABASE_URL = os.environ.get('DATABASE_URL', '')
    
    # External service APIs (future integrations)
    WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY', '')
    GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', '')
    
    @classmethod
    def validate_required_keys(cls):
        """Validate that required API keys are present"""
        missing_keys = []
        warnings = []
        
        # Check optional but recommended keys
        if not cls.SPACETRACK_USERNAME or not cls.SPACETRACK_PASSWORD:
            warnings.append("Space-Track.org credentials not found. Some features may be limited.")
        
        if not cls.N2YO_API_KEY:
            warnings.append("N2YO API key not found. Real-time tracking features limited.")
        
        return missing_keys, warnings
    
    @classmethod
    def get_spacetrack_credentials(cls):
        """Get Space-Track credentials"""
        return {
            'username': cls.SPACETRACK_USERNAME,
            'password': cls.SPACETRACK_PASSWORD,
            'available': bool(cls.SPACETRACK_USERNAME and cls.SPACETRACK_PASSWORD)
        }
    
    @classmethod
    def get_redis_url(cls):
        """Get Redis connection URL"""
        if cls.REDIS_USERNAME and cls.REDIS_PASSWORD:
            return f"redis://{cls.REDIS_USERNAME}:{cls.REDIS_PASSWORD}@localhost:6379/0"
        return os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    @classmethod
    def is_api_available(cls, api_name: str) -> bool:
        """Check if specific API is available"""
        api_map = {
            'spacetrack': bool(cls.SPACETRACK_USERNAME and cls.SPACETRACK_PASSWORD),
            'n2yo': bool(cls.N2YO_API_KEY),
            'weather': bool(cls.WEATHER_API_KEY),
            'maps': bool(cls.GOOGLE_MAPS_API_KEY)
        }
        return api_map.get(api_name.lower(), False)

# Example .env file content (create this file in your project root)
ENV_TEMPLATE = """
# Space-Track.org credentials (register at https://www.space-track.org)
SPACETRACK_USERNAME=your_username_here
SPACETRACK_PASSWORD=your_password_here

# N2YO API key (register at https://www.n2yo.com/api/)
N2YO_API_KEY=your_n2yo_api_key_here

# Redis configuration (if using cloud Redis)
REDIS_URL=redis://localhost:6379/0
REDIS_USERNAME=
REDIS_PASSWORD=

# Flask configuration
FLASK_ENV=development
FLASK_DEBUG=true
SECRET_KEY=your-secret-key-change-this-in-production

# External APIs (optional)
WEATHER_API_KEY=
GOOGLE_MAPS_API_KEY=

# Database (if using external database)
DATABASE_URL=
"""