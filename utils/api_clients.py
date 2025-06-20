import aiohttp
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from config.api_keys import APIKeys
from config.settings import Config

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, waiting {sleep_time:.1f} seconds")
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)

class CelesTrakClient:
    """Client for CelesTrak API"""
    
    def __init__(self):
        self.base_url = "https://celestrak.org"
        self.rate_limiter = RateLimiter(calls_per_minute=30)  # Conservative rate limit
        
    async def get_tle_by_norad_id(self, norad_id: str) -> Optional[Dict]:
        """Get TLE data for a specific satellite"""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/NORAD/elements/gp.php"
        params = {
            'CATNR': norad_id,
            'FORMAT': 'TLE'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        text = await response.text()
                        return self._parse_tle_response(text, norad_id)
                    else:
                        logger.warning(f"CelesTrak returned status {response.status} for {norad_id}")
                        
        except Exception as e:
            logger.error(f"Error fetching from CelesTrak: {e}")
            
        return None
    
    async def get_satellite_catalog(self, catalog_name: str = "stations") -> List[Dict]:
        """Get satellite catalog from CelesTrak"""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/NORAD/elements/gp.php"
        params = {
            'GROUP': catalog_name,
            'FORMAT': 'TLE'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        text = await response.text()
                        return self._parse_catalog_response(text)
                        
        except Exception as e:
            logger.error(f"Error fetching catalog from CelesTrak: {e}")
            
        return []
    
    def _parse_tle_response(self, response_text: str, norad_id: str) -> Optional[Dict]:
        """Parse TLE response from CelesTrak"""
        lines = response_text.strip().split('\n')
        
        if len(lines) >= 2:
            if len(lines) >= 3 and not lines[0].startswith(('1 ', '2 ')):
                satellite_name = lines[0].strip()
                line1 = lines[1].strip()
                line2 = lines[2].strip()
            else:
                satellite_name = f"SATELLITE {norad_id}"
                line1 = lines[0].strip()
                line2 = lines[1].strip()
            
            return {
                'norad_id': norad_id,
                'name': satellite_name,
                'line1': line1,
                'line2': line2,
                'source': 'celestrak',
                'updated_at': datetime.utcnow().isoformat()
            }
        
        return None
    
    def _parse_catalog_response(self, response_text: str) -> List[Dict]:
        """Parse catalog response from CelesTrak"""
        lines = response_text.strip().split('\n')
        satellites = []
        
        for i in range(0, len(lines), 3):
            if i + 2 < len(lines):
                name = lines[i].strip()
                line1 = lines[i + 1].strip()
                line2 = lines[i + 2].strip()
                
                if line1.startswith('1 ') and line2.startswith('2 '):
                    norad_id = line1[2:7].strip()
                    satellites.append({
                        'norad_id': norad_id,
                        'name': name,
                        'line1': line1,
                        'line2': line2,
                        'source': 'celestrak',
                        'updated_at': datetime.utcnow().isoformat()
                    })
        
        return satellites

class SpaceTrackClient:
    """Client for Space-Track.org API"""
    
    def __init__(self):
        self.base_url = "https://www.space-track.org"
        self.session_token = None
        self.token_expires = None
        self.rate_limiter = RateLimiter(calls_per_minute=20)  # More conservative
        
    async def authenticate(self) -> bool:
        """Authenticate with Space-Track.org"""
        credentials = APIKeys.get_spacetrack_credentials()
        if not credentials['available']:
            logger.warning("Space-Track credentials not available")
            return False
        
        url = f"{self.base_url}/ajaxauth/login"
        data = {
            'identity': credentials['username'],
            'password': credentials['password']
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, timeout=10) as response:
                    if response.status == 200:
                        cookies = response.cookies
                        if 'spacetrack_session' in cookies:
                            self.session_token = cookies['spacetrack_session'].value
                            self.token_expires = datetime.utcnow() + timedelta(hours=1)
                            logger.info("Successfully authenticated with Space-Track")
                            return True
                            
        except Exception as e:
            logger.error(f"Space-Track authentication failed: {e}")
            
        return False
    
    async def get_tle_by_norad_id(self, norad_id: str) -> Optional[Dict]:
        """Get TLE data from Space-Track"""
        if not await self._ensure_authenticated():
            return None
            
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/basicspacedata/query/class/tle_latest/NORAD_CAT_ID/{norad_id}/orderby/TLE_LINE1 ASC/format/tle"
        headers = {'Cookie': f'spacetrack_session={self.session_token}'}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        text = await response.text()
                        return self._parse_tle_response(text, norad_id)
                        
        except Exception as e:
            logger.error(f"Error fetching from Space-Track: {e}")
            
        return None
    
    async def search_satellites(self, query: str, limit: int = 50) -> List[Dict]:
        """Search satellites on Space-Track"""
        if not await self._ensure_authenticated():
            return []
            
        await self.rate_limiter.wait_if_needed()
        
        # Search by name
        url = f"{self.base_url}/basicspacedata/query/class/satcat/OBJECT_NAME/~~{query}/orderby/OBJECT_NAME ASC/limit/{limit}/format/json"
        headers = {'Cookie': f'spacetrack_session={self.session_token}'}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=20) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_search_response(data)
                        
        except Exception as e:
            logger.error(f"Error searching on Space-Track: {e}")
            
        return []
    
    async def _ensure_authenticated(self) -> bool:
        """Ensure we have a valid authentication token"""
        if (self.session_token is None or 
            self.token_expires is None or 
            datetime.utcnow() >= self.token_expires):
            return await self.authenticate()
        return True
    
    def _parse_tle_response(self, response_text: str, norad_id: str) -> Optional[Dict]:
        """Parse TLE response from Space-Track"""
        lines = response_text.strip().split('\n')
        
        if len(lines) >= 2:
            if len(lines) >= 3 and not lines[0].startswith(('1 ', '2 ')):
                satellite_name = lines[0].strip()
                line1 = lines[1].strip()
                line2 = lines[2].strip()
            else:
                satellite_name = f"SATELLITE {norad_id}"
                line1 = lines[0].strip()
                line2 = lines[1].strip()
            
            return {
                'norad_id': norad_id,
                'name': satellite_name,
                'line1': line1,
                'line2': line2,
                'source': 'spacetrack',
                'updated_at': datetime.utcnow().isoformat()
            }
        
        return None
    
    def _parse_search_response(self, data: List[Dict]) -> List[Dict]:
        """Parse search response from Space-Track"""
        satellites = []
        
        for item in data:
            satellites.append({
                'norad_id': item.get('NORAD_CAT_ID'),
                'name': item.get('OBJECT_NAME'),
                'country': item.get('COUNTRY'),
                'launch_date': item.get('LAUNCH_DATE'),
                'object_type': item.get('OBJECT_TYPE'),
                'source': 'spacetrack'
            })
        
        return satellites

class N2YOClient:
    """Client for N2YO API (real-time satellite tracking)"""
    
    def __init__(self):
        self.base_url = "https://api.n2yo.com/rest/v1/satellite"
        self.api_key = APIKeys.N2YO_API_KEY
        self.rate_limiter = RateLimiter(calls_per_minute=10)  # Very conservative
        
    async def get_satellite_positions(self, norad_id: str, observer_lat: float, 
                                    observer_lon: float, observer_alt: float = 0,
                                    seconds: int = 300) -> Optional[Dict]:
        """Get real-time satellite positions"""
        if not self.api_key:
            logger.warning("N2YO API key not available")
            return None
            
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/positions/{norad_id}/{observer_lat}/{observer_lon}/{observer_alt}/{seconds}"
        params = {'apiKey': self.api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                        
        except Exception as e:
            logger.error(f"Error fetching from N2YO: {e}")
            
        return None
    
    async def get_satellite_passes(self, norad_id: str, observer_lat: float,
                                 observer_lon: float, observer_alt: float = 0,
                                 days: int = 10) -> Optional[Dict]:
        """Get satellite pass predictions"""
        if not self.api_key:
            return None
            
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/passes/{norad_id}/{observer_lat}/{observer_lon}/{observer_alt}/{days}"
        params = {'apiKey': self.api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=20) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                        
        except Exception as e:
            logger.error(f"Error fetching passes from N2YO: {e}")
            
        return None

class APIClientManager:
    """Manages multiple API clients with fallback logic"""
    
    def __init__(self):
        self.celestrak = CelesTrakClient()
        self.spacetrack = SpaceTrackClient()
        self.n2yo = N2YOClient()
        
    async def get_tle_data(self, norad_id: str) -> Optional[Dict]:
        """Get TLE data with fallback between sources"""
        # Try CelesTrak first (no authentication required)
        tle_data = await self.celestrak.get_tle_by_norad_id(norad_id)
        if tle_data:
            return tle_data
        
        # Fallback to Space-Track if available
        if APIKeys.is_api_available('spacetrack'):
            tle_data = await self.spacetrack.get_tle_by_norad_id(norad_id)
            if tle_data:
                return tle_data
        
        logger.warning(f"Could not fetch TLE data for {norad_id} from any source")
        return None
    
    async def search_satellites(self, query: str, limit: int = 20) -> List[Dict]:
        """Search satellites across multiple sources"""
        results = []
        
        # Try Space-Track first (most comprehensive)
        if APIKeys.is_api_available('spacetrack'):
            try:
                spacetrack_results = await self.spacetrack.search_satellites(query, limit)
                results.extend(spacetrack_results)
            except Exception as e:
                logger.error(f"Space-Track search failed: {e}")
        
        # If no results or Space-Track unavailable, try catalog search
        if not results:
            try:
                catalog_results = await self.celestrak.get_satellite_catalog("stations")
                # Filter by query
                filtered_results = [
                    sat for sat in catalog_results 
                    if query.lower() in sat['name'].lower()
                ][:limit]
                results.extend(filtered_results)
            except Exception as e:
                logger.error(f"CelesTrak catalog search failed: {e}")
        
        return results[:limit]
    
    async def get_realtime_data(self, norad_id: str, observer_lat: float, 
                              observer_lon: float) -> Optional[Dict]:
        """Get real-time satellite data if N2YO is available"""
        if APIKeys.is_api_available('n2yo'):
            return await self.n2yo.get_satellite_positions(
                norad_id, observer_lat, observer_lon
            )
        return None