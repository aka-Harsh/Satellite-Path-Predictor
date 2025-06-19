import requests
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from config.settings import Config
from utils.validation import validate_tle, validate_norad_id
from utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class TLEService:
    """Service for fetching and managing TLE (Two-Line Element) data"""
    
    def __init__(self):
        self.cache = CacheManager()
        self.celestrak_url = Config.CELESTRAK_BASE_URL
        self.spacetrack_url = Config.SPACETRACK_BASE_URL
        self.session_token = None
        
    async def get_tle_by_norad_id(self, norad_id: str) -> Optional[Dict]:
        """
        Get TLE data for a specific satellite by NORAD ID
        
        Args:
            norad_id: NORAD catalog number
            
        Returns:
            Dictionary containing TLE data or None if not found
        """
        is_valid, error = validate_norad_id(norad_id)
        if not is_valid:
            raise ValueError(error)
            
        # Check cache first
        cache_key = f"tle_{norad_id}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            logger.info(f"Retrieved TLE for {norad_id} from cache")
            return cached_data
            
        # Try CelesTrak first (no authentication required)
        tle_data = await self._fetch_from_celestrak(norad_id)
        
        # If CelesTrak fails, try Space-Track (requires authentication)
        if not tle_data and Config.SPACETRACK_USERNAME:
            tle_data = await self._fetch_from_spacetrack(norad_id)
            
        if tle_data:
            # Cache the result
            self.cache.set(cache_key, tle_data, Config.TLE_CACHE_TIMEOUT)
            logger.info(f"Retrieved and cached TLE for {norad_id}")
            
        return tle_data
    
    async def _fetch_from_celestrak(self, norad_id: str) -> Optional[Dict]:
        """Fetch TLE data from CelesTrak"""
        try:
            url = f"{self.celestrak_url}/NORAD/elements/gp.php"
            params = {
                'CATNR': norad_id,
                'FORMAT': 'TLE'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        text = await response.text()
                        return self._parse_tle_response(text, norad_id)
                        
        except Exception as e:
            logger.error(f"Error fetching from CelesTrak: {e}")
            
        return None
    
    async def _fetch_from_spacetrack(self, norad_id: str) -> Optional[Dict]:
        """Fetch TLE data from Space-Track.org"""
        try:
            # Authenticate if needed
            if not self.session_token:
                await self._authenticate_spacetrack()
                
            if not self.session_token:
                return None
                
            url = f"{self.spacetrack_url}/basicspacedata/query/class/tle_latest/NORAD_CAT_ID/{norad_id}/orderby/TLE_LINE1 ASC/format/tle"
            
            headers = {'Cookie': f'spacetrack_session={self.session_token}'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        text = await response.text()
                        return self._parse_tle_response(text, norad_id)
                        
        except Exception as e:
            logger.error(f"Error fetching from Space-Track: {e}")
            
        return None
    
    async def _authenticate_spacetrack(self):
        """Authenticate with Space-Track.org"""
        if not Config.SPACETRACK_USERNAME or not Config.SPACETRACK_PASSWORD:
            return
            
        try:
            url = f"{self.spacetrack_url}/ajaxauth/login"
            data = {
                'identity': Config.SPACETRACK_USERNAME,
                'password': Config.SPACETRACK_PASSWORD
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        # Extract session token from cookies
                        cookies = response.cookies
                        if 'spacetrack_session' in cookies:
                            self.session_token = cookies['spacetrack_session'].value
                            logger.info("Successfully authenticated with Space-Track")
                        
        except Exception as e:
            logger.error(f"Space-Track authentication failed: {e}")
    
    def _parse_tle_response(self, response_text: str, norad_id: str) -> Optional[Dict]:
        """Parse TLE response text"""
        lines = response_text.strip().split('\n')
        
        if len(lines) >= 2:
            # Handle response with satellite name (3 lines) or without (2 lines)
            if len(lines) >= 3 and not lines[0].startswith(('1 ', '2 ')):
                satellite_name = lines[0].strip()
                line1 = lines[1].strip()
                line2 = lines[2].strip()
            else:
                satellite_name = f"SATELLITE {norad_id}"
                line1 = lines[0].strip()
                line2 = lines[1].strip()
            
            # Validate TLE
            is_valid, error = validate_tle(line1, line2)
            if is_valid:
                return {
                    'norad_id': norad_id,
                    'name': satellite_name,
                    'line1': line1,
                    'line2': line2,
                    'epoch': self._extract_epoch(line1),
                    'updated_at': datetime.utcnow().isoformat()
                }
            else:
                logger.warning(f"Invalid TLE for {norad_id}: {error}")
                
        return None
    
    def _extract_epoch(self, line1: str) -> str:
        """Extract epoch from TLE line 1"""
        try:
            epoch_year = int(line1[18:20])
            epoch_day = float(line1[20:32])
            
            # Convert 2-digit year to 4-digit
            if epoch_year < 57:
                full_year = 2000 + epoch_year
            else:
                full_year = 1900 + epoch_year
                
            # Convert day of year to date
            base_date = datetime(full_year, 1, 1)
            epoch_date = base_date + timedelta(days=epoch_day - 1)
            
            return epoch_date.isoformat()
            
        except (ValueError, IndexError):
            return datetime.utcnow().isoformat()
    
    async def get_popular_satellites_tle(self) -> List[Dict]:
        """Get TLE data for popular satellites"""
        popular_satellites = [
            "25544",  # ISS
            "20580",  # Hubble Space Telescope
            "44713",  # Starlink-1007
            "28654",  # NOAA 18
            "26360",  # GPS BIIR-2
            "43013",  # Starlink-1130
            "37849",  # GOES 16
            "40069",  # NROL-71
            "36411",  # GSAT-15
            "41463"   # Dragon CRS-20
        ]
        
        tasks = [self.get_tle_by_norad_id(norad_id) for norad_id in popular_satellites]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_tles = []
        for result in results:
            if isinstance(result, dict):
                valid_tles.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error fetching TLE: {result}")
                
        return valid_tles
    
    async def search_satellites(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search for satellites by name
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of satellite information
        """
        # For now, search within popular satellites
        # In production, this could query a satellite database
        popular_tles = await self.get_popular_satellites_tle()
        
        query_lower = query.lower()
        filtered_results = []
        
        for tle in popular_tles:
            if (query_lower in tle['name'].lower() or 
                query_lower in tle['norad_id']):
                filtered_results.append(tle)
                
        return filtered_results[:limit]
    
    def validate_tle_data(self, line1: str, line2: str) -> Tuple[bool, str]:
        """Validate TLE data"""
        return validate_tle(line1, line2)
    
    async def get_satellite_categories(self) -> Dict[str, List[str]]:
        """Get satellites organized by categories"""
        categories = {
            "Space Stations": ["25544"],  # ISS
            "Space Observatories": ["20580"],  # Hubble
            "Communication": ["44713", "43013"],  # Starlink
            "Weather": ["28654", "37849"],  # NOAA, GOES
            "Navigation": ["26360"],  # GPS
            "Earth Observation": ["40069"],
            "Experimental": ["41463"]
        }
        
        result = {}
        for category, norad_ids in categories.items():
            tasks = [self.get_tle_by_norad_id(norad_id) for norad_id in norad_ids]
            tle_data = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_satellites = []
            for tle in tle_data:
                if isinstance(tle, dict):
                    valid_satellites.append(tle)
                    
            if valid_satellites:
                result[category] = valid_satellites
                
        return result
    
    async def update_tle_cache(self):
        """Update TLE cache for popular satellites"""
        logger.info("Updating TLE cache for popular satellites")
        
        try:
            popular_tles = await self.get_popular_satellites_tle()
            
            # Save to file cache as backup
            cache_file = Path(Config.TLE_CACHE_DIR) / "popular_satellites.json"
            cache_file.parent.mkdir(exist_ok=True)
            
            with open(cache_file, 'w') as f:
                json.dump({
                    'updated_at': datetime.utcnow().isoformat(),
                    'satellites': popular_tles
                }, f, indent=2)
                
            logger.info(f"Updated TLE cache with {len(popular_tles)} satellites")
            
        except Exception as e:
            logger.error(f"Failed to update TLE cache: {e}")
    
    def get_cached_popular_satellites(self) -> List[Dict]:
        """Get popular satellites from file cache (fallback)"""
        try:
            cache_file = Path(Config.TLE_CACHE_DIR) / "popular_satellites.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    
                # Check if cache is recent (within 6 hours)
                updated_at = datetime.fromisoformat(data['updated_at'])
                if datetime.utcnow() - updated_at < timedelta(hours=6):
                    return data['satellites']
                    
        except Exception as e:
            logger.error(f"Error reading TLE cache: {e}")
            
        return []