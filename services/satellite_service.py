import numpy as np
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sgp4.api import Satrec, jday
import logging

from services.tle_service import TLEService
from utils.validation import validate_coordinates, validate_time_range
from utils.math_utils import calculate_ground_track, calculate_visibility

logger = logging.getLogger(__name__)

class SatelliteService:
    """Service for satellite tracking and orbital calculations"""
    
    def __init__(self):
        self.tle_service = TLEService()
        
    async def get_satellite_position(self, norad_id: str, target_time: Optional[datetime] = None) -> Dict:
        """
        Get current satellite position
        
        Args:
            norad_id: NORAD catalog number
            target_time: Time for position calculation (default: now)
            
        Returns:
            Dictionary containing position and velocity data
        """
        if target_time is None:
            target_time = datetime.utcnow()
            
        # Get TLE data
        tle_data = await self.tle_service.get_tle_by_norad_id(norad_id)
        if not tle_data:
            raise ValueError(f"TLE data not found for satellite {norad_id}")
            
        # Create satellite object
        satellite = Satrec.twoline2rv(tle_data['line1'], tle_data['line2'])
        
        # Calculate position
        jd, fr = jday(target_time.year, target_time.month, target_time.day,
                     target_time.hour, target_time.minute, target_time.second)
        
        error, position, velocity = satellite.sgp4(jd, fr)
        
        if error != 0:
            raise ValueError(f"SGP4 calculation error: {error}")
            
        # Convert to geographic coordinates
        lat, lon, alt = self._ecef_to_geodetic(position)
        
        return {
            'norad_id': norad_id,
            'name': tle_data['name'],
            'timestamp': target_time.isoformat(),
            'position': {
                'x': position[0],
                'y': position[1], 
                'z': position[2],
                'latitude': lat,
                'longitude': lon,
                'altitude': alt
            },
            'velocity': {
                'vx': velocity[0],
                'vy': velocity[1],
                'vz': velocity[2],
                'speed': math.sqrt(sum(v**2 for v in velocity))
            }
        }
    
    async def get_orbital_trajectory(self, norad_id: str, duration_hours: int = 24, 
                                   step_minutes: int = 10) -> Dict:
        """
        Calculate orbital trajectory over time
        
        Args:
            norad_id: NORAD catalog number
            duration_hours: Duration of trajectory in hours
            step_minutes: Time step in minutes
            
        Returns:
            Dictionary containing trajectory data
        """
        tle_data = await self.tle_service.get_tle_by_norad_id(norad_id)
        if not tle_data:
            raise ValueError(f"TLE data not found for satellite {norad_id}")
            
        satellite = Satrec.twoline2rv(tle_data['line1'], tle_data['line2'])
        
        # Generate time points
        start_time = datetime.utcnow()
        trajectory_points = []
        
        total_steps = int((duration_hours * 60) / step_minutes)
        
        for i in range(total_steps + 1):
            current_time = start_time + timedelta(minutes=i * step_minutes)
            
            jd, fr = jday(current_time.year, current_time.month, current_time.day,
                         current_time.hour, current_time.minute, current_time.second)
            
            error, position, velocity = satellite.sgp4(jd, fr)
            
            if error == 0:
                lat, lon, alt = self._ecef_to_geodetic(position)
                
                trajectory_points.append({
                    'timestamp': current_time.isoformat(),
                    'position': {
                        'x': position[0],
                        'y': position[1],
                        'z': position[2],
                        'latitude': lat,
                        'longitude': lon,
                        'altitude': alt
                    },
                    'velocity': {
                        'vx': velocity[0],
                        'vy': velocity[1],
                        'vz': velocity[2]
                    }
                })
            else:
                logger.warning(f"SGP4 error at step {i}: {error}")
                
        return {
            'norad_id': norad_id,
            'name': tle_data['name'],
            'start_time': start_time.isoformat(),
            'duration_hours': duration_hours,
            'step_minutes': step_minutes,
            'trajectory': trajectory_points,
            'orbital_period': self._calculate_orbital_period(satellite)
        }
    
    def _ecef_to_geodetic(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Convert ECEF coordinates to geodetic (lat, lon, alt)
        
        Args:
            position: ECEF position (x, y, z) in km
            
        Returns:
            Tuple of (latitude, longitude, altitude)
        """
        x, y, z = position
        
        # WGS84 constants
        a = 6378.137  # Semi-major axis in km
        f = 1/298.257223563  # Flattening
        b = a * (1 - f)  # Semi-minor axis
        e2 = 1 - (b**2 / a**2)  # First eccentricity squared
        
        # Calculate longitude
        lon = math.atan2(y, x)
        
        # Calculate latitude and altitude iteratively
        p = math.sqrt(x**2 + y**2)
        lat = math.atan2(z, p * (1 - e2))
        
        for _ in range(5):  # Iterate for accuracy
            N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
            alt = p / math.cos(lat) - N
            lat = math.atan2(z, p * (1 - e2 * N / (N + alt)))
            
        # Convert to degrees
        lat_deg = math.degrees(lat)
        lon_deg = math.degrees(lon)
        
        return lat_deg, lon_deg, alt
    
    def _calculate_orbital_period(self, satellite: Satrec) -> float:
        """Calculate orbital period in minutes"""
        try:
            mean_motion = satellite.no_kozai  # revolutions per day
            if mean_motion > 0:
                return 1440.0 / mean_motion  # minutes
        except:
            pass
        return 90.0  # Default LEO period
    
    async def get_ground_track(self, norad_id: str, duration_hours: int = 24) -> Dict:
        """
        Calculate satellite ground track
        
        Args:
            norad_id: NORAD catalog number
            duration_hours: Duration for ground track
            
        Returns:
            Dictionary containing ground track data
        """
        trajectory = await self.get_orbital_trajectory(norad_id, duration_hours, 5)
        
        ground_track = []
        for point in trajectory['trajectory']:
            ground_track.append({
                'timestamp': point['timestamp'],
                'latitude': point['position']['latitude'],
                'longitude': point['position']['longitude'],
                'altitude': point['position']['altitude']
            })
            
        return {
            'norad_id': norad_id,
            'name': trajectory['name'],
            'ground_track': ground_track,
            'duration_hours': duration_hours
        }
    
    async def calculate_passes(self, norad_id: str, observer_lat: float, 
                             observer_lon: float, observer_alt: float = 0,
                             duration_hours: int = 48) -> List[Dict]:
        """
        Calculate satellite passes for an observer location
        
        Args:
            norad_id: NORAD catalog number
            observer_lat: Observer latitude in degrees
            observer_lon: Observer longitude in degrees  
            observer_alt: Observer altitude in km
            duration_hours: Duration to look ahead
            
        Returns:
            List of pass predictions
        """
        is_valid, error = validate_coordinates(observer_lat, observer_lon)
        if not is_valid:
            raise ValueError(error)
            
        trajectory = await self.get_orbital_trajectory(norad_id, duration_hours, 1)
        
        passes = []
        current_pass = None
        min_elevation = 10.0  # Minimum elevation for visible pass
        
        for point in trajectory['trajectory']:
            # Calculate elevation and azimuth
            sat_lat = point['position']['latitude']
            sat_lon = point['position']['longitude']
            sat_alt = point['position']['altitude']
            
            elevation, azimuth, distance = self._calculate_look_angles(
                observer_lat, observer_lon, observer_alt,
                sat_lat, sat_lon, sat_alt
            )
            
            if elevation >= min_elevation:
                if current_pass is None:
                    # Start of new pass
                    current_pass = {
                        'start_time': point['timestamp'],
                        'max_elevation': elevation,
                        'max_elevation_time': point['timestamp'],
                        'start_azimuth': azimuth,
                        'points': []
                    }
                else:
                    # Update max elevation
                    if elevation > current_pass['max_elevation']:
                        current_pass['max_elevation'] = elevation
                        current_pass['max_elevation_time'] = point['timestamp']
                        
                current_pass['points'].append({
                    'timestamp': point['timestamp'],
                    'elevation': elevation,
                    'azimuth': azimuth,
                    'distance': distance
                })
            else:
                if current_pass is not None:
                    # End of pass
                    current_pass['end_time'] = current_pass['points'][-1]['timestamp']
                    current_pass['end_azimuth'] = current_pass['points'][-1]['azimuth']
                    current_pass['duration_minutes'] = len(current_pass['points'])
                    
                    # Only include passes with reasonable duration
                    if len(current_pass['points']) >= 3:
                        passes.append(current_pass)
                        
                    current_pass = None
                    
        return passes
    
    def _calculate_look_angles(self, obs_lat: float, obs_lon: float, obs_alt: float,
                              sat_lat: float, sat_lon: float, sat_alt: float) -> Tuple[float, float, float]:
        """
        Calculate elevation, azimuth, and distance from observer to satellite
        
        Returns:
            Tuple of (elevation, azimuth, distance)
        """
        # Convert to radians
        obs_lat_rad = math.radians(obs_lat)
        obs_lon_rad = math.radians(obs_lon)
        sat_lat_rad = math.radians(sat_lat)
        sat_lon_rad = math.radians(sat_lon)
        
        # Earth radius
        R = 6378.137  # km
        
        # Observer position in ECEF
        obs_x = (R + obs_alt) * math.cos(obs_lat_rad) * math.cos(obs_lon_rad)
        obs_y = (R + obs_alt) * math.cos(obs_lat_rad) * math.sin(obs_lon_rad)
        obs_z = (R + obs_alt) * math.sin(obs_lat_rad)
        
        # Satellite position in ECEF  
        sat_x = (R + sat_alt) * math.cos(sat_lat_rad) * math.cos(sat_lon_rad)
        sat_y = (R + sat_alt) * math.cos(sat_lat_rad) * math.sin(sat_lon_rad)
        sat_z = (R + sat_alt) * math.sin(sat_lat_rad)
        
        # Relative position vector
        dx = sat_x - obs_x
        dy = sat_y - obs_y
        dz = sat_z - obs_z
        
        # Distance
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Transform to local coordinate system
        # East-North-Up (ENU) coordinates
        sin_lat = math.sin(obs_lat_rad)
        cos_lat = math.cos(obs_lat_rad)
        sin_lon = math.sin(obs_lon_rad)
        cos_lon = math.cos(obs_lon_rad)
        
        east = -sin_lon * dx + cos_lon * dy
        north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
        
        # Calculate elevation and azimuth
        range_horizontal = math.sqrt(east**2 + north**2)
        elevation = math.degrees(math.atan2(up, range_horizontal))
        azimuth = math.degrees(math.atan2(east, north))
        
        # Normalize azimuth to 0-360
        if azimuth < 0:
            azimuth += 360
            
        return elevation, azimuth, distance
    
    async def get_satellite_info(self, norad_id: str) -> Dict:
        """
        Get comprehensive satellite information
        
        Args:
            norad_id: NORAD catalog number
            
        Returns:
            Dictionary containing satellite information
        """
        tle_data = await self.tle_service.get_tle_by_norad_id(norad_id)
        if not tle_data:
            raise ValueError(f"Satellite {norad_id} not found")
            
        satellite = Satrec.twoline2rv(tle_data['line1'], tle_data['line2'])
        
        # Extract orbital elements
        orbital_info = {
            'inclination': math.degrees(satellite.inclo),
            'eccentricity': satellite.ecco,
            'right_ascension': math.degrees(satellite.nodeo),
            'argument_of_perigee': math.degrees(satellite.argpo),
            'mean_anomaly': math.degrees(satellite.mo),
            'mean_motion': satellite.no_kozai * 1440 / (2 * math.pi),  # rev/day
            'orbital_period': self._calculate_orbital_period(satellite),
            'semi_major_axis': self._calculate_semi_major_axis(satellite.no_kozai)
        }
        
        # Calculate current position
        current_pos = await self.get_satellite_position(norad_id)
        
        return {
            'norad_id': norad_id,
            'name': tle_data['name'],
            'tle_epoch': tle_data['epoch'],
            'last_updated': tle_data['updated_at'],
            'orbital_elements': orbital_info,
            'current_position': current_pos['position'],
            'current_velocity': current_pos['velocity']
        }
    
    def _calculate_semi_major_axis(self, mean_motion: float) -> float:
        """Calculate semi-major axis from mean motion"""
        # GM for Earth in km³/s²
        mu = 398600.4418
        
        # Convert mean motion from rad/min to rad/s
        n = mean_motion / 60.0
        
        # Calculate semi-major axis using Kepler's third law
        a = (mu / (n**2))**(1/3)
        
        return a
    
    async def compare_satellites(self, norad_ids: List[str]) -> Dict:
        """
        Compare multiple satellites
        
        Args:
            norad_ids: List of NORAD catalog numbers
            
        Returns:
            Dictionary containing comparison data
        """
        satellites_info = []
        
        for norad_id in norad_ids:
            try:
                info = await self.get_satellite_info(norad_id)
                satellites_info.append(info)
            except Exception as e:
                logger.error(f"Error getting info for satellite {norad_id}: {e}")
                
        return {
            'satellites': satellites_info,
            'comparison_time': datetime.utcnow().isoformat(),
            'count': len(satellites_info)
        }