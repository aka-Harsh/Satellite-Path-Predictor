import numpy as np
import math
from typing import Tuple, List, Dict
from datetime import datetime, timedelta

def calculate_ground_track(positions: List[Tuple[float, float, float]]) -> List[Dict]:
    """
    Calculate ground track from satellite positions
    
    Args:
        positions: List of (x, y, z) positions in km
        
    Returns:
        List of ground track points with lat/lon
    """
    ground_track = []
    
    for position in positions:
        lat, lon, alt = ecef_to_geodetic(position)
        ground_track.append({
            'latitude': lat,
            'longitude': lon,
            'altitude': alt
        })
    
    return ground_track

def ecef_to_geodetic(position: Tuple[float, float, float]) -> Tuple[float, float, float]:
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

def geodetic_to_ecef(lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
    """
    Convert geodetic coordinates to ECEF
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in km
        
    Returns:
        Tuple of (x, y, z) in km
    """
    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # WGS84 constants
    a = 6378.137  # Semi-major axis in km
    f = 1/298.257223563  # Flattening
    e2 = 1 - (1 - f)**2  # First eccentricity squared
    
    # Calculate radius of curvature
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    
    # Calculate ECEF coordinates
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (N * (1 - e2) + alt) * math.sin(lat_rad)
    
    return x, y, z

def calculate_visibility(sat_position: Tuple[float, float, float],
                        observer_position: Tuple[float, float, float],
                        min_elevation: float = 10.0) -> Dict:
    """
    Calculate satellite visibility from observer location
    
    Args:
        sat_position: Satellite position (x, y, z) in km
        observer_position: Observer position (lat, lon, alt)
        min_elevation: Minimum elevation angle for visibility
        
    Returns:
        Dictionary containing visibility information
    """
    obs_lat, obs_lon, obs_alt = observer_position
    
    # Convert observer to ECEF
    obs_x, obs_y, obs_z = geodetic_to_ecef(obs_lat, obs_lon, obs_alt)
    
    # Calculate look angles
    elevation, azimuth, distance = calculate_look_angles(
        (obs_x, obs_y, obs_z), sat_position
    )
    
    is_visible = elevation >= min_elevation
    
    return {
        'visible': is_visible,
        'elevation': elevation,
        'azimuth': azimuth,
        'distance': distance,
        'observer_position': observer_position,
        'satellite_position': sat_position
    }

def calculate_look_angles(observer_ecef: Tuple[float, float, float],
                         satellite_ecef: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Calculate elevation, azimuth, and range from observer to satellite
    
    Args:
        observer_ecef: Observer position in ECEF (x, y, z) km
        satellite_ecef: Satellite position in ECEF (x, y, z) km
        
    Returns:
        Tuple of (elevation, azimuth, range) in degrees, degrees, km
    """
    obs_x, obs_y, obs_z = observer_ecef
    sat_x, sat_y, sat_z = satellite_ecef
    
    # Calculate observer geodetic coordinates
    obs_lat, obs_lon, obs_alt = ecef_to_geodetic(observer_ecef)
    obs_lat_rad = math.radians(obs_lat)
    obs_lon_rad = math.radians(obs_lon)
    
    # Relative position vector
    dx = sat_x - obs_x
    dy = sat_y - obs_y
    dz = sat_z - obs_z
    
    # Range
    range_km = math.sqrt(dx**2 + dy**2 + dz**2)
    
    # Transform to local coordinate system (East-North-Up)
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