from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import math

@dataclass
class TLEData:
    """Two-Line Element data structure"""
    norad_id: str
    name: str
    line1: str
    line2: str
    epoch: str
    source: str = "unknown"
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'norad_id': self.norad_id,
            'name': self.name,
            'line1': self.line1,
            'line2': self.line2,
            'epoch': self.epoch,
            'source': self.source,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TLEData':
        """Create from dictionary"""
        return cls(
            norad_id=data['norad_id'],
            name=data['name'],
            line1=data['line1'],
            line2=data['line2'],
            epoch=data['epoch'],
            source=data.get('source', 'unknown'),
            updated_at=data.get('updated_at', datetime.utcnow().isoformat())
        )
    
    def is_recent(self, max_age_hours: int = 24) -> bool:
        """Check if TLE data is recent"""
        try:
            updated = datetime.fromisoformat(self.updated_at.replace('Z', ''))
            age = datetime.utcnow() - updated
            return age.total_seconds() / 3600 < max_age_hours
        except:
            return False
    
    def extract_orbital_elements(self) -> Dict:
        """Extract basic orbital elements from TLE"""
        try:
            # Parse line 2 for orbital elements
            inclination = float(self.line2[8:16])
            raan = float(self.line2[17:25])
            eccentricity = float('0.' + self.line2[26:33])
            arg_perigee = float(self.line2[34:42])
            mean_anomaly = float(self.line2[43:51])
            mean_motion = float(self.line2[52:63])
            
            return {
                'inclination': inclination,
                'raan': raan,
                'eccentricity': eccentricity,
                'argument_of_perigee': arg_perigee,
                'mean_anomaly': mean_anomaly,
                'mean_motion': mean_motion,
                'orbital_period': 1440 / mean_motion if mean_motion > 0 else 0  # minutes
            }
        except Exception as e:
            return {}

@dataclass
class Position:
    """3D position in space"""
    x: float  # km
    y: float  # km
    z: float  # km
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @property
    def magnitude(self) -> float:
        """Distance from Earth center"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    @property
    def altitude(self) -> float:
        """Altitude above Earth surface (assuming spherical Earth)"""
        earth_radius = 6371.0  # km
        return self.magnitude - earth_radius
    
    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'magnitude': self.magnitude,
            'altitude': self.altitude,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        return cls(
            x=data['x'],
            y=data['y'],
            z=data['z'],
            timestamp=data.get('timestamp', datetime.utcnow().isoformat())
        )

@dataclass
class Velocity:
    """3D velocity in space"""
    vx: float  # km/s
    vy: float  # km/s
    vz: float  # km/s
    
    @property
    def magnitude(self) -> float:
        """Speed magnitude"""
        return math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
    
    def to_dict(self) -> Dict:
        return {
            'vx': self.vx,
            'vy': self.vy,
            'vz': self.vz,
            'magnitude': self.magnitude
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Velocity':
        return cls(
            vx=data['vx'],
            vy=data['vy'],
            vz=data['vz']
        )

@dataclass
class GeodeticPosition:
    """Position in geodetic coordinates"""
    latitude: float   # degrees
    longitude: float  # degrees
    altitude: float   # km above Earth surface
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GeodeticPosition':
        return cls(
            latitude=data['latitude'],
            longitude=data['longitude'],
            altitude=data['altitude'],
            timestamp=data.get('timestamp', datetime.utcnow().isoformat())
        )
    
    def is_over_location(self, target_lat: float, target_lon: float, 
                        tolerance_deg: float = 1.0) -> bool:
        """Check if position is over a specific location"""
        lat_diff = abs(self.latitude - target_lat)
        lon_diff = abs(self.longitude - target_lon)
        return lat_diff <= tolerance_deg and lon_diff <= tolerance_deg

@dataclass
class OrbitalElements:
    """Classical orbital elements"""
    semi_major_axis: float      # km
    eccentricity: float         # 0-1
    inclination: float          # degrees
    raan: float                 # right ascension of ascending node, degrees
    argument_of_periapsis: float # degrees
    true_anomaly: float         # degrees
    mean_motion: float          # radians per second
    period_minutes: float       # orbital period in minutes
    
    def to_dict(self) -> Dict:
        return {
            'semi_major_axis': self.semi_major_axis,
            'eccentricity': self.eccentricity,
            'inclination': self.inclination,
            'raan': self.raan,
            'argument_of_periapsis': self.argument_of_periapsis,
            'true_anomaly': self.true_anomaly,
            'mean_motion': self.mean_motion,
            'period_minutes': self.period_minutes,
            'period_hours': self.period_minutes / 60,
            'apogee': self.semi_major_axis * (1 + self.eccentricity) - 6371,  # km above Earth
            'perigee': self.semi_major_axis * (1 - self.eccentricity) - 6371   # km above Earth
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OrbitalElements':
        return cls(
            semi_major_axis=data['semi_major_axis'],
            eccentricity=data['eccentricity'],
            inclination=data['inclination'],
            raan=data['raan'],
            argument_of_periapsis=data['argument_of_periapsis'],
            true_anomaly=data['true_anomaly'],
            mean_motion=data['mean_motion'],
            period_minutes=data['period_minutes']
        )
    
    @property
    def is_circular(self) -> bool:
        """Check if orbit is approximately circular"""
        return self.eccentricity < 0.01
    
    @property
    def is_polar(self) -> bool:
        """Check if orbit is polar"""
        return 80 <= self.inclination <= 100
    
    @property
    def is_geostationary(self) -> bool:
        """Check if orbit is geostationary"""
        return (1430 <= self.period_minutes <= 1450 and  # ~24 hours
                abs(self.inclination) < 1 and              # Near equatorial
                self.eccentricity < 0.01)                  # Near circular

@dataclass
class SatelliteInfo:
    """Complete satellite information"""
    norad_id: str
    name: str
    category: str = "Unknown"
    country: str = "Unknown"
    launch_date: Optional[str] = None
    status: str = "Unknown"
    mass_kg: Optional[float] = None
    
    # Current state
    tle_data: Optional[TLEData] = None
    position: Optional[Position] = None
    velocity: Optional[Velocity] = None
    geodetic_position: Optional[GeodeticPosition] = None
    orbital_elements: Optional[OrbitalElements] = None
    
    # Metadata
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = {
            'norad_id': self.norad_id,
            'name': self.name,
            'category': self.category,
            'country': self.country,
            'launch_date': self.launch_date,
            'status': self.status,
            'mass_kg': self.mass_kg,
            'last_updated': self.last_updated
        }
        
        # Add nested objects if they exist
        if self.tle_data:
            result['tle_data'] = self.tle_data.to_dict()
        if self.position:
            result['position'] = self.position.to_dict()
        if self.velocity:
            result['velocity'] = self.velocity.to_dict()
        if self.geodetic_position:
            result['geodetic_position'] = self.geodetic_position.to_dict()
        if self.orbital_elements:
            result['orbital_elements'] = self.orbital_elements.to_dict()
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SatelliteInfo':
        """Create from dictionary"""
        # Extract nested objects
        tle_data = None
        if 'tle_data' in data and data['tle_data']:
            tle_data = TLEData.from_dict(data['tle_data'])
        
        position = None
        if 'position' in data and data['position']:
            position = Position.from_dict(data['position'])
        
        velocity = None
        if 'velocity' in data and data['velocity']:
            velocity = Velocity.from_dict(data['velocity'])
        
        geodetic_position = None
        if 'geodetic_position' in data and data['geodetic_position']:
            geodetic_position = GeodeticPosition.from_dict(data['geodetic_position'])
        
        orbital_elements = None
        if 'orbital_elements' in data and data['orbital_elements']:
            orbital_elements = OrbitalElements.from_dict(data['orbital_elements'])
        
        return cls(
            norad_id=data['norad_id'],
            name=data['name'],
            category=data.get('category', 'Unknown'),
            country=data.get('country', 'Unknown'),
            launch_date=data.get('launch_date'),
            status=data.get('status', 'Unknown'),
            mass_kg=data.get('mass_kg'),
            tle_data=tle_data,
            position=position,
            velocity=velocity,
            geodetic_position=geodetic_position,
            orbital_elements=orbital_elements,
            last_updated=data.get('last_updated', datetime.utcnow().isoformat())
        )
    
    def update_position(self, position: Position, velocity: Velocity, 
                       geodetic_position: Optional[GeodeticPosition] = None):
        """Update satellite position and velocity"""
        self.position = position
        self.velocity = velocity
        self.geodetic_position = geodetic_position
        self.last_updated = datetime.utcnow().isoformat()
    
    def is_data_fresh(self, max_age_minutes: int = 30) -> bool:
        """Check if satellite data is fresh"""
        try:
            updated = datetime.fromisoformat(self.last_updated.replace('Z', ''))
            age = datetime.utcnow() - updated
            return age.total_seconds() / 60 < max_age_minutes
        except:
            return False

@dataclass
class TrajectoryPoint:
    """Single point in a satellite trajectory"""
    timestamp: str
    position: Position
    velocity: Velocity
    geodetic_position: Optional[GeodeticPosition] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = {
            'timestamp': self.timestamp,
            'position': self.position.to_dict(),
            'velocity': self.velocity.to_dict()
        }
        if self.geodetic_position:
            result['geodetic_position'] = self.geodetic_position.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrajectoryPoint':
        """Create from dictionary"""
        position = Position.from_dict(data['position'])
        velocity = Velocity.from_dict(data['velocity'])
        
        geodetic_position = None
        if 'geodetic_position' in data and data['geodetic_position']:
            geodetic_position = GeodeticPosition.from_dict(data['geodetic_position'])
        
        return cls(
            timestamp=data['timestamp'],
            position=position,
            velocity=velocity,
            geodetic_position=geodetic_position
        )

@dataclass
class Trajectory:
    """Complete satellite trajectory"""
    norad_id: str
    name: str
    start_time: str
    end_time: str
    points: List[TrajectoryPoint] = field(default_factory=list)
    orbital_period: Optional[float] = None  # minutes
    
    def to_dict(self) -> Dict:
        return {
            'norad_id': self.norad_id,
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'points': [point.to_dict() for point in self.points],
            'orbital_period': self.orbital_period,
            'point_count': len(self.points)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Trajectory':
        """Create from dictionary"""
        points = [TrajectoryPoint.from_dict(point_data) for point_data in data.get('points', [])]
        
        return cls(
            norad_id=data['norad_id'],
            name=data['name'],
            start_time=data['start_time'],
            end_time=data['end_time'],
            points=points,
            orbital_period=data.get('orbital_period')
        )
    
    def get_positions_array(self) -> List[List[float]]:
        """Get positions as array for visualization"""
        return [[p.position.x, p.position.y, p.position.z] for p in self.points]
    
    def get_ground_track(self) -> List[Dict]:
        """Get ground track points"""
        ground_track = []
        for point in self.points:
            if point.geodetic_position:
                ground_track.append({
                    'timestamp': point.timestamp,
                    'latitude': point.geodetic_position.latitude,
                    'longitude': point.geodetic_position.longitude,
                    'altitude': point.geodetic_position.altitude
                })
        return ground_track
    
    def get_altitude_profile(self) -> List[Dict]:
        """Get altitude profile over time"""
        profile = []
        for point in self.points:
            profile.append({
                'timestamp': point.timestamp,
                'altitude': point.position.altitude
            })
        return profile

@dataclass
class PassPrediction:
    """Satellite pass prediction for an observer"""
    start_time: str
    end_time: str
    max_elevation: float  # degrees
    max_elevation_time: str
    start_azimuth: float  # degrees
    end_azimuth: float   # degrees
    duration_minutes: float
    points: List[Dict] = field(default_factory=list)  # elevation, azimuth, distance for each point
    
    def to_dict(self) -> Dict:
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'max_elevation': self.max_elevation,
            'max_elevation_time': self.max_elevation_time,
            'start_azimuth': self.start_azimuth,
            'end_azimuth': self.end_azimuth,
            'duration_minutes': self.duration_minutes,
            'points': self.points,
            'is_visible': self.max_elevation >= 10.0,  # Above horizon
            'is_good_pass': self.max_elevation >= 30.0,  # High elevation
            'pass_type': self.get_pass_type()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PassPrediction':
        """Create from dictionary"""
        return cls(
            start_time=data['start_time'],
            end_time=data['end_time'],
            max_elevation=data['max_elevation'],
            max_elevation_time=data['max_elevation_time'],
            start_azimuth=data['start_azimuth'],
            end_azimuth=data['end_azimuth'],
            duration_minutes=data['duration_minutes'],
            points=data.get('points', [])
        )
    
    def get_pass_type(self) -> str:
        """Classify pass quality"""
        if self.max_elevation >= 60:
            return "Excellent"
        elif self.max_elevation >= 40:
            return "Very Good"
        elif self.max_elevation >= 20:
            return "Good"
        elif self.max_elevation >= 10:
            return "Fair"
        else:
            return "Poor"
    
    @property
    def is_good_pass(self) -> bool:
        """Check if this is a good pass (high elevation)"""
        return self.max_elevation >= 30.0

@dataclass
class SatelliteCategory:
    """Satellite category information"""
    name: str
    description: str
    typical_altitude: str
    examples: List[str]
    count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'typical_altitude': self.typical_altitude,
            'examples': self.examples,
            'count': self.count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SatelliteCategory':
        return cls(
            name=data['name'],
            description=data['description'],
            typical_altitude=data['typical_altitude'],
            examples=data['examples'],
            count=data.get('count', 0)
        )

class SatelliteDatabase:
    """In-memory satellite database"""
    
    def __init__(self):
        self.satellites: Dict[str, SatelliteInfo] = {}
        self.categories: Dict[str, SatelliteCategory] = {}
        self.last_updated = datetime.utcnow()
    
    def add_satellite(self, satellite: SatelliteInfo):
        """Add satellite to database"""
        self.satellites[satellite.norad_id] = satellite
        self.last_updated = datetime.utcnow()
    
    def get_satellite(self, norad_id: str) -> Optional[SatelliteInfo]:
        """Get satellite by NORAD ID"""
        return self.satellites.get(norad_id)
    
    def search_satellites(self, query: str, limit: int = 20) -> List[SatelliteInfo]:
        """Search satellites by name"""
        query_lower = query.lower()
        results = []
        
        for satellite in self.satellites.values():
            if (query_lower in satellite.name.lower() or 
                query_lower in satellite.norad_id or
                query_lower in satellite.category.lower()):
                results.append(satellite)
                if len(results) >= limit:
                    break
        
        # Sort by relevance (exact matches first)
        results.sort(key=lambda s: (
            query_lower not in s.name.lower(),
            query_lower not in s.norad_id,
            s.name.lower()
        ))
        
        return results[:limit]
    
    def get_by_category(self, category: str) -> List[SatelliteInfo]:
        """Get satellites by category"""
        return [sat for sat in self.satellites.values() if sat.category == category]
    
    def get_popular_satellites(self, limit: int = 20) -> List[SatelliteInfo]:
        """Get popular/well-known satellites"""
        # Prioritize by category importance
        priority_order = [
            "Space Station", "Space Observatory", "Communication", 
            "Weather", "Navigation", "Earth Observation"
        ]
        
        popular = []
        for category in priority_order:
            category_sats = self.get_by_category(category)
            # Sort by name within category for consistency
            category_sats.sort(key=lambda s: s.name)
            popular.extend(category_sats[:3])  # Top 3 from each category
            if len(popular) >= limit:
                break
        
        return popular[:limit]
    
    def get_active_satellites(self) -> List[SatelliteInfo]:
        """Get all active satellites"""
        return [sat for sat in self.satellites.values() if sat.status.lower() in ['active', 'operational']]
    
    def get_satellites_by_country(self, country: str) -> List[SatelliteInfo]:
        """Get satellites by country"""
        return [sat for sat in self.satellites.values() if sat.country.lower() == country.lower()]
    
    def update_statistics(self):
        """Update category statistics"""
        category_counts = {}
        for satellite in self.satellites.values():
            category = satellite.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        for category_name, category in self.categories.items():
            category.count = category_counts.get(category_name, 0)
    
    def to_dict(self) -> Dict:
        """Convert database to dictionary"""
        return {
            'satellites': {norad_id: sat.to_dict() for norad_id, sat in self.satellites.items()},
            'categories': {name: cat.to_dict() for name, cat in self.categories.items()},
            'last_updated': self.last_updated.isoformat(),
            'total_satellites': len(self.satellites),
            'active_satellites': len(self.get_active_satellites())
        }
    
    def load_from_json(self, json_data: Dict):
        """Load database from JSON data"""
        # Load categories
        if 'categories' in json_data:
            for name, cat_data in json_data['categories'].items():
                if isinstance(cat_data, dict) and 'description' in cat_data:
                    self.categories[name] = SatelliteCategory(
                        name=name,
                        description=cat_data['description'],
                        typical_altitude=cat_data.get('typical_altitude', 'Unknown'),
                        examples=cat_data.get('examples', [])
                    )
        
        # Load popular satellites from the structure
        if 'popular_satellites' in json_data:
            for sat_data in json_data['popular_satellites']:
                try:
                    # Convert the JSON structure to SatelliteInfo
                    satellite = SatelliteInfo(
                        norad_id=sat_data['norad_id'],
                        name=sat_data['name'],
                        category=sat_data['category'],
                        country=sat_data.get('country', 'Unknown'),
                        launch_date=sat_data.get('launch_date'),
                        status=sat_data.get('status', 'Unknown'),
                        mass_kg=sat_data.get('mass_kg')
                    )
                    self.satellites[satellite.norad_id] = satellite
                except Exception as e:
                    print(f"Error loading satellite {sat_data.get('norad_id', 'unknown')}: {e}")
        
        # Load satellites if they exist
        if 'satellites' in json_data:
            for norad_id, sat_data in json_data['satellites'].items():
                try:
                    satellite = SatelliteInfo.from_dict(sat_data)
                    self.satellites[norad_id] = satellite
                except Exception as e:
                    print(f"Error loading satellite {norad_id}: {e}")
        
        self.update_statistics()
    
    def export_to_json(self) -> Dict:
        """Export database to JSON format"""
        return self.to_dict()
    
    def clear(self):
        """Clear all data"""
        self.satellites.clear()
        self.categories.clear()
        self.last_updated = datetime.utcnow()

# Global satellite database instance
satellite_db = SatelliteDatabase()