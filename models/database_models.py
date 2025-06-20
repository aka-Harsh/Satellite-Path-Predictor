import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from config.settings import Config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """SQLite database manager for satellite data"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.SQLITE_DB_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Satellites table
                CREATE TABLE IF NOT EXISTS satellites (
                    norad_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT,
                    country TEXT,
                    launch_date TEXT,
                    status TEXT,
                    mass_kg REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- TLE data table
                CREATE TABLE IF NOT EXISTS tle_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    norad_id TEXT NOT NULL,
                    line1 TEXT NOT NULL,
                    line2 TEXT NOT NULL,
                    epoch TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (norad_id) REFERENCES satellites (norad_id)
                );
                
                -- Predictions table
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    norad_id TEXT NOT NULL,
                    prediction_time TIMESTAMP NOT NULL,
                    target_time TIMESTAMP NOT NULL,
                    hours_ahead REAL NOT NULL,
                    position_x REAL,
                    position_y REAL,
                    position_z REAL,
                    velocity_x REAL,
                    velocity_y REAL,
                    velocity_z REAL,
                    method TEXT,
                    metrics_json TEXT,
                    confidence_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (norad_id) REFERENCES satellites (norad_id)
                );
                
                -- Trajectories table
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    norad_id TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP NOT NULL,
                    points_json TEXT NOT NULL,
                    orbital_period REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (norad_id) REFERENCES satellites (norad_id)
                );
                
                -- Pass predictions table
                CREATE TABLE IF NOT EXISTS pass_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    norad_id TEXT NOT NULL,
                    observer_lat REAL NOT NULL,
                    observer_lon REAL NOT NULL,
                    observer_alt REAL DEFAULT 0,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP NOT NULL,
                    max_elevation REAL NOT NULL,
                    max_elevation_time TIMESTAMP NOT NULL,
                    duration_minutes REAL NOT NULL,
                    points_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (norad_id) REFERENCES satellites (norad_id)
                );
                
                -- Model performance table
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    test_period TEXT,
                    sample_count INTEGER,
                    mean_position_error REAL,
                    mean_velocity_error REAL,
                    performance_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Cache table for general caching
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create indexes for better performance
                CREATE INDEX IF NOT EXISTS idx_tle_norad_id ON tle_data (norad_id);
                CREATE INDEX IF NOT EXISTS idx_tle_created_at ON tle_data (created_at);
                CREATE INDEX IF NOT EXISTS idx_pred_norad_id ON predictions (norad_id);
                CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions (prediction_time);
                CREATE INDEX IF NOT EXISTS idx_traj_norad_id ON trajectories (norad_id);
                CREATE INDEX IF NOT EXISTS idx_traj_time ON trajectories (start_time);
                CREATE INDEX IF NOT EXISTS idx_pass_norad_id ON pass_predictions (norad_id);
                CREATE INDEX IF NOT EXISTS idx_pass_observer ON pass_predictions (observer_lat, observer_lon);
                CREATE INDEX IF NOT EXISTS idx_pass_time ON pass_predictions (start_time);
                CREATE INDEX IF NOT EXISTS idx_perf_model ON model_performance (model_name);
                CREATE INDEX IF NOT EXISTS idx_perf_time ON model_performance (created_at);
                CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache (expires_at);
            """)
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def add_satellite(self, satellite_data: Dict) -> bool:
        """Add or update satellite information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO satellites 
                    (norad_id, name, category, country, launch_date, status, mass_kg, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    satellite_data['norad_id'],
                    satellite_data['name'],
                    satellite_data.get('category'),
                    satellite_data.get('country'),
                    satellite_data.get('launch_date'),
                    satellite_data.get('status'),
                    satellite_data.get('mass_kg'),
                    datetime.utcnow().isoformat()
                ))
            return True
        except Exception as e:
            logger.error(f"Error adding satellite: {e}")
            return False
    
    def get_satellite(self, norad_id: str) -> Optional[Dict]:
        """Get satellite by NORAD ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM satellites WHERE norad_id = ?", (norad_id,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting satellite: {e}")
            return None
    
    def search_satellites(self, query: str, limit: int = 20) -> List[Dict]:
        """Search satellites by name or NORAD ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM satellites 
                    WHERE name LIKE ? OR norad_id LIKE ? OR category LIKE ?
                    ORDER BY 
                        CASE 
                            WHEN name LIKE ? THEN 1
                            WHEN norad_id LIKE ? THEN 2
                            ELSE 3
                        END,
                        name 
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", f"%{query}%", f"{query}%", f"{query}%", limit))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error searching satellites: {e}")
            return []
    
    def get_satellites_by_category(self, category: str) -> List[Dict]:
        """Get satellites by category"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM satellites WHERE category = ? ORDER BY name", (category,)
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting satellites by category: {e}")
            return []
    
    def get_all_satellites(self, limit: int = 1000) -> List[Dict]:
        """Get all satellites"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM satellites ORDER BY name LIMIT ?", (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting all satellites: {e}")
            return []
    
    def add_tle_data(self, tle_data: Dict) -> bool:
        """Add TLE data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO tle_data (norad_id, line1, line2, epoch, source)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    tle_data['norad_id'],
                    tle_data['line1'],
                    tle_data['line2'],
                    tle_data.get('epoch'),
                    tle_data.get('source', 'unknown')
                ))
            return True
        except Exception as e:
            logger.error(f"Error adding TLE data: {e}")
            return False
    
    def get_latest_tle(self, norad_id: str) -> Optional[Dict]:
        """Get latest TLE data for satellite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM tle_data 
                    WHERE norad_id = ? 
                    ORDER BY created_at DESC LIMIT 1
                """, (norad_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting TLE data: {e}")
            return None
    
    def get_tle_history(self, norad_id: str, days_back: int = 7) -> List[Dict]:
        """Get TLE data history for satellite"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days_back)
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM tle_data 
                    WHERE norad_id = ? AND created_at > ?
                    ORDER BY created_at DESC
                """, (norad_id, cutoff_time.isoformat()))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting TLE history: {e}")
            return []
    
    def add_prediction(self, prediction_data: Dict) -> bool:
        """Add prediction result"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                prediction = prediction_data.get('prediction', {})
                position = prediction.get('position', [0, 0, 0])
                velocity = prediction.get('velocity', [0, 0, 0])
                
                conn.execute("""
                    INSERT INTO predictions 
                    (norad_id, prediction_time, target_time, hours_ahead,
                     position_x, position_y, position_z,
                     velocity_x, velocity_y, velocity_z,
                     method, metrics_json, confidence_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction_data['norad_id'],
                    prediction_data['prediction_time'],
                    prediction_data['target_time'],
                    prediction_data['hours_ahead'],
                    position[0], position[1], position[2],
                    velocity[0], velocity[1], velocity[2],
                    prediction_data['method'],
                    json.dumps(prediction_data.get('metrics')),
                    json.dumps(prediction_data.get('confidence'))
                ))
            return True
        except Exception as e:
            logger.error(f"Error adding prediction: {e}")
            return False
    
    def get_predictions(self, norad_id: str, limit: int = 50) -> List[Dict]:
        """Get recent predictions for satellite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM predictions 
                    WHERE norad_id = ? 
                    ORDER BY created_at DESC LIMIT ?
                """, (norad_id, limit))
                
                results = []
                for row in cursor.fetchall():
                    prediction = dict(row)
                    # Parse JSON fields
                    if prediction['metrics_json']:
                        try:
                            prediction['metrics'] = json.loads(prediction['metrics_json'])
                        except:
                            prediction['metrics'] = None
                    if prediction['confidence_json']:
                        try:
                            prediction['confidence'] = json.loads(prediction['confidence_json'])
                        except:
                            prediction['confidence'] = None
                    results.append(prediction)
                
                return results
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return []
    
    def get_prediction_accuracy(self, model_name: str, days_back: int = 7) -> Dict:
        """Get prediction accuracy statistics for a model"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days_back)
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT metrics_json FROM predictions 
                    WHERE method = ? AND created_at > ? AND metrics_json IS NOT NULL
                """, (model_name, cutoff_time.isoformat()))
                
                rmse_values = []
                mae_values = []
                
                for row in cursor.fetchall():
                    try:
                        metrics = json.loads(row['metrics_json'])
                        if 'position_metrics' in metrics:
                            rmse_values.append(metrics['position_metrics']['rmse'])
                            mae_values.append(metrics['position_metrics']['mae'])
                    except:
                        continue
                
                if rmse_values:
                    return {
                        'model_name': model_name,
                        'sample_count': len(rmse_values),
                        'avg_rmse': sum(rmse_values) / len(rmse_values),
                        'avg_mae': sum(mae_values) / len(mae_values),
                        'max_rmse': max(rmse_values),
                        'min_rmse': min(rmse_values),
                        'time_period_days': days_back
                    }
                else:
                    return {'model_name': model_name, 'sample_count': 0}
                    
        except Exception as e:
            logger.error(f"Error getting prediction accuracy: {e}")
            return {}
    
    def add_trajectory(self, trajectory_data: Dict) -> bool:
        """Add trajectory data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trajectories 
                    (norad_id, start_time, end_time, points_json, orbital_period)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    trajectory_data['norad_id'],
                    trajectory_data['start_time'],
                    trajectory_data['end_time'],
                    json.dumps(trajectory_data['trajectory']),
                    trajectory_data.get('orbital_period')
                ))
            return True
        except Exception as e:
            logger.error(f"Error adding trajectory: {e}")
            return False
    
    def get_trajectory(self, norad_id: str, start_time: str = None) -> Optional[Dict]:
        """Get trajectory data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if start_time:
                    cursor = conn.execute("""
                        SELECT * FROM trajectories 
                        WHERE norad_id = ? AND start_time >= ?
                        ORDER BY created_at DESC LIMIT 1
                    """, (norad_id, start_time))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM trajectories 
                        WHERE norad_id = ? 
                        ORDER BY created_at DESC LIMIT 1
                    """, (norad_id,))
                
                row = cursor.fetchone()
                if row:
                    trajectory = dict(row)
                    try:
                        trajectory['trajectory'] = json.loads(trajectory['points_json'])
                    except:
                        trajectory['trajectory'] = []
                    return trajectory
                return None
        except Exception as e:
            logger.error(f"Error getting trajectory: {e}")
            return None
    
    def add_pass_prediction(self, pass_data: Dict) -> bool:
        """Add satellite pass prediction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO pass_predictions 
                    (norad_id, observer_lat, observer_lon, observer_alt,
                     start_time, end_time, max_elevation, max_elevation_time,
                     duration_minutes, points_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pass_data['norad_id'],
                    pass_data['observer_lat'],
                    pass_data['observer_lon'],
                    pass_data.get('observer_alt', 0),
                    pass_data['start_time'],
                    pass_data['end_time'],
                    pass_data['max_elevation'],
                    pass_data['max_elevation_time'],
                    pass_data['duration_minutes'],
                    json.dumps(pass_data.get('points', []))
                ))
            return True
        except Exception as e:
            logger.error(f"Error adding pass prediction: {e}")
            return False
    
    def get_pass_predictions(self, norad_id: str, observer_lat: float, 
                           observer_lon: float, days_ahead: int = 7) -> List[Dict]:
        """Get pass predictions for location"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=1)  # Recent predictions only
            future_time = datetime.utcnow() + timedelta(days=days_ahead)
            
            # Use tolerance for location matching
            lat_tolerance = 0.1
            lon_tolerance = 0.1
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM pass_predictions 
                    WHERE norad_id = ? 
                    AND observer_lat BETWEEN ? AND ?
                    AND observer_lon BETWEEN ? AND ?
                    AND start_time BETWEEN ? AND ?
                    AND created_at > ?
                    ORDER BY start_time
                """, (
                    norad_id,
                    observer_lat - lat_tolerance, observer_lat + lat_tolerance,
                    observer_lon - lon_tolerance, observer_lon + lon_tolerance,
                    datetime.utcnow().isoformat(), future_time.isoformat(),
                    cutoff_time.isoformat()
                ))
                
                results = []
                for row in cursor.fetchall():
                    pass_pred = dict(row)
                    if pass_pred['points_json']:
                        try:
                            pass_pred['points'] = json.loads(pass_pred['points_json'])
                        except:
                            pass_pred['points'] = []
                    results.append(pass_pred)
                
                return results
        except Exception as e:
            logger.error(f"Error getting pass predictions: {e}")
            return []
    
    def add_model_performance(self, performance_data: Dict) -> bool:
        """Add model performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO model_performance 
                    (model_name, test_period, sample_count, 
                     mean_position_error, mean_velocity_error, performance_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    performance_data['model_name'],
                    performance_data.get('test_period'),
                    performance_data.get('sample_count'),
                    performance_data.get('mean_position_error'),
                    performance_data.get('mean_velocity_error'),
                    json.dumps(performance_data)
                ))
            return True
        except Exception as e:
            logger.error(f"Error adding model performance: {e}")
            return False
    
    def get_model_performance(self, model_name: str = None, days_back: int = 30) -> List[Dict]:
        """Get model performance history"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days_back)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if model_name:
                    cursor = conn.execute("""
                        SELECT * FROM model_performance 
                        WHERE model_name = ? AND created_at > ?
                        ORDER BY created_at DESC
                    """, (model_name, cutoff_time.isoformat()))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM model_performance 
                        WHERE created_at > ?
                        ORDER BY created_at DESC
                    """, (cutoff_time.isoformat(),))
                
                results = []
                for row in cursor.fetchall():
                    perf = dict(row)
                    if perf['performance_json']:
                        try:
                            perf_data = json.loads(perf['performance_json'])
                            perf.update(perf_data)
                        except:
                            pass
                    results.append(perf)
                
                return results
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return []
    
    def cache_set(self, key: str, value: Any, timeout_seconds: int = 3600) -> bool:
        """Set cache value"""
        try:
            expires_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache (key, value, expires_at)
                    VALUES (?, ?, ?)
                """, (key, json.dumps(value), expires_at.isoformat()))
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT value FROM cache 
                    WHERE key = ? AND expires_at > ?
                """, (key, datetime.utcnow().isoformat()))
                
                row = cursor.fetchone()
                if row:
                    try:
                        return json.loads(row['value'])
                    except:
                        return None
                return None
        except Exception as e:
            logger.error(f"Error getting cache: {e}")
            return None
    
    def cache_delete(self, key: str) -> bool:
        """Delete cache value"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            return True
        except Exception as e:
            logger.error(f"Error deleting cache: {e}")
            return False
    
    def cache_clear_expired(self) -> int:
        """Clear expired cache entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM cache WHERE expires_at < ?", 
                    (datetime.utcnow().isoformat(),)
                )
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")
            return 0
    
    def cleanup_expired(self):
        """Clean up expired and old data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.utcnow()
                
                # Clean expired cache
                cache_deleted = conn.execute(
                    "DELETE FROM cache WHERE expires_at < ?", 
                    (now.isoformat(),)
                ).rowcount
                
                # Clean old TLE data (keep last 30 days)
                cutoff = now - timedelta(days=30)
                tle_deleted = conn.execute(
                    "DELETE FROM tle_data WHERE created_at < ?", 
                    (cutoff.isoformat(),)
                ).rowcount
                
                # Clean old predictions (keep last 7 days)
                cutoff = now - timedelta(days=7)
                pred_deleted = conn.execute(
                    "DELETE FROM predictions WHERE created_at < ?", 
                    (cutoff.isoformat(),)
                ).rowcount
                
                # Clean old trajectories (keep last 3 days)
                cutoff = now - timedelta(days=3)
                traj_deleted = conn.execute(
                    "DELETE FROM trajectories WHERE created_at < ?", 
                    (cutoff.isoformat(),)
                ).rowcount
                
                # Clean old pass predictions (keep last 1 day)
                cutoff = now - timedelta(days=1)
                pass_deleted = conn.execute(
                    "DELETE FROM pass_predictions WHERE created_at < ?", 
                    (cutoff.isoformat(),)
                ).rowcount
                
                logger.info(f"Database cleanup completed: cache={cache_deleted}, "
                           f"tle={tle_deleted}, predictions={pred_deleted}, "
                           f"trajectories={traj_deleted}, passes={pass_deleted}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Count records in each table
                tables = ['satellites', 'tle_data', 'predictions', 'trajectories', 'pass_predictions', 'cache']
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                # Recent activity (last 24 hours)
                cutoff = datetime.utcnow() - timedelta(hours=24)
                cursor = conn.execute("SELECT COUNT(*) FROM predictions WHERE created_at > ?", (cutoff.isoformat(),))
                stats['recent_predictions'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM tle_data WHERE created_at > ?", (cutoff.isoformat(),))
                stats['recent_tle_updates'] = cursor.fetchone()[0]
                
                # Database size
                cursor = conn.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                size_result = cursor.fetchone()
                stats['database_size_bytes'] = size_result[0] if size_result else 0
                
                # Top categories
                cursor = conn.execute("""
                    SELECT category, COUNT(*) as count 
                    FROM satellites 
                    WHERE category IS NOT NULL 
                    GROUP BY category 
                    ORDER BY count DESC 
                    LIMIT 5
                """)
                stats['top_categories'] = dict(cursor.fetchall())
                
                # Most active satellites (by prediction count)
                cursor = conn.execute("""
                    SELECT s.name, COUNT(p.id) as prediction_count
                    FROM satellites s 
                    LEFT JOIN predictions p ON s.norad_id = p.norad_id
                    WHERE p.created_at > ?
                    GROUP BY s.norad_id, s.name
                    ORDER BY prediction_count DESC
                    LIMIT 5
                """, (cutoff.isoformat(),))
                stats['most_predicted_satellites'] = dict(cursor.fetchall())
                
                stats['last_updated'] = datetime.utcnow().isoformat()
                
                return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def export_data(self, output_file: str, table_name: str = None, 
                   include_large_fields: bool = False):
        """Export data to JSON file"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                export_data = {}
                
                if table_name:
                    tables = [table_name]
                else:
                    tables = ['satellites', 'tle_data', 'predictions', 'trajectories', 'pass_predictions']
                
                for table in tables:
                    if table == 'trajectories' and not include_large_fields:
                        # Exclude large JSON fields for basic export
                        cursor = conn.execute(f"""
                            SELECT id, norad_id, start_time, end_time, orbital_period, created_at 
                            FROM {table}
                        """)
                    elif table == 'predictions' and not include_large_fields:
                        cursor = conn.execute(f"""
                            SELECT id, norad_id, prediction_time, target_time, hours_ahead,
                                   position_x, position_y, position_z,
                                   velocity_x, velocity_y, velocity_z,
                                   method, created_at
                            FROM {table}
                        """)
                    else:
                        cursor = conn.execute(f"SELECT * FROM {table}")
                    
                    export_data[table] = [dict(row) for row in cursor.fetchall()]
                
                # Add metadata
                export_data['metadata'] = {
                    'export_time': datetime.utcnow().isoformat(),
                    'database_statistics': self.get_statistics(),
                    'include_large_fields': include_large_fields
                }
                
                with open(output_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                logger.info(f"Data exported to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
    
    def import_data(self, input_file: str, overwrite: bool = False):
        """Import data from JSON file"""
        try:
            with open(input_file, 'r') as f:
                import_data = json.load(f)
            
            with sqlite3.connect(self.db_path) as conn:
                # Import satellites
                if 'satellites' in import_data:
                    for sat_data in import_data['satellites']:
                        if overwrite:
                            conn.execute("""
                                INSERT OR REPLACE INTO satellites 
                                (norad_id, name, category, country, launch_date, status, mass_kg, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                sat_data['norad_id'], sat_data['name'], sat_data.get('category'),
                                sat_data.get('country'), sat_data.get('launch_date'),
                                sat_data.get('status'), sat_data.get('mass_kg'),
                                sat_data.get('updated_at', datetime.utcnow().isoformat())
                            ))
                        else:
                            conn.execute("""
                                INSERT OR IGNORE INTO satellites 
                                (norad_id, name, category, country, launch_date, status, mass_kg, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                sat_data['norad_id'], sat_data['name'], sat_data.get('category'),
                                sat_data.get('country'), sat_data.get('launch_date'),
                                sat_data.get('status'), sat_data.get('mass_kg'),
                                sat_data.get('updated_at', datetime.utcnow().isoformat())
                            ))
                
                logger.info(f"Data imported from {input_file}")
        except Exception as e:
            logger.error(f"Error importing data: {e}")
    
    def vacuum_database(self):
        """Optimize database by reclaiming space"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")
            logger.info("Database vacuum completed")
        except Exception as e:
            logger.error(f"Error during vacuum: {e}")
    
    def backup_database(self, backup_path: str):
        """Create database backup"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
        except Exception as e:
            logger.error(f"Error creating backup: {e}")

# Global database instance
db_manager = DatabaseManager()