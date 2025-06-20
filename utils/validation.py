from typing import Tuple, Dict, Any
from datetime import datetime
import math

def validate_tle(line1: str, line2: str) -> Tuple[bool, str]:
    """
    Simple TLE validation
    
    Args:
        line1: First line of TLE data
        line2: Second line of TLE data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    
    if not line1 or not line2:
        return False, "Both TLE lines are required"
        
    line1 = line1.strip()
    line2 = line2.strip()
    
    # Check line lengths
    if len(line1) != 69:
        return False, f"Line 1 must be exactly 69 characters (current: {len(line1)})"
    if len(line2) != 69:
        return False, f"Line 2 must be exactly 69 characters (current: {len(line2)})"
    
    # Check line identifiers
    if not line1.startswith('1 '):
        return False, "Line 1 must start with '1 '"
    if not line2.startswith('2 '):
        return False, "Line 2 must start with '2 '"
    
    # Extract and validate satellite numbers
    try:
        sat_num1 = line1[2:7].strip()
        sat_num2 = line2[2:7].strip()
        
        if sat_num1 != sat_num2:
            return False, f"Satellite numbers don't match: {sat_num1} vs {sat_num2}"
            
        # Validate satellite number format
        if not sat_num1.isdigit():
            return False, f"Invalid satellite number format: {sat_num1}"
            
    except (IndexError, ValueError):
        return False, "Invalid satellite number format"
    
    return True, "Valid TLE format"

def validate_prediction_timeframe(hours: int) -> Tuple[bool, str]:
    """
    Validate prediction timeframe
    
    Args:
        hours: Number of hours for prediction
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    MAX_PREDICTION_HOURS = 168  # 7 days
    
    if hours <= 0:
        return False, "Prediction timeframe must be positive"
        
    if hours > MAX_PREDICTION_HOURS:
        return False, f"Prediction timeframe cannot exceed {MAX_PREDICTION_HOURS} hours"
        
    return True, "Valid prediction timeframe"

def validate_coordinates(latitude: float, longitude: float) -> Tuple[bool, str]:
    """
    Validate geographical coordinates
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (-90 <= latitude <= 90):
        return False, f"Invalid latitude: {latitude}° (must be -90 to 90°)"
        
    if not (-180 <= longitude <= 180):
        return False, f"Invalid longitude: {longitude}° (must be -180 to 180°)"
        
    return True, "Valid coordinates"

def validate_norad_id(norad_id: str) -> Tuple[bool, str]:
    """
    Validate NORAD catalog number
    
    Args:
        norad_id: NORAD catalog number
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not norad_id:
        return False, "NORAD ID is required"
        
    if not str(norad_id).isdigit():
        return False, "NORAD ID must contain only digits"
        
    norad_int = int(norad_id)
    if not (1 <= norad_int <= 99999):
        return False, "NORAD ID must be between 1 and 99999"
        
    return True, "Valid NORAD ID"

def sanitize_satellite_name(name: str) -> str:
    """
    Sanitize satellite name for safe usage
    
    Args:
        name: Raw satellite name
        
    Returns:
        Sanitized satellite name
    """
    if not name:
        return ""
        
    # Remove special characters and normalize spaces
    import re
    sanitized = re.sub(r'[^\w\s\-\(\)]', '', name)
    sanitized = re.sub(r'\s+', ' ', sanitized)
    return sanitized.strip().upper()

def validate_api_response(response_data: dict, required_fields: list) -> Tuple[bool, str]:
    """
    Validate API response structure
    
    Args:
        response_data: API response data
        required_fields: List of required field names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(response_data, dict):
        return False, "Response must be a dictionary"
        
    missing_fields = []
    for field in required_fields:
        if field not in response_data:
            missing_fields.append(field)
            
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
        
    return True, "Valid API response"

def format_prediction_result(prediction: Any) -> Dict:
    """
    Format prediction results for API response
    
    Args:
        prediction: Raw prediction data
        
    Returns:
        Formatted prediction result
    """
    try:
        if isinstance(prediction, dict):
            # Already formatted
            if 'position' in prediction and 'velocity' in prediction:
                return {
                    'position': prediction['position'],
                    'velocity': prediction['velocity']
                }
            
            # Extract from nested structure
            if 'prediction' in prediction:
                pred = prediction['prediction']
                return {
                    'position': pred.get('position', [0, 0, 0]),
                    'velocity': pred.get('velocity', [0, 0, 0])
                }
        
        # Fallback format
        return {
            'position': [0, 0, 0],
            'velocity': [0, 0, 0]
        }
        
    except Exception as e:
        return {
            'position': [0, 0, 0],
            'velocity': [0, 0, 0],
            'error': str(e)
        }

def validate_time_range(start_time: datetime, end_time: datetime) -> Tuple[bool, str]:
    """
    Validate time range for predictions
    
    Args:
        start_time: Start datetime
        end_time: End datetime
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if start_time >= end_time:
        return False, "Start time must be before end time"
        
    time_diff = end_time - start_time
    max_duration = 168 * 3600  # 168 hours in seconds
    
    if time_diff.total_seconds() > max_duration:
        return False, f"Time range cannot exceed 168 hours"
        
    return True, "Valid time range"