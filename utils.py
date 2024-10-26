def validate_tle(line1, line2):
    """Validates TLE format and returns (bool, str) tuple with validation status and error message"""
    
    if not line1 or not line2:
        return False, "Both TLE lines are required"
        
    line1 = line1.strip()
    line2 = line2.strip()
    
    if len(line1) != 69:
        return False, f"Line 1 must be exactly 69 characters (current length: {len(line1)})"
    if len(line2) != 69:
        return False, f"Line 2 must be exactly 69 characters (current length: {len(line2)})"
    
    if not line1.startswith('1 '):
        return False, "Line 1 must start with '1 '"
    if not line2.startswith('2 '):
        return False, "Line 2 must start with '2 '"
    
    sat_num1 = line1[2:7].strip()
    sat_num2 = line2[2:7].strip()
    if sat_num1 != sat_num2:
        return False, f"Satellite numbers don't match: {sat_num1} vs {sat_num2}"
    
    return True, "Valid TLE format"

def format_prediction(prediction):
    """Formats prediction results"""
    return {
        'position': prediction[0][:3].tolist(),
        'velocity': prediction[0][3:].tolist()
    }
