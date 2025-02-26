import json
import logging
from typing import Tuple, Optional, Dict, Any

def extract_alt_az_from_json(json_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract altitude and azimuth values from JSON data, handling different possible structures
    
    Parameters:
    json_data: Parsed JSON data
    
    Returns:
    (altitude, azimuth) or (None, None) if not found
    """
    logger = logging.getLogger(__name__)
    alt = None
    az = None
    
    # Try different possible JSON structures
    # Structure 1: motors top-level key with ALT/AZ
    if 'motors' in json_data:
        if isinstance(json_data['motors'], dict):
            alt = json_data['motors'].get('ALT')
            az = json_data['motors'].get('AZ')
            if alt is not None and az is not None:
                logger.debug("Found ALT/AZ in 'motors' dict")
                return alt, az
    
    # Structure 2: position top-level key with altitude/azimuth
    if 'position' in json_data:
        if isinstance(json_data['position'], dict):
            alt = json_data['position'].get('altitude')
            az = json_data['position'].get('azimuth')
            if alt is not None and az is not None:
                logger.debug("Found altitude/azimuth in 'position' dict")
                return alt, az
                
    # Structure 3: telescope key with position or motors subkeys
    if 'telescope' in json_data and isinstance(json_data['telescope'], dict):
        telescope = json_data['telescope']
        
        # Check for motors subkey
        if 'motors' in telescope and isinstance(telescope['motors'], dict):
            alt = telescope['motors'].get('ALT')
            az = telescope['motors'].get('AZ')
            if alt is not None and az is not None:
                logger.debug("Found ALT/AZ in 'telescope.motors' dict")
                return alt, az
                
        # Check for position subkey
        if 'position' in telescope and isinstance(telescope['position'], dict):
            alt = telescope['position'].get('altitude')
            az = telescope['position'].get('azimuth')
            if alt is not None and az is not None:
                logger.debug("Found altitude/azimuth in 'telescope.position' dict")
                return alt, az
    
    # Structure 4: Look for any key with 'alt' and 'az' in the name (case insensitive)
    for key, value in json_data.items():
        if isinstance(value, dict):
            alt_candidates = [k for k in value.keys() if 'alt' in k.lower()]
            az_candidates = [k for k in value.keys() if 'az' in k.lower()]
            
            if alt_candidates and az_candidates:
                alt = value[alt_candidates[0]]
                az = value[az_candidates[0]]
                logger.debug(f"Found ALT/AZ in '{key}' dict with keys {alt_candidates[0]}/{az_candidates[0]}")
                return alt, az
    
    # If all attempts fail, return None
    logger.error("Could not find altitude/azimuth in JSON data")
    logger.debug(f"Available top-level keys: {list(json_data.keys())}")
    return None, None

def load_and_parse_json(json_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[float]]:
    """
    Load JSON file and extract altitude and azimuth values
    
    Parameters:
    json_path: Path to JSON file
    
    Returns:
    (json_data, altitude, azimuth) or (None, None, None) if error occurs
    """
    logger = logging.getLogger(__name__)
    
    try:
        with open(json_path) as f:
            json_data = json.load(f)
            
        # Extract altitude and azimuth
        alt, az = extract_alt_az_from_json(json_data)
        
        if alt is not None and az is not None:
            logger.info(f"Extracted Alt={alt:.2f}°, Az={az:.2f}° from {json_path}")
            return json_data, alt, az
        else:
            logger.error(f"Could not extract Alt/Az from {json_path}")
            return json_data, None, None
            
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {json_path}: {e}")
        return None, None, None
    except FileNotFoundError as e:
        logger.error(f"File not found: {json_path}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error loading JSON file {json_path}: {e}")
        return None, None, None
