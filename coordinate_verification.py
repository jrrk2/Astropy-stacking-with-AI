from astropy.coordinates import SkyCoord
import astropy.units as u
import logging
from typing import Optional, Tuple

def verify_coordinates(calc_ra, calc_dec, target_coords, max_separation_deg=1.0, target_name=None):
    """
    Verify calculated coordinates are close to target object
    
    Parameters:
    calc_ra: Calculated RA in degrees
    calc_dec: Calculated Dec in degrees
    target_coords: SkyCoord object for target coordinates
    max_separation_deg: Maximum allowed separation in degrees
    target_name: Optional name of target for logging
    
    Returns:
    (is_valid, separation_deg, target_coords)
    """
    logger = logging.getLogger(__name__)
    
    if target_coords is None:
        logger.warning("Target coordinates not provided for verification")
        return False, None, None
        
    # Create SkyCoord for calculated position
    try:
        calc_coords = SkyCoord(calc_ra * u.deg, calc_dec * u.deg)
    except Exception as e:
        logger.error(f"Error creating SkyCoord for calculated coordinates: {e}")
        return False, None, None
    
    # Calculate separation
    try:
        separation = calc_coords.separation(target_coords)
        separation_deg = float(separation.deg)
    except Exception as e:
        logger.error(f"Error calculating separation: {e}")
        return False, None, None
    
    # Determine if valid based on separation
    is_valid = separation_deg <= max_separation_deg
    
    # Log results
    target_desc = target_name if target_name else "target"
    if not is_valid:
        logger.warning(f"Calculated position is {separation_deg:.2f}° from {target_desc}")
        logger.warning(f"Calculated: RA={calc_ra:.4f}°, Dec={calc_dec:.4f}°")
        logger.warning(f"Expected:   RA={target_coords.ra.deg:.4f}°, Dec={target_coords.dec.deg:.4f}°")
    else:
        logger.info(f"Position OK - {separation_deg:.2f}° from {target_desc}")
    
    return is_valid, separation_deg, target_coords

def verify_with_astrometry(
    fits_path, 
    ra_hint, 
    dec_hint, 
    target_coords, 
    max_separation_deg=1.0, 
    target_name=None,
    config=None
):
    """
    Verify image coordinates using astrometry plate solving
    
    Parameters:
    fits_path: Path to FITS file
    ra_hint: RA hint in degrees
    dec_hint: Dec hint in degrees
    target_coords: SkyCoord object for target coordinates
    max_separation_deg: Maximum allowed separation in degrees
    target_name: Optional name of target for logging
    config: Configuration object
    
    Returns:
    (success, separation_deg, solved_coords)
    """
    logger = logging.getLogger(__name__)
    
    if not target_coords:
        logger.warning("Target coordinates not provided, skipping astrometry verification")
        return False, None, None
    
    # Try to plate solve the image
    success, result = solve_with_astrometry(
        fits_path, 
        ra_hint=ra_hint, 
        dec_hint=dec_hint,
        config=config
    )
    
    if not success:
        logger.warning(f"Astrometry solve failed: {result}")
        return False, None, None
    
    # Extract solved RA/Dec from WCS header
    solved_ra = result.get('CRVAL1')
    solved_dec = result.get('CRVAL2')
    
    if solved_ra is None or solved_dec is None:
        logger.warning("Could not extract coordinates from astrometry solution")
        return False, None, None
    
    # Create SkyCoord for solved coordinates
    try:
        solved_coords = SkyCoord(solved_ra * u.deg, solved_dec * u.deg)
    except Exception as e:
        logger.error(f"Error creating SkyCoord for solved coordinates: {e}")
        return False, None, None
    
    # Calculate separation
    try:
        separation = solved_coords.separation(target_coords)
        separation_deg = float(separation.deg)
    except Exception as e:
        logger.error(f"Error calculating separation: {e}")
        return False, None, None
    
    # Determine if valid based on separation
    is_valid = separation_deg <= max_separation_deg
    
    # Log results
    target_desc = target_name if target_name else "target"
    if not is_valid:
        logger.warning(f"Astrometry separation too large: {separation_deg:.2f}° from {target_desc}")
        logger.debug(f"Solved position: RA={solved_ra:.4f}°, Dec={solved_dec:.4f}°")
        logger.debug(f"Target position: RA={target_coords.ra.deg:.4f}°, Dec={target_coords.dec.deg:.4f}°")
    else:
        logger.info(f"Position verified by astrometry - {separation_deg:.2f}° from {target_desc}")
    
    return is_valid, separation_deg, solved_coords
