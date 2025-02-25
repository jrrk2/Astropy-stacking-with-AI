import os
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
import logging
from pathlib import Path
import argparse
from scipy.stats import pearsonr
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SimbadResult:
    """Result from SIMBAD query"""
    identifier: str
    ra_deg: float
    dec_deg: float
    mag_v: Optional[float] = None

def parse_ra(ra_str: str) -> float:
    """Parse RA string from SIMBAD to decimal degrees"""
    if not ra_str or ra_str.strip() == "":
        raise ValueError("Empty RA string")
        
    # Handle decimal degrees format
    if " " not in ra_str and ":" not in ra_str and "h" not in ra_str:
        return float(ra_str)
        
    # Handle sexagesimal format (replace h, m, s with :)
    ra_str = ra_str.replace('h', ':').replace('m', ':').replace('s', '')
    
    # Split by : or space
    parts = ra_str.replace(':', ' ').split()
    
    # Convert to decimal degrees
    if len(parts) >= 3:
        h = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        ra_deg = 15.0 * (h + m/60.0 + s/3600.0)  # 15 degrees per hour
        return ra_deg
    elif len(parts) == 2:
        h = float(parts[0])
        m = float(parts[1])
        ra_deg = 15.0 * (h + m/60.0)
        return ra_deg
    else:
        try:
            return float(parts[0]) * 15.0  # Assuming hours
        except ValueError:
            raise ValueError(f"Could not parse RA: {ra_str}")

def parse_dec(dec_str: str) -> float:
    """Parse Dec string from SIMBAD to decimal degrees"""
    if not dec_str or dec_str.strip() == "":
        raise ValueError("Empty Dec string")
        
    # Handle decimal degrees format
    if " " not in dec_str and ":" not in dec_str and "d" not in dec_str:
        return float(dec_str)
        
    # Handle sexagesimal format (replace d, m, s with :)
    dec_str = dec_str.replace('d', ':').replace('m', ':').replace('s', '')
    
    # Split by : or space
    parts = dec_str.replace(':', ' ').split()
    
    # Handle sign
    sign = 1.0
    if parts[0].startswith('-'):
        sign = -1.0
        parts[0] = parts[0][1:]
    elif parts[0].startswith('+'):
        parts[0] = parts[0][1:]
        
    # Convert to decimal degrees
    if len(parts) >= 3:
        d = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        dec_deg = sign * (d + m/60.0 + s/3600.0)
        return dec_deg
    elif len(parts) == 2:
        d = float(parts[0])
        m = float(parts[1])
        dec_deg = sign * (d + m/60.0)
        return dec_deg
    else:
        try:
            return sign * float(parts[0])
        except ValueError:
            raise ValueError(f"Could not parse Dec: {dec_str}")

def query_simbad(target: str) -> Optional[SimbadResult]:
    """Query SIMBAD database for target coordinates using VOTable format"""
    url = "http://simbad.u-strasbg.fr/simbad/sim-id"
    params = {
        "output.format": "VOTABLE",
        "output.params": "main_id,ra,dec,flux(V),flux_unit(mag)",
        "Ident": target
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        logger.info(f"Query URL: {response.url}")
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Register namespace
        ns = {'v': 'http://www.ivoa.net/xml/VOTable/v1.2'}
        
        # First look for error response
        info = root.find('.//v:INFO[@name="Error"]', ns)
        if info is not None:
            logger.error(f"SIMBAD Error: {info.get('value')}")
            return None
            
        # Find the TABLEDATA section
        tabledata = root.find('.//v:TABLEDATA', ns)
        if tabledata is None:
            logger.error("No TABLEDATA found in response")
            logger.debug("Full response:")
            logger.debug(response.content.decode())
            return None
            
        # Get first row
        tr = tabledata.find('v:TR', ns)
        if tr is None:
            logger.error("No data row found")
            return None
            
        # Get cells
        cells = tr.findall('v:TD', ns)
        logger.debug(f"Found {len(cells)} data cells")
        for i, cell in enumerate(cells):
            logger.debug(f"Cell {i}: {cell.text}")
        
        if len(cells) >= 3:
            identifier = cells[0].text
            ra_str = cells[1].text
            dec_str = cells[2].text
            mag_str = cells[3].text if len(cells) > 3 else None
            
            logger.debug(f"Raw coordinates: RA='{ra_str}', Dec='{dec_str}'")
            
            if ra_str and dec_str:
                ra_deg = parse_ra(ra_str)
                dec_deg = parse_dec(dec_str)
                mag_v = float(mag_str) if mag_str and mag_str.strip() != "" else None
                
                logger.info(f"Parsed: RA={ra_deg:.4f}°, Dec={dec_deg:.4f}°")
                return SimbadResult(identifier, ra_deg, dec_deg, mag_v)
        
        logger.error("Could not find coordinate data in response")
        return None
        
    except Exception as e:
        logger.error(f"SIMBAD query error: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_sources(fits_file: str, fwhm: float = 3.0, threshold_sigma: float = 5.0,
                   sharplo: float = 0.5, sharphi: float = 2.0,
                   roundlo: float = 0.5, roundhi: float = 1.5,
                   minsize: int = 2, min_flux_ratio: float = 0.1) -> Table | None:
    """
    Extract sources from the image using photutils with improved filtering.
    
    Args:
        fits_file: Path to the FITS file
        fwhm: Full width at half maximum of stars in pixels
        threshold_sigma: Detection threshold in sigma above background
        sharplo: Lower bound on sharpness for star detection
        sharphi: Upper bound on sharpness for star detection
        roundlo: Lower bound on roundness for star detection
        roundhi: Upper bound on roundness for star detection
        minsize: Minimum size of stars in pixels
        min_flux_ratio: Minimum flux ratio compared to brightest star
        
    Returns:
        astropy.table.Table: Table of extracted sources or None if extraction fails
    """
    try:
        if not os.path.exists(fits_file):
            logger.error(f"FITS file not found: {fits_file}")
            return None
            
        # Read FITS file
        with fits.open(fits_file) as hdul:
            # Get primary HDU data
            data = hdul[0].data
            header = hdul[0].header
            
            # Extract field rotation angle and mount position if available
            rotation_angle = header.get('DER', None)
            alt = header.get('ALT', None)
            az = header.get('AZ', None)
            
            if rotation_angle is not None:
                logger.info(f"Derotator angle: {rotation_angle:.2f}°")
            else:
                logger.info("No derotator angle (DER) found in header")
                
            if alt is not None and az is not None:
                logger.info(f"Mount position: ALT={alt:.2f}°, AZ={az:.2f}°")
            else:
                logger.info("No ALT/AZ position found in header")
            
            if data is None:
                # Try the first extension if primary HDU has no data
                if len(hdul) > 1 and hdul[1].data is not None:
                    data = hdul[1].data
                    header = hdul[1].header
                else:
                    logger.error(f"No image data found in {fits_file}")
                    return None
            
            # Get WCS information
            try:
                wcs = WCS(header)
            except Exception as e:
                logger.error(f"Error parsing WCS: {str(e)}")
                return None
        
        # Estimate background - FIXED: Use SigmaClip class instead of float value
        try:
            sigma_clip = SigmaClip(sigma=3.0)  # Fixed: Using SigmaClip object
            bkg_estimator = MedianBackground()
            bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                              sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
            data_sub = data - bkg.background
            threshold = threshold_sigma * bkg.background_rms.mean()
        except Exception as e:
            logger.warning(f"Error estimating background: {str(e)}. Using simple statistics.")
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            data_sub = data - median
            threshold = threshold_sigma * std
        
        # Source detection with improved filtering for hot pixels
        daofind = DAOStarFinder(
            fwhm=fwhm,
            threshold=threshold,
            sharplo=sharplo,
            sharphi=sharphi,
            roundlo=roundlo,
            roundhi=roundhi,
            peakmax=None  # No upper limit on peak flux
        )
        
        sources = daofind(data_sub)
        
        if sources is None or len(sources) == 0:
            logger.warning(f"No sources detected in {fits_file}")
            return None
            
        # Filter out faint sources that might be noise
        if 'flux' in sources.colnames:
            max_flux = np.max(sources['flux'])
            flux_threshold = max_flux * min_flux_ratio
            bright_sources = sources[sources['flux'] > flux_threshold]
            
            # Skip if we've filtered out too many sources
            if len(bright_sources) < 10:
                logger.warning(f"Only {len(bright_sources)} sources above flux threshold. Using all sources.")
            else:
                sources = bright_sources
                logger.info(f"Filtered to {len(sources)} sources based on minimum flux threshold")
            
        # Apply additional filtering based on star shape if needed
        if 'sharpness' in sources.colnames and 'roundness1' in sources.colnames:
            good_sources = np.logical_and(
                np.logical_and(sources['sharpness'] > sharplo, sources['sharpness'] < sharphi),
                np.logical_and(sources['roundness1'] > roundlo, sources['roundness1'] < roundhi)
            )
            
            filtered_sources = sources[good_sources]
            
            # Skip if we've filtered out too many sources
            if len(filtered_sources) < 10:
                logger.warning(f"Only {len(filtered_sources)} sources after shape filtering. Using all sources.")
            else:
                sources = filtered_sources
                logger.info(f"Filtered to {len(sources)} sources based on star shape")
            
        # Add RA/Dec columns using WCS
        try:
            # Extract pixel coordinates
            x_coords = sources['xcentroid']
            y_coords = sources['ycentroid']
            
            # Convert pixel to world coordinates safely
            world_positions = wcs.pixel_to_world(x_coords, y_coords)
            
            # Ensure we have a SkyCoord object
            if hasattr(world_positions, 'ra') and hasattr(world_positions, 'dec'):
                sources['ALPHA_J2000'] = world_positions.ra.deg
                sources['DELTA_J2000'] = world_positions.dec.deg
            else:
                # Handle case where WCS transformation returns something unexpected
                logger.error("WCS transformation failed to return proper SkyCoord objects")
                return None
        except Exception as e:
            logger.error(f"Error in WCS coordinate transformation: {str(e)}")
            return None
        
        logger.info(f"Successfully extracted {len(sources)} sources")
        return sources
        
    except Exception as e:
        logger.error(f"Error extracting sources: {str(e)}")
        return None

def apply_rotation_correction(image_coords, center_coords, rotation_angle):
    """
    Apply a rotation correction to image coordinates based on field rotation angle.
    
    Args:
        image_coords: SkyCoord object with image source coordinates
        center_coords: SkyCoord object with field center coordinates
        rotation_angle: Rotation angle in degrees
        
    Returns:
        SkyCoord: Corrected coordinates
    """
    try:
        # For each source, calculate position angle and separation from center
        separations = center_coords.separation(image_coords)
        position_angles = center_coords.position_angle(image_coords)
        
        # Apply rotation (derotator angle adjustment)
        corrected_angles = position_angles + rotation_angle * u.deg
        
        # Calculate new positions
        new_coords = []
        for i in range(len(image_coords)):
            new_coord = center_coords.directional_offset_by(
                corrected_angles[i], separations[i])
            new_coords.append(new_coord)
        
        # Create new SkyCoord from list of coordinates
        return SkyCoord(new_coords)
    except Exception as e:
        logger.warning(f"Rotation correction failed: {e}")
        return image_coords

def cross_match_gaia(ra_center: float, dec_center: float, search_radius: float = 0.5, mag_limit: float = 15.0) -> Table | None:
    """
    Cross-match extracted sources with Gaia DR3 catalog.
    
    Args:
        ra_center: RA of field center in degrees
        dec_center: Dec of field center in degrees
        search_radius: Search radius in degrees
        mag_limit: Limiting magnitude for Gaia catalog query
        
    Returns:
        astropy.table.Table: Table of matched Gaia sources or None if query fails
    """
    try:
        # Validate coordinates
        if not (-360 <= ra_center <= 360 and -90 <= dec_center <= 90):
            logger.error(f"Invalid coordinates: RA={ra_center}, Dec={dec_center}")
            return None
            
        # Log the field center for debugging
        logger.info(f"Field center: RA={ra_center:.6f}°, Dec={dec_center:.6f}°")
        
        # Try astroquery.gaia first, fall back to local catalog if needed
        try:
            from astroquery.gaia import Gaia
            Gaia.ROW_LIMIT = -1  # No row limit
            
            # Create coordinate object
            coord = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.deg, u.deg))
            radius = search_radius * u.degree
            
            # Query Gaia DR3 - use supplied magnitude limit
            query = f"""
            SELECT 
                source_id, ra, dec, 
                phot_g_mean_mag, 
                parallax, parallax_error,                 
                pmra, pmdec,
                phot_bp_mean_mag, phot_rp_mean_mag
            FROM gaiadr3.gaia_source
            WHERE 
                1=CONTAINS(
                    POINT(ra, dec), 
                    CIRCLE({ra_center}, {dec_center}, {radius.value})
                )
                AND phot_g_mean_mag < {mag_limit}
            ORDER BY phot_g_mean_mag ASC  -- Sort by brightness
            """
            
            logger.info("Querying Gaia DR3 catalog...")
            job = Gaia.launch_job(query)
            results = job.get_results()
            
            # Verify we have necessary columns
            required_cols = ['ra', 'dec']
            for col in required_cols:
                if col not in results.colnames:
                    logger.error(f"Required column '{col}' missing from Gaia results")
                    return None
            
            if len(results) == 0:
                logger.warning(f"No Gaia sources found within {search_radius}° of field center")
            else:
                logger.info(f"Retrieved {len(results)} Gaia sources")
                
            return results
            
        except Exception as e:
            logger.error(f"Gaia catalog query via astroquery failed: {str(e)}")
            logger.info("Trying alternative catalog methods...")
            
            # Placeholder for alternative catalog methods if needed
            # This could include using a local catalog file or another service
            logger.error("No alternative catalog source available")
            return None
        
    except Exception as e:
        logger.error(f"Catalog query failed: {str(e)}")
        return None

def verify_pointing(fits_file: str, ra_center: float, dec_center: float, 
                   max_offset: float = 5.0, search_radius: float = 0.5,
                   fwhm: float = 3.0, threshold_sigma: float = 5.0,
                   sharplo: float = 0.5, sharphi: float = 2.0,
                   roundlo: float = 0.5, roundhi: float = 1.5,
                   minsize: int = 2, min_flux_ratio: float = 0.1,
                   apply_rotation: bool = True, mag_limit: float = 15.0) -> dict:
    """
    Verify telescope pointing accuracy by comparing extracted sources with Gaia catalog.
    
    Args:
        fits_file: Path to the FITS file
        ra_center: RA of field center in degrees
        dec_center: Dec of field center in degrees
        max_offset: Maximum allowed offset in arcseconds for matching
        search_radius: Search radius in degrees for catalog query
        fwhm: Full width at half maximum of stars in pixels
        threshold_sigma: Detection threshold in sigma above background
        sharplo: Lower bound on sharpness for star detection
        sharphi: Upper bound on sharpness for star detection
        roundlo: Lower bound on roundness for star detection
        roundhi: Upper bound on roundness for star detection
        minsize: Minimum size of stars in pixels
        min_flux_ratio: Minimum flux ratio compared to brightest star
        apply_rotation: Whether to apply rotation correction
        mag_limit: Limiting magnitude for Gaia catalog query
        
    Returns:
        dict: Dictionary containing analysis results
    """
    results = {
        'success': False,
        'n_extracted': 0,
        'n_catalog': 0,
        'n_matched': 0,
        'ra_offset_mean': None,
        'dec_offset_mean': None,
        'ra_offset_std': None,
        'dec_offset_std': None,
        'rotation_angle': None,
        'alt': None,
        'az': None
    }
    
    # Extract rotation angle and mount position from FITS header
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            results['rotation_angle'] = header.get('DER', None)
            results['alt'] = header.get('ALT', None)
            results['az'] = header.get('AZ', None)
    except Exception as e:
        logger.warning(f"Could not extract header data: {e}")
    
    # Create center coordinates for rotation correction
    center_coords = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.deg, u.deg))
    
    # Extract sources with improved filtering
    try:
        sources = extract_sources(
            fits_file,
            fwhm=fwhm,
            threshold_sigma=threshold_sigma,
            sharplo=sharplo,
            sharphi=sharphi,
            roundlo=roundlo,
            roundhi=roundhi,
            minsize=minsize,
            min_flux_ratio=min_flux_ratio
        )
        
        if sources is None:
            logger.warning("No sources extracted from the image")
            return results
            
        results['n_extracted'] = len(sources)
        
        # Get catalog sources using the provided coordinates
        catalog_sources = cross_match_gaia(ra_center, dec_center, search_radius=search_radius, mag_limit=mag_limit)
        if catalog_sources is None:
            logger.warning("No catalog sources retrieved")
            return results
            
        results['n_catalog'] = len(catalog_sources)
        
        if len(sources) == 0 or len(catalog_sources) == 0:
            logger.warning("No sources available for matching")
            return results
            
        # Verify required columns exist in the sources table
        required_cols = ['ALPHA_J2000', 'DELTA_J2000']
        for col in required_cols:
            if col not in sources.colnames:
                logger.error(f"Required column '{col}' missing from extracted sources")
                return results
    except Exception as e:
        logger.error(f"Error preparing sources for matching: {str(e)}")
        return results
        
    # Perform cross-matching
    try:
        # Ensure the required columns exist
        if 'ALPHA_J2000' not in sources.colnames or 'DELTA_J2000' not in sources.colnames:
            logger.error("Source table missing required RA/Dec columns")
            return results
            
        if 'ra' not in catalog_sources.colnames or 'dec' not in catalog_sources.colnames:
            logger.error("Catalog table missing required ra/dec columns")
            return results
        
        # Create SkyCoord objects for both sets of coordinates
        try:
            image_coords = SkyCoord(
                ra=sources['ALPHA_J2000'],
                dec=sources['DELTA_J2000'],
                unit=(u.deg, u.deg)
            )
            
            catalog_coords = SkyCoord(
                ra=catalog_sources['ra'],
                dec=catalog_sources['dec'],
                unit=(u.deg, u.deg)
            )
        except Exception as e:
            logger.error(f"Error creating SkyCoord objects: {str(e)}")
            return results
        
        # Get field rotation from results
        rotation_angle = results.get('rotation_angle')
        
        # If we have rotation information and center coordinates, apply correction
        original_image_coords = image_coords
        if apply_rotation and rotation_angle is not None:
            logger.info(f"Accounting for derotator angle: {rotation_angle:.2f}°")
            try:
                # Apply rotation correction
                image_coords = apply_rotation_correction(image_coords, center_coords, rotation_angle)
            except Exception as e:
                logger.warning(f"Rotation correction failed: {e}")
                # Fall back to original coords
                image_coords = original_image_coords
        
        # Match sources using progressive tolerances
        matching_tolerances = [1.0, 2.0, 3.0, 5.0, 10.0]  # in arcseconds
        matched_sources = None
        matched_catalog = None
        ra_offsets = None
        dec_offsets = None
        used_tolerance = max_offset
        
        for tolerance in matching_tolerances:
            if tolerance > max_offset:
                break
                
            # Match sources
            idx, d2d, _ = image_coords.match_to_catalog_sky(catalog_coords)
            match_mask = d2d < tolerance * u.arcsec
            
            temp_matched_sources = sources[match_mask]
            temp_matched_catalog = catalog_sources[idx[match_mask]]
            
            if len(temp_matched_sources) >= 3:  # We found enough matches
                matched_sources = temp_matched_sources
                matched_catalog = temp_matched_catalog
                used_tolerance = tolerance
                
                # Calculate offsets in arcseconds
                ra_offsets = (matched_sources['ALPHA_J2000'] - matched_catalog['ra']) * 3600
                dec_offsets = (matched_sources['DELTA_J2000'] - matched_catalog['dec']) * 3600
                
                logger.info(f"Found {len(matched_sources)} matches using {tolerance}\" tolerance")
                break
                
        # If we didn't find matches with the progressive approach, use the max tolerance
        if matched_sources is None:
            idx, d2d, _ = image_coords.match_to_catalog_sky(catalog_coords)
            match_mask = d2d < max_offset * u.arcsec
            
            matched_sources = sources[match_mask]
            matched_catalog = catalog_sources[idx[match_mask]]
            
            # Calculate offsets in arcseconds
            if len(matched_sources) > 0:
                ra_offsets = (matched_sources['ALPHA_J2000'] - matched_catalog['ra']) * 3600
                dec_offsets = (matched_sources['DELTA_J2000'] - matched_catalog['dec']) * 3600
        
        results['n_matched'] = len(matched_sources) if matched_sources is not None else 0
        
        if matched_sources is not None and len(matched_sources) > 0:
            # Log individual matches for debugging
            if len(matched_sources) <= 10:  # Only log details for a small number of matches
                logger.info("Match details:")
                for i in range(len(matched_sources)):
                    logger.info(f"  Match {i+1}: ΔRA={ra_offsets[i]:.2f}\", ΔDec={dec_offsets[i]:.2f}\" "
                               f"(Distance: {d2d[match_mask][i].arcsec:.2f}\")")
            
            # Calculate total pointing error (magnitude)
            total_offsets = np.sqrt(ra_offsets**2 + dec_offsets**2)
            
            # Calculate angular direction of the offset (in degrees)
            # 0° = North, 90° = East, etc.
            offset_angles = np.degrees(np.arctan2(ra_offsets, dec_offsets))
            # Convert to 0-360 range
            offset_angles = np.where(offset_angles < 0, offset_angles + 360, offset_angles)
            
            # If we have rotation information, include it in analysis
            if rotation_angle is not None:
                # Analyze if the offsets correlate with field rotation
                # A simple correlation metric
                try:
                    rot_correlation = np.corrcoef(np.array([offset_angles, np.ones_like(offset_angles) * rotation_angle]))[0, 1]
                except:
                    rot_correlation = np.nan
            else:
                rot_correlation = np.nan
            
            # Update results with detailed pointing analysis
            results.update({
                'success': True,
                'ra_offset_mean': float(np.mean(ra_offsets)),
                'dec_offset_mean': float(np.mean(dec_offsets)),
                'ra_offset_std': float(np.std(ra_offsets)) if len(matched_sources) > 1 else 0.0,
                'dec_offset_std': float(np.std(dec_offsets)) if len(matched_sources) > 1 else 0.0,
                'match_tolerance': used_tolerance,
                'pointing_analysis': {
                    'total_offset_mean': float(np.mean(total_offsets)),
                    'total_offset_std': float(np.std(total_offsets)) if len(matched_sources) > 1 else 0.0,
                    'angle_mean': float(np.mean(offset_angles)),
                    'angle_std': float(np.std(offset_angles)) if len(matched_sources) > 1 else 0.0,
                    'rotation_correlation': rot_correlation
                }
            })
            
            logger.info(f"Matched {len(matched_sources)} sources with {used_tolerance}\" tolerance")
            logger.info(f"Mean offset: RA={results['ra_offset_mean']:.2f}\", "
                       f"Dec={results['dec_offset_mean']:.2f}\"")
            
        else:
            logger.warning("No matches found within tolerance")
            
    except Exception as e:
        logger.error(f"Error during cross-matching: {str(e)}")
        
    return results

def process_directory(directory: str, target: str,
                     max_offset: float = 5.0, search_radius: float = 0.5,
                     fwhm: float = 3.0, threshold_sigma: float = 5.0,
                     sharplo: float = 0.5, sharphi: float = 2.0,
                     roundlo: float = 0.5, roundhi: float = 1.5,
                     minsize: int = 2, min_flux_ratio: float = 0.1,
                     apply_rotation: bool = True, mag_limit: float = 15.0) -> list:
    """
    Process all FITS files in a directory and its subdirectories.
    """
    all_results = []
    
    # First, query SIMBAD for the target coordinates
    logger.info(f"Querying SIMBAD for target: {target}")
    simbad_result = query_simbad(target)
    
    if simbad_result is None:
        logger.error(f"Could not resolve target '{target}' using SIMBAD")
        return all_results
        
    # Use the coordinates from SIMBAD
    ra_center = simbad_result.ra_deg
    dec_center = simbad_result.dec_deg
    
    logger.info(f"Target '{target}' resolved to: {simbad_result.identifier}")
    logger.info(f"Coordinates: RA={ra_center:.6f}°, Dec={dec_center:.6f}°")
    if simbad_result.mag_v is not None:
        logger.info(f"Visual magnitude: {simbad_result.mag_v:.2f}")
    
    try:
        fits_files = list(Path(directory).rglob('*.fits'))
        if not fits_files:
            logger.warning(f"No FITS files found in {directory}")
            return all_results
            
        logger.info(f"Found {len(fits_files)} FITS files to process")
        
        for fits_path in fits_files:
            logger.info(f"\nProcessing {fits_path.name}")
            results = verify_pointing(
                str(fits_path),
                ra_center=ra_center,
                dec_center=dec_center,
                max_offset=max_offset,
                search_radius=search_radius,
                fwhm=fwhm,
                threshold_sigma=threshold_sigma,
                sharplo=sharplo,
                sharphi=sharphi,
                roundlo=roundlo,
                roundhi=roundhi,
                minsize=minsize,
                min_flux_ratio=min_flux_ratio,
                apply_rotation=apply_rotation,
                mag_limit=mag_limit
            )
            
            if results['success']:
                logger.info("Analysis completed successfully")
                logger.info(f"RA offset: {results['ra_offset_mean']:.2f}\" ± {results['ra_offset_std']:.2f}\"")
                logger.info(f"Dec offset: {results['dec_offset_mean']:.2f}\" ± {results['dec_offset_std']:.2f}\"")
                
                # Log additional alt-az specific information
                if 'pointing_analysis' in results:
                    pa = results['pointing_analysis']
                    logger.info(f"Total offset: {pa['total_offset_mean']:.2f}\" ± {pa['total_offset_std']:.2f}\"")
                    logger.info(f"Offset angle: {pa['angle_mean']:.1f}° ± {pa['angle_std']:.1f}°")
                    
                if results['rotation_angle'] is not None:
                    logger.info(f"Derotator angle: {results['rotation_angle']:.2f}°")
                    if 'pointing_analysis' in results and results['pointing_analysis']['rotation_correlation'] is not None:
                        corr = results['pointing_analysis']['rotation_correlation']
                        logger.info(f"Rotation-offset correlation: {corr:.2f}")
                
                # Add filename and target info to results
                results['filename'] = fits_path.name
                results['target'] = target
                results['target_ra'] = ra_center
                results['target_dec'] = dec_center
                all_results.append(results)
            else:
                logger.warning("Analysis failed or produced no matches")
                
        # After processing all files, generate a summary
        if all_results:
            logger.info("\n===== ANALYSIS SUMMARY =====")
            logger.info(f"Target: {target} (RA={ra_center:.6f}°, Dec={dec_center:.6f}°)")
            logger.info(f"Successfully analyzed {len(all_results)} out of {len(fits_files)} files")
            
            # Calculate average offsets
            ra_offsets = [r['ra_offset_mean'] for r in all_results]
            dec_offsets = [r['dec_offset_mean'] for r in all_results]
            
            logger.info(f"Average RA offset: {np.mean(ra_offsets):.2f}\" ± {np.std(ra_offsets):.2f}\"")
            logger.info(f"Average Dec offset: {np.mean(dec_offsets):.2f}\" ± {np.std(dec_offsets):.2f}\"")
            
            # Report on rotation angles if available
            rot_angles = [r['rotation_angle'] for r in all_results if r['rotation_angle'] is not None]
            if rot_angles:
                logger.info(f"Derotator angles: {np.mean(rot_angles):.2f}° ± {np.std(rot_angles):.2f}°")
        
        return all_results
                
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        return all_results

def analyze_alt_az_results(results_list):
    """
    Perform specialized analysis for alt-azimuth mount data.
    """
    if not results_list:
        logger.warning("No results to analyze")
        return
    
    logger.info("\n===== ALT-AZ MOUNT ANALYSIS =====")
    
    # Extract pointing errors, rotation angles, and mount positions
    ra_offsets = []
    dec_offsets = []
    total_offsets = []
    angles = []
    rot_angles = []
    alts = []
    azs = []
    
    # Collect data from all successful results
    for result in results_list:
        if not result['success']:
            continue
            
        ra_offsets.append(result['ra_offset_mean'])
        dec_offsets.append(result['dec_offset_mean'])
        
        if 'pointing_analysis' in result:
            total_offsets.append(result['pointing_analysis']['total_offset_mean'])
            angles.append(result['pointing_analysis']['angle_mean'])
        
        if result['rotation_angle'] is not None:
            rot_angles.append(result['rotation_angle'])
            
        if result['alt'] is not None:
            alts.append(result['alt'])
            
        if result['az'] is not None:
            azs.append(result['az'])
    
    # Calculate vector statistics
    ra_mean = np.mean(ra_offsets) if ra_offsets else None
    dec_mean = np.mean(dec_offsets) if dec_offsets else None
    total_mean = np.mean(total_offsets) if total_offsets else None
    
    # Analyze correlation with rotation if we have rotation data
    if rot_angles and angles:
        try:
            # Simple correlation
            correlation = np.corrcoef(rot_angles, angles)[0, 1]
            logger.info(f"Correlation between derotator angle and offset direction: {correlation:.2f}")
        except:
            logger.info("Could not calculate correlation between derotator angle and offset direction")
        
        # Group by rotation angle ranges
        rot_groups = {
            "0-90°": [],
            "90-180°": [],
            "180-270°": [],
            "270-360°": []
        }
        
        for i, rot in enumerate(rot_angles):
            if i >= len(total_offsets):
                continue
                
            if 0 <= rot < 90:
                rot_groups["0-90°"].append(total_offsets[i])
            elif 90 <= rot < 180:
                rot_groups["90-180°"].append(total_offsets[i])
            elif 180 <= rot < 270:
                rot_groups["180-270°"].append(total_offsets[i])
            else:
                rot_groups["270-360°"].append(total_offsets[i])
        
        # Calculate average error by rotation quadrant
        logger.info("Average pointing error by derotator angle quadrant:")
        for quadrant, errors in rot_groups.items():
            if errors:
                logger.info(f"  {quadrant}: {np.mean(errors):.2f}\" (n={len(errors)})")
    
    # Output overall conclusion
    logger.info("\nCONCLUSION:")
    if total_mean:
        logger.info(f"Average pointing error: {total_mean:.2f} arcseconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify telescope pointing using Gaia catalog')
    parser.add_argument('directory', help='Directory containing FITS files')
    parser.add_argument('--target', type=str, required=True, help='Target name to resolve using SIMBAD (e.g., "M51", "NGC 5139")')
    parser.add_argument('--radius', type=float, default=0.5, help='Search radius in degrees (default: 0.5)')
    parser.add_argument('--match-tolerance', type=float, default=5.0, help='Match tolerance in arcseconds (default: 5.0)')
    parser.add_argument('--alt-az', action='store_true', help='Perform specialized analysis for alt-azimuth mounts')
    parser.add_argument('--fwhm', type=float, default=3.0, help='Full width at half maximum of stars in pixels (default: 3.0)')
    parser.add_argument('--threshold', type=float, default=5.0, help='Detection threshold in sigma above background (default: 5.0)')
    parser.add_argument('--sharplo', type=float, default=0.5, help='Lower bound on sharpness for star detection (default: 0.5)')
    parser.add_argument('--sharphi', type=float, default=2.0, help='Upper bound on sharpness for star detection (default: 2.0)')
    parser.add_argument('--roundlo', type=float, default=0.5, help='Lower bound on roundness for star detection (default: 0.5)')
    parser.add_argument('--roundhi', type=float, default=1.5, help='Upper bound on roundness for star detection (default: 1.5)')
    parser.add_argument('--minsize', type=int, default=2, help='Minimum size of stars in pixels (default: 2)')
    parser.add_argument('--min-flux-ratio', type=float, default=0.1, help='Minimum flux ratio compared to brightest star (default: 0.1)')
    parser.add_argument('--no-rotation', action='store_true', help='Disable rotation correction')
    parser.add_argument('--output', help='Output file for results (CSV format)')
    parser.add_argument('--mag-limit', type=float, default=15.0, help='Limiting magnitude for Gaia catalog query (default: 15.0)')
    
    args = parser.parse_args()
    
    results = process_directory(
        args.directory,
        target=args.target,
        max_offset=args.match_tolerance,
        search_radius=args.radius,
        fwhm=args.fwhm,
        threshold_sigma=args.threshold,
        sharplo=args.sharplo,
        sharphi=args.sharphi,
        roundlo=args.roundlo,
        roundhi=args.roundhi,
        minsize=args.minsize,
        min_flux_ratio=args.min_flux_ratio,
        apply_rotation=not args.no_rotation,
        mag_limit=args.mag_limit
    )
    
    # If we found results and alt-az flag is set, perform specialized analysis
    if results and args.alt_az:
        analyze_alt_az_results(results)
        
    # If output file is specified, save results to CSV
    if args.output and results:
        try:
            import csv
            with open(args.output, 'w', newline='') as csvfile:
                fieldnames = ['filename', 'target', 'target_ra', 'target_dec', 'success', 
                             'n_extracted', 'n_catalog', 'n_matched', 
                             'ra_offset_mean', 'dec_offset_mean', 'ra_offset_std', 'dec_offset_std',
                             'rotation_angle', 'alt', 'az']
                             
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    # Create a simplified version of the result dictionary
                    row = {field: result.get(field, None) for field in fieldnames}
                    writer.writerow(row)
                    
                logger.info(f"Results saved to {args.output}")
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")
