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
from typing import Optional, Tuple, List, Dict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

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
                   minsize: int = 2, min_flux_ratio: float = 0.1,
                   use_wcs: bool = False, plate_scale: float = None,
                   ra_center: float = None, dec_center: float = None) -> Table | None:
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
        use_wcs: Whether to try using WCS for coordinate conversion
        plate_scale: Plate scale in arcseconds per pixel (if None, calculated from header)
        ra_center: RA of field center in degrees (from SIMBAD)
        dec_center: Dec of field center in degrees (from SIMBAD)
        
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
            
            # Get image dimensions
            if data is None:
                # Try the first extension if primary HDU has no data
                if len(hdul) > 1 and hdul[1].data is not None:
                    data = hdul[1].data
                    header = hdul[1].header
                else:
                    logger.error(f"No image data found in {fits_file}")
                    return None
            
            img_height, img_width = data.shape
            img_center_x, img_center_y = img_width // 2, img_height // 2
            
            # For Stellina: calculate plate scale from pixel size and focal length if not provided
            if plate_scale is None and 'PIXSZ' in header and 'FOCAL' in header:
                pixsz = header['PIXSZ']  # pixel size in microns
                focal = header['FOCAL']  # focal length in mm
                plate_scale = (pixsz / focal) * 206.265  # arcsec/pixel
                logger.info(f"Calculated plate scale from header: {plate_scale:.4f} arcsec/pixel")
            elif plate_scale is None:
                # Use default Stellina plate scale if not provided and can't calculate
                plate_scale = 1.238
                logger.info(f"Using default Stellina plate scale: {plate_scale} arcsec/pixel")
            
            # Get WCS information if requested (but Stellina headers typically don't have WCS)
            wcs = None
            if use_wcs:
                try:
                    # Look for WCS keywords
                    has_wcs = all(keyword in header for keyword in ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2'])
                    
                    if has_wcs:
                        wcs = WCS(header)
                        # Test if WCS is valid by doing a test transformation
                        test_coords = wcs.pixel_to_world(img_center_x, img_center_y)
                        if not hasattr(test_coords, 'ra') or not hasattr(test_coords, 'dec'):
                            logger.warning("WCS transformation test failed, falling back to plate scale")
                            wcs = None
                        else:
                            logger.info(f"WCS initialized successfully. Center coordinates: RA={test_coords.ra.deg:.6f}°, Dec={test_coords.dec.deg:.6f}°")
                    else:
                        logger.info("No WCS keywords found in header, using plate scale instead")
                        wcs = None
                except Exception as e:
                    logger.warning(f"Error parsing WCS: {str(e)}. Will use plate scale instead.")
                    wcs = None
        
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
        
        # Add RA/Dec columns using WCS or plate scale
        try:
            # Extract pixel coordinates
            x_coords = sources['xcentroid']
            y_coords = sources['ycentroid']
            
            if wcs is not None:
                # Try using WCS for coordinate conversion
                try:
                    # Convert pixel to world coordinates
                    world_positions = wcs.pixel_to_world(x_coords, y_coords)
                    
                    # Ensure we have a SkyCoord object with ra/dec attributes
                    if hasattr(world_positions, 'ra') and hasattr(world_positions, 'dec'):
                        sources['ALPHA_J2000'] = world_positions.ra.deg
                        sources['DELTA_J2000'] = world_positions.dec.deg
                        logger.info(f"Successfully converted {len(sources)} sources using WCS")
                    else:
                        raise ValueError("WCS transformation did not return SkyCoord objects with ra/dec attributes")
                except Exception as e:
                    logger.warning(f"WCS coordinate transformation failed: {str(e)}. Using plate scale instead.")
                    wcs = None
            
            # Use plate scale method with SIMBAD coordinates as the field center
            if wcs is None:
                if ra_center is None or dec_center is None:
                    logger.error("No RA/Dec center coordinates provided. Cannot calculate world coordinates.")
                    return None
                
                # Calculate RA/Dec using plate scale and SIMBAD coordinates as reference
                # Convert from pixel offsets to arcseconds
                x_offset_arcsec = (x_coords - img_center_x) * plate_scale
                y_offset_arcsec = (y_coords - img_center_y) * plate_scale
                
                # Convert from arcseconds to degrees
                x_offset_deg = x_offset_arcsec / 3600.0
                y_offset_deg = y_offset_arcsec / 3600.0
                
                # RA increases to the east (negative x direction)
                # cos(dec) factor corrects for spherical distortion
                sources['ALPHA_J2000'] = ra_center - x_offset_deg / np.cos(np.radians(dec_center))
                sources['DELTA_J2000'] = dec_center + y_offset_deg
                
                logger.info(f"Successfully calculated coordinates for {len(sources)} sources using plate scale")
            
            logger.info(f"Successfully extracted {len(sources)} sources")
            return sources
            
        except Exception as e:
            logger.error(f"Error in coordinate transformation: {str(e)}")
            return None
        
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
                   max_offset: float = 3.0, search_radius: float = 0.5,
                   fwhm: float = 3.0, threshold_sigma: float = 5.0,
                   sharplo: float = 0.5, sharphi: float = 2.0,
                   roundlo: float = 0.5, roundhi: float = 1.5,
                   minsize: int = 2, min_flux_ratio: float = 0.1,
                   mag_limit: float = 15.0, use_wcs: bool = False,
                   plate_scale: float = None, fast_mode: bool = True,
                   max_sources: int = 50, gaia_catalog: Table = None) -> dict:
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
        'dec_offset_std': None
    }
    
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
            min_flux_ratio=min_flux_ratio,
            use_wcs=use_wcs,
            plate_scale=plate_scale,
            ra_center=ra_center,
            dec_center=dec_center
        )
        
        if sources is None:
            logger.warning("No sources extracted from the image")
            return results
            
        results['n_extracted'] = len(sources)
        
        # Get catalog sources - either use the provided catalog or query Gaia
        catalog_sources = gaia_catalog
        if catalog_sources is None:
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
        
        # If using fast mode, limit the number of sources for quicker matching
        if fast_mode and len(sources) > max_sources:
            # Sort sources by flux (brightest first)
            if 'flux' in sources.colnames:
                sort_idx = np.argsort(sources['flux'])[::-1]  # Descending order
                sources = sources[sort_idx[:max_sources]]
                logger.info(f"Fast mode: Limiting to {len(sources)} brightest sources for matching")
            else:
                # If no flux column, take the first max_sources
                sources = sources[:max_sources]
                logger.info(f"Fast mode: Limiting to first {len(sources)} sources for matching")
            
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
            
            # DEBUGGING: Log coordinate ranges to diagnose matching issues
            min_ra = np.min(sources['ALPHA_J2000'])
            max_ra = np.max(sources['ALPHA_J2000'])
            min_dec = np.min(sources['DELTA_J2000'])
            max_dec = np.max(sources['DELTA_J2000'])
            logger.info(f"Source RA range: {min_ra:.6f}° to {max_ra:.6f}°")
            logger.info(f"Source Dec range: {min_dec:.6f}° to {max_dec:.6f}°")
            
            cat_min_ra = np.min(catalog_sources['ra'])
            cat_max_ra = np.max(catalog_sources['ra'])
            cat_min_dec = np.min(catalog_sources['dec'])
            cat_max_dec = np.max(catalog_sources['dec'])
            logger.info(f"Catalog RA range: {cat_min_ra:.6f}° to {cat_max_ra:.6f}°")
            logger.info(f"Catalog Dec range: {cat_min_dec:.6f}° to {cat_max_dec:.6f}°")
            
            # Test if there's an approximate overlap between the ranges
            ra_overlap = (min_ra <= cat_max_ra and max_ra >= cat_min_ra)
            dec_overlap = (min_dec <= cat_max_dec and max_dec >= cat_min_dec)
            if not (ra_overlap and dec_overlap):
                logger.warning("COORDINATE MISMATCH: Source and catalog coordinate ranges don't overlap!")
                
            # DEBUGGING: Try larger matching tolerance to check if any matches are possible
            debug_tolerance = 60.0  # 1 arcminute
            debug_idx, debug_d2d, _ = image_coords.match_to_catalog_sky(catalog_coords)
            debug_matches = np.sum(debug_d2d < debug_tolerance * u.arcsec)
            if debug_matches > 0:
                closest_match = np.min(debug_d2d.arcsec)
                logger.info(f"DIAGNOSTIC: Found {debug_matches} matches with {debug_tolerance}\" tolerance. Closest match: {closest_match:.2f}\"")
                
                # Log a few of the closest matches
                closest_indices = np.argsort(debug_d2d.arcsec)[:5]
                logger.info("Closest potential matches:")
                for i in closest_indices:
                    if debug_d2d[i].arcsec < debug_tolerance:
                        src_ra = sources['ALPHA_J2000'][i]
                        src_dec = sources['DELTA_J2000'][i]
                        cat_ra = catalog_sources['ra'][debug_idx[i]]
                        cat_dec = catalog_sources['dec'][debug_idx[i]]
                        dist = debug_d2d[i].arcsec
                        logger.info(f"  Match dist: {dist:.2f}\", Source: ({src_ra:.6f}°, {src_dec:.6f}°), Catalog: ({cat_ra:.6f}°, {cat_dec:.6f}°)")
                        
                # DEBUGGING: Calculate average offset
                if debug_matches >= 3:
                    close_matches = debug_d2d < debug_tolerance * u.arcsec
                    src_matches = sources[close_matches]
                    cat_matches = catalog_sources[debug_idx[close_matches]]
                    ra_offsets = (src_matches['ALPHA_J2000'] - cat_matches['ra']) * 3600  # to arcsec
                    dec_offsets = (src_matches['DELTA_J2000'] - cat_matches['dec']) * 3600
                    avg_ra_offset = np.mean(ra_offsets)
                    avg_dec_offset = np.mean(dec_offsets)
                    logger.info(f"DIAGNOSTIC: Average offset: RA={avg_ra_offset:.2f}\", Dec={avg_dec_offset:.2f}\"")
                    logger.info(f"SUGGESTION: Try --match-tolerance {max(10.0, closest_match*1.5):.1f} with these offsets")
            else:
                logger.warning("DIAGNOSTIC: NO matches even with 60\" tolerance! Coordinate systems may be completely misaligned.")
                # Check if the fields are completely different
                center_dist = SkyCoord(ra=ra_center, dec=dec_center, unit=u.deg).separation(
                    SkyCoord(ra=np.mean(catalog_sources['ra']), dec=np.mean(catalog_sources['dec']), unit=u.deg)
                )
                logger.warning(f"DIAGNOSTIC: Distance between field center and catalog center: {center_dist.deg:.4f}° ({center_dist.arcmin:.2f}')")
                
                # Recommend a fix
                logger.warning("SUGGESTION: Try increasing --radius or check that target coordinates are correct")
        except Exception as e:
            logger.error(f"Error creating SkyCoord objects: {str(e)}")
            return results
        
        matched_sources = None
        matched_catalog = None
        ra_offsets = None
        dec_offsets = None
        used_tolerance = max_offset
        
        # In fast mode, use a single matching pass with the specified tolerance
        if fast_mode:
            logger.info(f"Fast mode: Using fixed matching tolerance of {max_offset} arcseconds")
            
            # Match sources
            idx, d2d, _ = image_coords.match_to_catalog_sky(catalog_coords)
            match_mask = d2d < max_offset * u.arcsec
            
            matched_sources = sources[match_mask]
            matched_catalog = catalog_sources[idx[match_mask]]
            
            if len(matched_sources) > 0:
                # Calculate offsets in arcseconds
                ra_offsets = (matched_sources['ALPHA_J2000'] - matched_catalog['ra']) * 3600
                dec_offsets = (matched_sources['DELTA_J2000'] - matched_catalog['dec']) * 3600
        else:
            # Match sources using progressive tolerances
            matching_tolerances = [1.0, 2.0, 3.0, 5.0, 10.0]  # in arcseconds
            
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
            
            # Update results with pointing analysis
            results.update({
                'success': True,
                'ra_offset_mean': float(np.mean(ra_offsets)),
                'dec_offset_mean': float(np.mean(dec_offsets)),
                'ra_offset_std': float(np.std(ra_offsets)) if len(matched_sources) > 1 else 0.0,
                'dec_offset_std': float(np.std(dec_offsets)) if len(matched_sources) > 1 else 0.0,
                'match_tolerance': used_tolerance,
                'total_offset_mean': float(np.mean(total_offsets)),
                'total_offset_std': float(np.std(total_offsets)) if len(matched_sources) > 1 else 0.0
            })
            
            logger.info(f"Matched {len(matched_sources)} sources with {used_tolerance}\" tolerance")
            logger.info(f"Mean offset: RA={results['ra_offset_mean']:.2f}\", Dec={results['dec_offset_mean']:.2f}\"")
            logger.info(f"Total offset: {results['total_offset_mean']:.2f}\" ± {results['total_offset_std']:.2f}\"")
            
        else:
            logger.warning("No matches found within tolerance")
            
    except Exception as e:
        logger.error(f"Error during cross-matching: {str(e)}")
        
    return results

def process_single_file(args):
    """
    Process a single FITS file - designed to be used with multiprocessing.
    
    Args:
        args: Tuple containing (fits_path, params_dict)
        
    Returns:
        Results dictionary or None if processing failed
    """
    fits_path, params = args
    
    try:
        logger.info(f"\nProcessing {fits_path.name}")
        result = verify_pointing(
            str(fits_path),
            ra_center=params['ra_center'],
            dec_center=params['dec_center'],
            max_offset=params['max_offset'],
            search_radius=params['search_radius'],
            fwhm=params['fwhm'],
            threshold_sigma=params['threshold_sigma'],
            sharplo=params['sharplo'],
            sharphi=params['sharphi'],
            roundlo=params['roundlo'],
            roundhi=params['roundhi'],
            minsize=params['minsize'],
            min_flux_ratio=params['min_flux_ratio'],
            mag_limit=params['mag_limit'],
            use_wcs=params['use_wcs'],
            plate_scale=params['plate_scale'],
            fast_mode=params['fast_mode'],
            max_sources=params['max_sources'],
            gaia_catalog=params['gaia_catalog']
        )
        
        if result['success']:
            logger.info("Analysis completed successfully")
            logger.info(f"RA offset: {result['ra_offset_mean']:.2f}\" ± {result['ra_offset_std']:.2f}\"")
            logger.info(f"Dec offset: {result['dec_offset_mean']:.2f}\" ± {result['dec_offset_std']:.2f}\"")
            logger.info(f"Total offset: {result['total_offset_mean']:.2f}\" ± {result['total_offset_std']:.2f}\"")
            
            # Add filename and target info to results
            result['filename'] = fits_path.name
            result['target'] = params['target']
            result['target_ra'] = params['ra_center']
            result['target_dec'] = params['dec_center']
            return result
        else:
            logger.warning("Analysis failed or produced no matches")
            return None
            
    except Exception as e:
        logger.error(f"Error processing file {fits_path.name}: {str(e)}")
        return None

def process_directory(directory: str, target: str,
                     max_offset: float = 3.0, search_radius: float = 0.5,
                     fwhm: float = 3.0, threshold_sigma: float = 5.0,
                     sharplo: float = 0.5, sharphi: float = 2.0,
                     roundlo: float = 0.5, roundhi: float = 1.5,
                     minsize: int = 2, min_flux_ratio: float = 0.1,
                     mag_limit: float = 15.0, use_wcs: bool = False,
                     plate_scale: float = None, fast_mode: bool = True,
                     max_sources: int = 50, num_workers: int = 8) -> list:
    """
    Process all FITS files in a directory and its subdirectories using parallel processing.
    """
    all_results = []
    start_time = time.time()
    
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
        # Find all FITS files
        fits_files = list(Path(directory).rglob('*.fits'))
        if not fits_files:
            logger.warning(f"No FITS files found in {directory}")
            return all_results
            
        num_files = len(fits_files)
        logger.info(f"Found {num_files} FITS files to process")
        
        # Query Gaia catalog once for all files
        logger.info(f"Performing initial Gaia catalog query for field center...")
        gaia_catalog = cross_match_gaia(ra_center, dec_center, search_radius=search_radius, mag_limit=mag_limit)
        if gaia_catalog is None:
            logger.warning("Failed to query Gaia catalog. Will try individual queries per file.")
        else:
            logger.info(f"Retrieved {len(gaia_catalog)} Gaia sources for matching")
        
        # Determine number of workers
        if num_workers is None or num_workers <= 0:
            # Use half the available cores by default
            num_workers = max(1, multiprocessing.cpu_count() // 2)
        logger.info(f"Using {num_workers} worker processes for parallel processing")
        
        # Create parameters dictionary
        params = {
            'target': target,
            'ra_center': ra_center,
            'dec_center': dec_center,
            'max_offset': max_offset,
            'search_radius': search_radius,
            'fwhm': fwhm,
            'threshold_sigma': threshold_sigma,
            'sharplo': sharplo,
            'sharphi': sharphi,
            'roundlo': roundlo,
            'roundhi': roundhi,
            'minsize': minsize,
            'min_flux_ratio': min_flux_ratio,
            'mag_limit': mag_limit,
            'use_wcs': use_wcs,
            'plate_scale': plate_scale,
            'fast_mode': fast_mode,
            'max_sources': max_sources,
            'gaia_catalog': gaia_catalog
        }
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            args_list = [(fits_path, params) for fits_path in fits_files]
            futures = [executor.submit(process_single_file, args) for args in args_list]
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result is not None and result.get('success', False):
                    all_results.append(result)
                
                # Report progress
                if (i+1) % 10 == 0 or (i+1) == num_files:
                    elapsed = time.time() - start_time
                    files_per_sec = (i+1) / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {i+1}/{num_files} files ({files_per_sec:.2f} files/sec)")
                
        # After processing all files, generate a summary
        if all_results:
            logger.info("\n===== ANALYSIS SUMMARY =====")
            logger.info(f"Target: {target} (RA={ra_center:.6f}°, Dec={dec_center:.6f}°)")
            logger.info(f"Successfully analyzed {len(all_results)} out of {len(fits_files)} files")
            
            # Calculate average offsets
            ra_offsets = [r['ra_offset_mean'] for r in all_results]
            dec_offsets = [r['dec_offset_mean'] for r in all_results]
            total_offsets = [r['total_offset_mean'] for r in all_results]
            
            logger.info(f"Average RA offset: {np.mean(ra_offsets):.2f}\" ± {np.std(ra_offsets):.2f}\"")
            logger.info(f"Average Dec offset: {np.mean(dec_offsets):.2f}\" ± {np.std(dec_offsets):.2f}\"")
            logger.info(f"Average total offset: {np.mean(total_offsets):.2f}\" ± {np.std(total_offsets):.2f}\"")
            
            # Performance statistics
            total_time = time.time() - start_time
            logger.info(f"\nPerformance Summary:")
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"Average time per file: {total_time/num_files:.2f} seconds")
            logger.info(f"Files processed per second: {num_files/total_time:.2f}")
        
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
    # Configure logging to show timestamps and include the process ID for parallel processing
    log_format = '%(asctime)s - [PID %(process)d] - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    
    parser = argparse.ArgumentParser(description='Verify telescope pointing using Gaia catalog')
    parser.add_argument('directory', help='Directory containing FITS files')
    parser.add_argument('--target', type=str, required=True, help='Target name to resolve using SIMBAD (e.g., "M51", "NGC 5139")')
    parser.add_argument('--radius', type=float, default=0.5, help='Search radius in degrees (default: 0.5)')
    parser.add_argument('--match-tolerance', type=float, default=3.0, help='Match tolerance in arcseconds (default: 3.0)')
    parser.add_argument('--fwhm', type=float, default=3.0, help='Full width at half maximum of stars in pixels (default: 3.0)')
    parser.add_argument('--threshold', type=float, default=5.0, help='Detection threshold in sigma above background (default: 5.0)')
    parser.add_argument('--sharplo', type=float, default=0.5, help='Lower bound on sharpness for star detection (default: 0.5)')
    parser.add_argument('--sharphi', type=float, default=2.0, help='Upper bound on sharpness for star detection (default: 2.0)')
    parser.add_argument('--roundlo', type=float, default=0.5, help='Lower bound on roundness for star detection (default: 0.5)')
    parser.add_argument('--roundhi', type=float, default=1.5, help='Upper bound on roundness for star detection (default: 1.5)')
    parser.add_argument('--minsize', type=int, default=2, help='Minimum size of stars in pixels (default: 2)')
    parser.add_argument('--min-flux-ratio', type=float, default=0.1, help='Minimum flux ratio compared to brightest star (default: 0.1)')
    parser.add_argument('--output', help='Output file for results (CSV format)')
    parser.add_argument('--mag-limit', type=float, default=15.0, help='Limiting magnitude for Gaia catalog query (default: 15.0)')
    parser.add_argument('--use-wcs', action='store_true', help='Try to use WCS for coordinate transformation (not recommended for Stellina)')
    parser.add_argument('--plate-scale', type=float, help='Override plate scale in arcsec/pixel (default: auto-calculate from FITS header or use 1.238 for Stellina)')
    parser.add_argument('--no-fast', action='store_true', help='Disable fast mode (fast mode is enabled by default)')
    parser.add_argument('--max-sources', type=int, default=50, help='Maximum number of sources to use for matching in fast mode (default: 50)')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes for parallel processing (default: 8)')
    
    args = parser.parse_args()
    
    # Record start time for performance measurement
    start_time = time.time()
    
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
        mag_limit=args.mag_limit,
        use_wcs=args.use_wcs,
        plate_scale=args.plate_scale,
        fast_mode=not args.no_fast,
        max_sources=args.max_sources,
        num_workers=args.workers
    )
    
    # If output file is specified, save results to CSV
    if args.output and results:
        try:
            import csv
            with open(args.output, 'w', newline='') as csvfile:
                fieldnames = ['filename', 'target', 'target_ra', 'target_dec', 'success', 
                             'n_extracted', 'n_catalog', 'n_matched', 
                             'ra_offset_mean', 'dec_offset_mean', 'ra_offset_std', 'dec_offset_std',
                             'total_offset_mean', 'total_offset_std']
                             
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    # Create a simplified version of the result dictionary
                    row = {field: result.get(field, None) for field in fieldnames}
                    writer.writerow(row)
                    
                logger.info(f"Results saved to {args.output}")
                
            # Report overall performance
            total_time = time.time() - start_time
            logger.info(f"\nTotal execution time: {total_time:.2f} seconds")
            if results:
                logger.info(f"Time per successful analysis: {total_time/len(results):.2f} seconds")
                
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")
