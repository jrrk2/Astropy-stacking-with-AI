import os
import numpy as np
import logging
import argparse
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import requests
import xml.etree.ElementTree as ET

# Astronomy libraries
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground

# Import astrometry library for solving
import astrometry

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

@dataclass
class StarSourceData:
    """Container for extracted star sources"""
    pixel_coords: List[List[float]]  # [x, y] pairs
    fluxes: Optional[List[float]] = None
    sharpness: Optional[List[float]] = None
    roundness: Optional[List[float]] = None

def extract_sources(fits_file: str, fwhm: float = 3.0, threshold_sigma: float = 5.0,
                   sharplo: float = 0.5, sharphi: float = 2.0,
                   roundlo: float = 0.5, roundhi: float = 1.5,
                   min_flux_ratio: float = 0.1) -> Optional[StarSourceData]:
    """
    Extract star sources from the image using photutils with filtering.
    
    Args:
        fits_file: Path to the FITS file
        fwhm: Full width at half maximum of stars in pixels
        threshold_sigma: Detection threshold in sigma above background
        sharplo: Lower bound on sharpness for star detection
        sharphi: Upper bound on sharpness for star detection
        roundlo: Lower bound on roundness for star detection
        roundhi: Upper bound on roundness for star detection
        min_flux_ratio: Minimum flux ratio compared to brightest star
        
    Returns:
        StarSourceData: Container with star coordinates and properties or None if extraction fails
    """
    try:
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
        
        # Estimate background
        try:
            sigma_clip = SigmaClip(sigma=3.0)
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
        
        # Source detection with filtering for hot pixels
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
        
        # Extract pixel coordinates, fluxes, and shape parameters
        pixel_coords = []
        fluxes = []
        sharpness_values = []
        roundness_values = []
        
        for i in range(len(sources)):
            x = float(sources['xcentroid'][i])
            y = float(sources['ycentroid'][i])
            pixel_coords.append([x, y])
            
            if 'flux' in sources.colnames:
                fluxes.append(float(sources['flux'][i]))
            
            if 'sharpness' in sources.colnames:
                sharpness_values.append(float(sources['sharpness'][i]))
                
            if 'roundness1' in sources.colnames:
                roundness_values.append(float(sources['roundness1'][i]))
        
        # Create and return the source data container
        return StarSourceData(
            pixel_coords=pixel_coords,
            fluxes=fluxes if fluxes else None,
            sharpness=sharpness_values if sharpness_values else None,
            roundness=roundness_values if roundness_values else None
        )
        
    except Exception as e:
        logger.error(f"Error extracting sources: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

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

def calculate_plate_scale_from_header(header):
    """
    Calculate plate scale from FITS header information
    
    Args:
        header: FITS header
        
    Returns:
        plate_scale in arcsec/pixel or None if calculation fails
    """
    try:
        if 'PIXSZ' in header and 'FOCAL' in header:
            pixsz = float(header['PIXSZ'])  # pixel size in microns
            focal = float(header['FOCAL'])  # focal length in mm
            plate_scale = (pixsz / focal) * 206.265  # arcsec/pixel
            return plate_scale
        return None
    except Exception as e:
        logger.warning(f"Error calculating plate scale from header: {e}")
        return None

def estimate_field_parameters(fits_file: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Estimate field parameters (center RA/Dec, field size) from FITS header if available.
    
    Args:
        fits_file: Path to the FITS file
        
    Returns:
        Tuple of (ra_deg, dec_deg, field_radius_deg) or None for any missing value
    """
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            
            # Try to get RA/Dec from header
            ra_deg = None
            dec_deg = None
            field_radius_deg = None
            
            # Check for standard WCS keywords
            if all(k in header for k in ['CRVAL1', 'CRVAL2']):
                ra_deg = float(header['CRVAL1'])
                dec_deg = float(header['CRVAL2'])
                logger.info(f"Found WCS center in header: RA={ra_deg:.6f}°, Dec={dec_deg:.6f}°")
            
            # Check for target coordinates
            elif all(k in header for k in ['OBJCTRA', 'OBJCTDEC']):
                # Parse string coordinates (format varies)
                try:
                    ra_str = header['OBJCTRA']
                    dec_str = header['OBJCTDEC']
                    coords = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                    ra_deg = coords.ra.deg
                    dec_deg = coords.dec.deg
                    logger.info(f"Found target coordinates in header: RA={ra_deg:.6f}°, Dec={dec_deg:.6f}°")
                except:
                    logger.warning("Could not parse OBJCTRA/OBJCTDEC from header")
            
            # Check for target name to use as fallback
            elif 'OBJECT' in header:
                logger.info(f"Found target name in header: {header['OBJECT']}")
                logger.info("Use --ra and --dec parameters to specify coordinates for this target")
            
            # Try to estimate field size
            field_radius_deg = None
            plate_scale = calculate_plate_scale_from_header(header)
            
            if plate_scale is not None:
                # Get image dimensions
                if hdul[0].data is not None:
                    height, width = hdul[0].data.shape
                elif len(hdul) > 1 and hdul[1].data is not None:
                    height, width = hdul[1].data.shape
                else:
                    height, width = 0, 0
                
                if height > 0 and width > 0:
                    # Calculate field size in degrees
                    field_width_deg = (width * plate_scale) / 3600.0
                    field_height_deg = (height * plate_scale) / 3600.0
                    field_radius_deg = max(field_width_deg, field_height_deg) / 2.0
                    logger.info(f"Estimated field radius: {field_radius_deg:.4f}°")
                    
                    # Log the plate scale
                    logger.info(f"Calculated plate scale: {plate_scale:.4f} arcsec/pixel")
            
            return ra_deg, dec_deg, field_radius_deg
                
    except Exception as e:
        logger.error(f"Error reading FITS header: {str(e)}")
        return None, None, None

def solve_field_astrometry(source_data: StarSourceData, 
                          ra_hint: Optional[float] = None, 
                          dec_hint: Optional[float] = None,
                          radius_hint: Optional[float] = None,
                          lower_scale: float = 0.5,
                          upper_scale: float = 5.0,
                          scales: Optional[set] = None) -> Optional[astrometry.Solution]:
    """
    Solve the field using astrometry.net
    
    Args:
        source_data: Extracted star source data
        ra_hint: Optional RA hint in degrees
        dec_hint: Optional Dec hint in degrees
        radius_hint: Optional search radius in degrees
        lower_scale: Lower bound on plate scale in arcsec/pixel
        upper_scale: Upper bound on plate scale in arcsec/pixel
        scales: Optional set of index scales to use
        
    Returns:
        astrometry.Solution or None if solving fails
    """
    try:
        # Use default scales if not specified
        if scales is None:
            scales = {8}
        else:
            # Make sure all provided scales are valid
            valid_scales = {4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
            scales = {s for s in scales if s in valid_scales}
            if not scales:
                logger.warning("No valid scales provided, using default scales {8}")
                scales = {8}
        
        logger.info(f"Using index scales: {scales}")
        
        # Set up the astrometry solver
        with astrometry.Solver(
            astrometry.series_4100.index_files(
                cache_directory="astrometry_cache",
                scales=scales,
            )
        ) as solver:
            # Prepare size hint
            size_hint = astrometry.SizeHint(
                lower_arcsec_per_pixel=lower_scale,
                upper_arcsec_per_pixel=upper_scale,
            )
            
            # Prepare position hint if available
            position_hint = None
            if ra_hint is not None and dec_hint is not None:
                position_hint = astrometry.PositionHint(
                    ra_deg=ra_hint,
                    dec_deg=dec_hint,
                    radius_deg=radius_hint if radius_hint is not None else 2.0,
                )
                logger.info(f"Using position hint: RA={ra_hint:.6f}°, Dec={dec_hint:.6f}°, radius={radius_hint if radius_hint is not None else 2.0}°")
            
            # Solve with the extracted star positions
            
            # Log some information about the solving attempt
            logger.info(f"Attempting to solve field with {len(source_data.pixel_coords)} stars")
            if position_hint:
                logger.info(f"Position hint: RA={position_hint.ra_deg:.6f}°, Dec={position_hint.dec_deg:.6f}°, radius={position_hint.radius_deg}°")
            logger.info(f"Size hint: {lower_scale}-{upper_scale} arcsec/pixel")
            
            solution = solver.solve(
                stars=source_data.pixel_coords,
                size_hint=size_hint,
                position_hint=position_hint,
                solution_parameters=astrometry.SolutionParameters(),
            )
            
            if solution.has_match():
                logger.info(f"Field solved successfully!")
                best_match = solution.best_match()
                logger.info(f"Solution center: RA={best_match.center_ra_deg:.6f}°, Dec={best_match.center_dec_deg:.6f}°")
                logger.info(f"Pixel scale: {best_match.scale_arcsec_per_pixel:.4f} arcsec/pixel")
                logger.info(f"Field rotation: {best_match.orientation_deg:.2f} degrees")
                return solution
            else:
                logger.warning("Field solving failed, no match found")
                return None
    
    except Exception as e:
        logger.error(f"Error solving field: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_wcs_from_solution(solution: astrometry.Solution, image_width: int, image_height: int) -> WCS:
    """
    Create a WCS object from the astrometry solution
    
    Args:
        solution: Astrometry solution
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        
    Returns:
        astropy.wcs.WCS: WCS object
    """
    # Get the best match
    match = solution.best_match()
    
    # Create a new WCS object
    wcs = WCS(naxis=2)
    
    # Calculate the reference pixel (image center)
    crpix1 = image_width / 2.0
    crpix2 = image_height / 2.0
    
    # Get the center RA/Dec
    crval1 = match.center_ra_deg
    crval2 = match.center_dec_deg
    
    # Get the pixel scale in degrees/pixel
    cdelt1 = match.scale_arcsec_per_pixel / 3600.0
    cdelt2 = match.scale_arcsec_per_pixel / 3600.0
    
    # Get the rotation angle in degrees
    rot_angle_rad = np.radians(match.orientation_deg)
    
    # Calculate the CD matrix elements
    cd1_1 = -cdelt1 * np.cos(rot_angle_rad)
    cd1_2 = cdelt2 * np.sin(rot_angle_rad)
    cd2_1 = cdelt1 * np.sin(rot_angle_rad)
    cd2_2 = cdelt2 * np.cos(rot_angle_rad)
    
    # Set the WCS parameters
    wcs.wcs.crpix = [crpix1, crpix2]
    wcs.wcs.crval = [crval1, crval2]
    wcs.wcs.cd = [[cd1_1, cd1_2], [cd2_1, cd2_2]]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    return wcs

def update_fits_with_wcs(fits_file: str, wcs: WCS, output_file: Optional[str] = None) -> str:
    """
    Update FITS file with WCS information
    
    Args:
        fits_file: Path to the input FITS file
        wcs: WCS object
        output_file: Optional output file path (if None, creates a new file)
        
    Returns:
        Path to the updated FITS file
    """
    # Create output filename if not provided
    if output_file is None:
        path = Path(fits_file)
        output_file = str(path.parent / f"{path.stem}_wcs{path.suffix}")
    
    # Read the original FITS file
    with fits.open(fits_file) as hdul:
        # Add WCS header to the primary HDU (or first HDU with data)
        if hdul[0].data is not None:
            wcs_header = wcs.to_header()
            hdul[0].header.update(wcs_header)
        elif len(hdul) > 1 and hdul[1].data is not None:
            wcs_header = wcs.to_header()
            hdul[1].header.update(wcs_header)
        
        # Write the updated FITS file
        hdul.writeto(output_file, overwrite=True)
    
    logger.info(f"Updated FITS file saved to: {output_file}")
    return output_file

def process_single_file(args):
    """
    Process a single FITS file - designed to be used with multiprocessing
    
    Args:
        args: Tuple containing (fits_path, params_dict)
        
    Returns:
        Result dictionary
    """
    fits_path, params = args
    
    result = {
        'filename': fits_path.name,
        'success': False,
        'output_file': None
    }
    
    try:
        logger.info(f"\nProcessing {fits_path.name}")
        
        # Get image dimensions and FITS header
        plate_scale = None
        with fits.open(str(fits_path)) as hdul:
            header = hdul[0].header
            
            # Calculate plate scale from header if available
            plate_scale = calculate_plate_scale_from_header(header)
            if plate_scale:
                logger.info(f"Calculated plate scale from header: {plate_scale:.4f} arcsec/pixel")
                # Update the lower and upper scale hints based on the calculated value
                margin = 0.2  # 20% margin
                params['lower_scale'] = plate_scale * (1.0 - margin)
                params['upper_scale'] = plate_scale * (1.0 + margin)
                logger.info(f"Set plate scale search range: {params['lower_scale']:.4f} to {params['upper_scale']:.4f} arcsec/pixel")
            
            if hdul[0].data is not None:
                height, width = hdul[0].data.shape
            elif len(hdul) > 1 and hdul[1].data is not None:
                height, width = hdul[1].data.shape
            else:
                logger.error(f"No image data found in {fits_path.name}")
                return result
        
        # Extract sources from the image
        source_data = extract_sources(
            str(fits_path),
            fwhm=params['fwhm'],
            threshold_sigma=params['threshold_sigma'],
            sharplo=params['sharplo'],
            sharphi=params['sharphi'],
            roundlo=params['roundlo'],
            roundhi=params['roundhi'],
            min_flux_ratio=params['min_flux_ratio']
        )
        
        if source_data is None or len(source_data.pixel_coords) < 10:
            logger.warning(f"Too few sources extracted from {fits_path.name}")
            return result
            
        logger.info(f"Extracted {len(source_data.pixel_coords)} sources from {fits_path.name}")
        
        # Get field hints from FITS header if available
        ra_hint, dec_hint, radius_hint = estimate_field_parameters(str(fits_path))
        
        # Override with command-line parameters if provided
        if params['ra_center'] is not None:
            ra_hint = params['ra_center']
        if params['dec_center'] is not None:
            dec_hint = params['dec_center']
        if params['radius'] is not None:
            radius_hint = params['radius']
            
        # Make sure we have the required hint parameters
        if ra_hint is None or dec_hint is None:
            logger.warning(f"Missing RA/Dec hints for {fits_path.name}. Using target coordinates.")
            if params['ra_center'] is not None and params['dec_center'] is not None:
                ra_hint = params['ra_center']
                dec_hint = params['dec_center']
            else:
                logger.warning(f"No target coordinates available for {fits_path.name}. Field solving may fail.")
                
        if radius_hint is None:
            logger.info(f"Using default field radius of 2.0° for {fits_path.name}")
            radius_hint = 2.0
        
        # Solve the field with astrometry.net
        solution = solve_field_astrometry(
            source_data,
            ra_hint=ra_hint,
            dec_hint=dec_hint,
            radius_hint=radius_hint,
            lower_scale=params['lower_scale'],
            upper_scale=params['upper_scale'],
            scales=params['scales']
        )
        
        if solution is None or not solution.has_match():
            logger.warning(f"Field solving failed for {fits_path.name}")
            return result
        
        # Create WCS from solution
        wcs = create_wcs_from_solution(solution, width, height)
        
        # Create output filename
        output_dir = params['output_dir']
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{fits_path.stem}_wcs{fits_path.suffix}")
        else:
            output_file = None  # Will create a file next to the original
        
        # Update FITS file with WCS
        result['output_file'] = update_fits_with_wcs(str(fits_path), wcs, output_file)
        result['success'] = True
        
        # Add solution details to result
        best_match = solution.best_match()
        result['center_ra'] = best_match.center_ra_deg
        result['center_dec'] = best_match.center_dec_deg
        result['pixel_scale'] = best_match.scale_arcsec_per_pixel
        result['orientation'] = best_match.orientation_deg
        
        logger.info(f"Successfully processed {fits_path.name}")
        return result
            
    except Exception as e:
        logger.error(f"Error processing file {fits_path.name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return result

def process_directory(directory: str, 
                      output_dir: Optional[str] = None,
                      target: Optional[str] = None,
                      ra_center: Optional[float] = None,
                      dec_center: Optional[float] = None,
                      radius: Optional[float] = None,
                      fwhm: float = 3.0,
                      threshold_sigma: float = 5.0,
                      sharplo: float = 0.5,
                      sharphi: float = 2.0,
                      roundlo: float = 0.5,
                      roundhi: float = 1.5,
                      min_flux_ratio: float = 0.1,
                      lower_scale: float = 0.5,
                      upper_scale: float = 5.0,
                      index_scales: Optional[str] = None,
                      recursive: bool = False,
                      num_workers: int = 8) -> list:
    """
    Process all FITS files in a directory using parallel processing
    
    Args:
        directory: Directory containing FITS files
        output_dir: Optional output directory for processed files
        ra_center: Optional RA hint in degrees
        dec_center: Optional Dec hint in degrees
        radius: Optional search radius in degrees
        fwhm: FWHM parameter for source extraction
        threshold_sigma: Threshold sigma for source extraction
        sharplo: Lower bound on sharpness for star detection
        sharphi: Upper bound on sharpness for star detection
        roundlo: Lower bound on roundness for star detection
        roundhi: Upper bound on roundness for star detection
        min_flux_ratio: Minimum flux ratio for star detection
        lower_scale: Lower bound on plate scale in arcsec/pixel
        upper_scale: Upper bound on plate scale in arcsec/pixel
        index_scales: Optional index scales to use (comma-separated list)
        recursive: Whether to search for FITS files recursively
        num_workers: Number of worker processes
        
    Returns:
        List of result dictionaries for processed files
    """
    all_results = []
    start_time = time.time()
    
    try:
        # If a target name was provided, query SIMBAD for coordinates
        if target and (ra_center is None or dec_center is None):
            logger.info(f"Querying SIMBAD for target: {target}")
            simbad_result = query_simbad(target)
            
            if simbad_result:
                ra_center = simbad_result.ra_deg
                dec_center = simbad_result.dec_deg
                logger.info(f"Target '{target}' resolved to: {simbad_result.identifier}")
                logger.info(f"Coordinates: RA={ra_center:.6f}°, Dec={dec_center:.6f}°")
                if simbad_result.mag_v is not None:
                    logger.info(f"Visual magnitude: {simbad_result.mag_v:.2f}")
            else:
                logger.warning(f"Could not resolve target '{target}' using SIMBAD")
        
        # Parse index scales if provided
        scales = None
        if index_scales:
            try:
                scales = set(int(s.strip()) for s in index_scales.split(','))
                # Validate scales - don't log yet as we'll do that in solve_field_astrometry
                valid_scales = {4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
                invalid_scales = scales - valid_scales
                if invalid_scales:
                    logger.warning(f"Invalid scales provided: {invalid_scales}")
                    scales = scales & valid_scales
                    if not scales:
                        logger.warning("No valid scales provided, will use default scales")
                        scales = None
            except:
                logger.warning(f"Could not parse index scales: {index_scales}")
                scales = None
        
        # Find all FITS files
        glob_pattern = '**/*.fits' if recursive else '*.fits'
        fits_files = sorted(list(Path(directory).glob(glob_pattern)))
        fits_files.extend(list(Path(directory).glob(glob_pattern.replace('.fits', '.fit'))))
        fits_files.extend(list(Path(directory).glob(glob_pattern.replace('.fits', '.FIT'))))
        fits_files.extend(list(Path(directory).glob(glob_pattern.replace('.fits', '.FITS'))))
        
        if not fits_files:
            logger.warning(f"No FITS files found in {directory}")
            return all_results
            
        num_files = len(fits_files)
        logger.info(f"Found {num_files} FITS files to process")
        
        # Determine number of workers
        if num_workers is None or num_workers <= 0:
            # Use half the available cores by default
            num_workers = max(1, multiprocessing.cpu_count() // 2)
        logger.info(f"Using {num_workers} worker processes for parallel processing")
        
        # Create parameters dictionary
        params = {
            'ra_center': ra_center,
            'dec_center': dec_center,
            'radius': radius,
            'fwhm': fwhm,
            'threshold_sigma': threshold_sigma,
            'sharplo': sharplo,
            'sharphi': sharphi,
            'roundlo': roundlo,
            'roundhi': roundhi,
            'min_flux_ratio': min_flux_ratio,
            'lower_scale': lower_scale,
            'upper_scale': upper_scale,
            'scales': scales,
            'output_dir': output_dir,
            'target': target
        }
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            args_list = [(fits_path, params) for fits_path in fits_files]
            futures = [executor.submit(process_single_file, args) for args in args_list]
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                all_results.append(result)
                
                # Report progress
                if (i+1) % 10 == 0 or (i+1) == num_files:
                    elapsed = time.time() - start_time
                    files_per_sec = (i+1) / elapsed if elapsed > 0 else 0
                    success_count = sum(1 for r in all_results if r['success'])
                    logger.info(f"Processed {i+1}/{num_files} files ({files_per_sec:.2f} files/sec, {success_count} successful)")
                
        # Generate summary
        success_count = sum(1 for r in all_results if r['success'])
        logger.info("\n===== PROCESSING SUMMARY =====")
        logger.info(f"Total files: {num_files}")
        logger.info(f"Successfully processed: {success_count}")
        logger.info(f"Failed: {num_files - success_count}")
        
        # Log some details about successful results
        if success_count > 0:
            plate_scales = [r['pixel_scale'] for r in all_results if r.get('pixel_scale')]
            if plate_scales:
                avg_scale = sum(plate_scales) / len(plate_scales)
                logger.info(f"Average plate scale: {avg_scale:.4f} arcsec/pixel")
            
            # List the output files
            logger.info("\nOutput files:")
            for result in all_results:
                if result['success'] and result['output_file']:
                    logger.info(f"  {result['output_file']}")
        
        # Performance statistics
        total_time = time.time() - start_time
        logger.info(f"\nPerformance Summary:")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Average time per file: {total_time/num_files:.2f} seconds")
        logger.info(f"Files processed per second: {num_files/total_time:.2f}")
        
        return all_results
                
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        import traceback
        traceback.print_exc()
        return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate FITS files with WCS using astrometry.net')
    parser.add_argument('directory', help='Directory containing FITS files')
    parser.add_argument('--output-dir', help='Output directory for annotated FITS files')
    parser.add_argument('--target', help='Target name to resolve using SIMBAD (e.g., "M51", "NGC 5139")')
    parser.add_argument('--ra', type=float, help='RA hint in degrees')
    parser.add_argument('--dec', type=float, help='Dec hint in degrees')
    parser.add_argument('--stellina', action='store_true', help='Optimized settings for Stellina telescope images')
    parser.add_argument('--radius', type=float, default=2.0, help='Search radius in degrees (default: 2.0)')
    parser.add_argument('--fwhm', type=float, default=3.0, help='FWHM for source extraction in pixels (default: 3.0)')
    parser.add_argument('--threshold', type=float, default=5.0, help='Detection threshold in sigma (default: 5.0)')
    parser.add_argument('--sharplo', type=float, default=0.5, help='Lower bound on sharpness (default: 0.5)')
    parser.add_argument('--sharphi', type=float, default=2.0, help='Upper bound on sharpness (default: 2.0)')
    parser.add_argument('--roundlo', type=float, default=0.5, help='Lower bound on roundness (default: 0.5)')
    parser.add_argument('--roundhi', type=float, default=1.5, help='Upper bound on roundness (default: 1.5)')
    parser.add_argument('--min-flux-ratio', type=float, default=0.1, help='Minimum flux ratio (default: 0.1)')
    parser.add_argument('--lower-scale', type=float, help='Lower bound on plate scale in arcsec/pixel (default: auto from FITS header or 0.5)')
    parser.add_argument('--upper-scale', type=float, help='Upper bound on plate scale in arcsec/pixel (default: auto from FITS header or 5.0)')
    parser.add_argument('--index-scales', help='Index scales to use (comma-separated list, e.g. "5,6,7")')
    parser.add_argument('--recursive', action='store_true', help='Search for FITS files recursively')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes (default: 8)')
    
    args = parser.parse_args()
    
    # Record start time for performance measurement
    start_time = time.time()
    
    results = process_directory(
        directory=args.directory,
        output_dir=args.output_dir,
        target=args.target,
        ra_center=args.ra,
        dec_center=args.dec,
        radius=args.radius,
        fwhm=args.fwhm,
        threshold_sigma=args.threshold,
        sharplo=args.sharplo,
        sharphi=args.sharphi,
        roundlo=args.roundlo,
        roundhi=args.roundhi,
        min_flux_ratio=args.min_flux_ratio,
        lower_scale=args.lower_scale,
        upper_scale=args.upper_scale,
        index_scales=args.index_scales,
        recursive=args.recursive,
        num_workers=args.workers
    )
    
    # Report overall performance
    total_time = time.time() - start_time
    logger.info(f"\nTotal execution time: {total_time:.2f} seconds")
