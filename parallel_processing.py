import concurrent.futures
import multiprocessing
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from astropy.coordinates import SkyCoord
from json_parser import load_and_parse_json, extract_alt_az_from_json
from datetime import datetime
from coordinate_verification import verify_coordinates
from astrometry_solver import solve_with_astrometry
import re
import logging
import traceback
import requests
import ephem
import xml.etree.ElementTree as ET
import astropy.units as u
from dataclasses import dataclass
from astropy.io import fits
import json

@dataclass
class SimbadResult:
    identifier: str
    ra_deg: float
    dec_deg: float
    mag_v: Optional[float] = None

def parse_ra(fmt: str) -> float:
    """Convert RA string to decimal degrees with multiple format support"""
    try:
        # Try HH MM SS.S format
        parts = fmt.strip().split()
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 15.0 + m / 4.0 + s / 240.0
        # Try HH MM format
        elif len(parts) == 2:
            h, m = map(float, parts)
            return h * 15.0 + m / 4.0
        # Try HHhMMmSS.Ss format
        elif 'h' in fmt and 'm' in fmt and 's' in fmt:
            h, rest = fmt.split('h')
            m, rest = rest.split('m')
            s = rest.rstrip('s')
            return float(h) * 15.0 + float(m) / 4.0 + float(s) / 240.0
        return 0.0
    except Exception as e:
        logging.error(f"RA parse error '{fmt}': {e}")
        return 0.0

def parse_dec(fmt: str) -> float:
    """Convert Dec string to decimal degrees with multiple format support"""
    try:
        fmt = fmt.strip()
        neg = fmt.startswith('-')
        parts = fmt.lstrip('-').split()
        
        # Try DD MM SS.S format
        if len(parts) == 3:
            d, m, s = map(float, parts)
            result = d + m/60.0 + s/3600.0
            return -result if neg else result
        # Try DD MM format
        elif len(parts) == 2:
            d, m = map(float, parts)
            result = d + m/60.0
            return -result if neg else result
        # Try decimal degrees
        else:
            return float(fmt)
    except Exception as e:
        logging.error(f"Dec parse error '{fmt}': {e}")
        return 0.0    

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
        
        logging.info(f"Query URL: {response.url}")
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Register namespace
        ns = {'v': 'http://www.ivoa.net/xml/VOTable/v1.2'}
        
        # First look for error response
        info = root.find('.//v:INFO[@name="Error"]', ns)
        if info is not None:
            logging.error(f"SIMBAD Error: {info.get('value')}")
            return None
            
        # Find the TABLEDATA section
        tabledata = root.find('.//v:TABLEDATA', ns)
        if tabledata is None:
            logging.error("No TABLEDATA found in response")
            logging.debug("Full response:")
            logging.debug(response.content.decode())
            return None
            
        # Get first row
        tr = tabledata.find('v:TR', ns)
        if tr is None:
            logging.error("No data row found")
            return None
            
        # Get cells
        cells = tr.findall('v:TD', ns)
        logging.debug(f"Found {len(cells)} data cells")
        for i, cell in enumerate(cells):
            logging.debug(f"Cell {i}: {cell.text}")
        
        if len(cells) >= 3:
            identifier = cells[0].text
            ra_str = cells[1].text
            dec_str = cells[2].text
            mag_str = cells[3].text if len(cells) > 3 else None
            
            logging.debug(f"Raw coordinates: RA='{ra_str}', Dec='{dec_str}'")
            
            if ra_str and dec_str:
                ra_deg = parse_ra(ra_str)
                dec_deg = parse_dec(dec_str)
                mag_v = float(mag_str) if mag_str else None
                
                logging.info(f"Parsed: RA={ra_deg:.4f}°, Dec={dec_deg:.4f}°")
                return SimbadResult(identifier, ra_deg, dec_deg, mag_v)
        
        logging.error("Could not find coordinate data in response")
        return None
        
    except Exception as e:
        logging.error(f"SIMBAD query error: {e}")
        traceback.print_exc()
        return None

def get_object_coordinates(name: str) -> Optional[SkyCoord]:
    """Get coordinates from SIMBAD with failover to hardcoded values"""
    result = query_simbad(name)
    if result:
        logging.info(f"Found {result.identifier}: RA={result.ra_deg:.4f}°, Dec={result.dec_deg:.4f}°" + 
              (f", V={result.mag_v:.1f}" if result.mag_v is not None else ""))
        return SkyCoord(result.ra_deg * u.deg, result.dec_deg * u.deg)
        
def alt_az_to_radec(alt, az, date_obs, lat=None, lon=None, config=None):
    """Convert Alt/Az to RA/Dec"""
    # Implementation directly here rather than importing to avoid parameter mismatch
    # Use configuration values if not provided
    if config:
        if lat is None:
            lat = float(config['observatory']['latitude'])
        if lon is None:
            lon = float(config['observatory']['longitude'])
        elevation = float(config['observatory']['elevation'])
    else:
        # Default values
        if lat is None:
            lat = 52.2
        if lon is None:
            lon = 0.12
        elevation = 20
    
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.elevation = elevation
    
    # Convert ISO format to ephem date format
    try:
        # Parse ISO format datetime
        dt = datetime.strptime(date_obs, '%Y-%m-%dT%H:%M:%S')
        # Convert to ephem date format
        observer.date = ephem.Date(dt)
        
        logging.debug(f"Converting Alt={alt}°, Az={az}° at {date_obs} (lat={lat}°, lon={lon}°)")
        
        # Convert to radians for ephem
        az_rad = az * ephem.pi/180
        alt_rad = alt * ephem.pi/180
        
        ra, dec = observer.radec_of(az_rad, alt_rad)
        
        # Convert back to degrees
        ra_deg = float(ra) * 180/ephem.pi
        dec_deg = float(dec) * 180/ephem.pi
        
        logging.debug(f"Conversion result: RA={ra_deg:.4f}°, Dec={dec_deg:.4f}°")
        return ra_deg, dec_deg
    except ValueError as e:
        logging.error(f"Date format error: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error converting coordinates: {str(e)}")
        return None, None

def get_new_filepath(fits_path, base_dir='lights'):
    """Determine new filepath based on temperature and DATE-OBS"""
    try:
        fits_path = Path(fits_path)
        base_dir = Path(base_dir)
        
        with fits.open(fits_path) as hdul:
            temp = hdul[0].header.get('TEMP')
            date_obs = hdul[0].header.get('DATE-OBS')
            if not temp or not date_obs:
                return None, "Missing TEMP or DATE-OBS in FITS header"
        
        # Convert temperature to directory name
        temp_k = int(round(temp + 273.15))
        temp_dir = f"temp_{temp_k}"
        
        # Convert DATE-OBS to filename format
        dt = datetime.strptime(date_obs, '%Y-%m-%dT%H:%M:%S')
        filename = f"light_{dt.strftime('%Y%m%d_%H%M%S')}.fits"
        
        return base_dir / temp_dir / filename, None
        
    except Exception as e:
        return None, str(e)

def annotate_fits_from_json(json_path, fits_path, use_astrometry=True, config=None):
    """Add metadata from JSON file to FITS header and solve astrometry if requested"""
    try:
        # Read JSON data
        with open(json_path) as f:
            data = json.load(f)
            
        # Open FITS file in update mode
        with fits.open(fits_path, mode='update') as hdul:
            header = hdul[0].header
            
            # Get timestamp for RA/DEC calculation
            date_obs = header.get('DATE-OBS')
            
            # Extract altitude and azimuth from JSON
            alt, az = extract_alt_az_from_json(data)
            if alt is None or az is None:
                logging.error(f"Could not extract ALT/AZ from JSON: {json_path}")
                return False
            
            # Calculate approximate RA/DEC for hints
            ra_hint, dec_hint = alt_az_to_radec(alt, az, date_obs, config=config)
            
            # Use astrometry library to get precise WCS if requested
            if use_astrometry and ra_hint is not None and dec_hint is not None:
                success, result = solve_with_astrometry(fits_path, ra_hint, dec_hint, config=config)
                if success:
                    # Update header with WCS information from astrometry solution
                    for key, value in result.items():
                        header[key] = value
                    logging.info(f"Successfully added WCS from astrometry solution")
                else:
                    logging.warning(f"Astrometry solve failed: {result}. Using approximate coordinates.")
                    # Fall back to approximate coordinates
                    header['CTYPE1'] = 'RA---TAN'
                    header['CTYPE2'] = 'DEC--TAN'
                    header['CRVAL1'] = ra_hint
                    header['CRVAL2'] = dec_hint
                    header['CRPIX1'] = 1536  # Center of 3072
                    header['CRPIX2'] = 1040  # Center of 2080
                    header['CDELT1'] = -0.000305556
                    header['CDELT2'] = 0.000305556
            else:
                # Only add WCS if coordinate conversion succeeded
                if ra_hint is not None and dec_hint is not None:
                    header['CTYPE1'] = 'RA---TAN'
                    header['CTYPE2'] = 'DEC--TAN'
                    header['CRVAL1'] = ra_hint
                    header['CRVAL2'] = dec_hint
                    header['CRPIX1'] = 1536  # Center of 3072
                    header['CRPIX2'] = 1040  # Center of 2080
                    header['CDELT1'] = -0.000305556
                    header['CDELT2'] = 0.000305556

            # Add telescope information
            header['TELESCOP'] = ('Stellina', 'Telescope model')
            header['TELID'] = (data.get('telescopeId', ''), 'Telescope identifier')
            header['BOOTCNT'] = (data.get('bootCount', 0), 'Telescope boot count')
            
            # Add motor positions
            motors = data.get('motors', {})
            if isinstance(motors, dict):
                header['ALT'] = (motors.get('ALT', alt), '[deg] Altitude')
                header['AZ'] = (motors.get('AZ', az), '[deg] Azimuth')
                header['DER'] = (motors.get('DER', 0), '[deg] Derotator angle')
                header['MAP'] = (motors.get('MAP', 0), 'Motor map position')
            
            # Add stacking information
            header['IMGIDX'] = (data.get('index', 0), 'Image index')
            header['MEAN'] = (data.get('mean', 0), 'Mean pixel value')
            header['ACQTIME'] = (data.get('acqTime', 0), 'Acquisition time counter')
            
            # Add stacking data if available
            if 'stackingData' in data:
                sd = data['stackingData']
                header['STACKCNT'] = (sd.get('stackingCount', 0), 'Number of stacked frames')
                header['STACKERR'] = (sd.get('stackingErrorCount', 0), 'Number of stacking errors')
                
                # Add live registration data if available
                if 'liveRegistrationResult' in sd:
                    reg = sd['liveRegistrationResult']
                    header['REGSTARS'] = (reg.get('starsTotal', 0), 'Total stars detected')
                    header['REGUSED'] = (reg.get('starsUsed', 0), 'Stars used in registration')
                    header['REGREJ'] = (reg.get('starsRejected', 0), 'Stars rejected in registration')
                    header['ROUND'] = (reg.get('roundness', 0), 'Star roundness metric')
                    header['FOCUS'] = (reg.get('focus', 0), 'Focus metric')
                    header['FOCUSQ'] = (reg.get('focusQuality', 0), 'Focus quality metric')
                    
                    if 'correction' in reg:
                        header['REGDX'] = (reg['correction'].get('x', 0), 'Registration X correction')
                        header['REGDY'] = (reg['correction'].get('y', 0), 'Registration Y correction')
                        header['REGDROT'] = (reg['correction'].get('rot', 0), 'Registration rotation correction')
            
            hdul.flush()
            
        return True
        
    except Exception as e:
        logging.error(f"Error annotating {fits_path}: {str(e)}")
        traceback.print_exc()
        return False
    
def process_file_pair(args):
    """
    Process a single JSON/FITS file pair
    
    This function is designed to be called by the process pool executor
    and contains all the logic to process a single file pair completely.
    
    Parameters:
    args: Tuple containing (json_path, fits_path, target_coords, config, max_separation_deg)
    
    Returns:
    Tuple of (success, output_path, message)
    """
    json_path, fits_path, target_coords, config, max_separation_deg = args
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Processing pair: {json_path.name} / {fits_path.name}")
        
        # Load JSON data and extract Alt/Az
        json_data, alt, az = load_and_parse_json(str(json_path))
        if json_data is None or alt is None or az is None:
            logger.error(f"Could not extract data from JSON: {json_path}")
            return False, None, "Failed to extract Alt/Az data"
        
        # Get timestamp from FITS header
        try:
            with fits.open(fits_path) as hdul:
                date_obs = hdul[0].header.get('DATE-OBS')
                if not date_obs:
                    logger.error(f"Missing DATE-OBS in FITS header: {fits_path}")
                    return False, None, "Missing DATE-OBS in FITS header"
        except Exception as e:
            logger.error(f"Error reading FITS header: {e}")
            return False, None, f"Error reading FITS header: {e}"
        
        # Convert Alt/Az to RA/Dec
        ra, dec = alt_az_to_radec(alt, az, date_obs, config=config)
        if ra is None or dec is None:
            logger.error(f"Failed to convert Alt/Az to RA/Dec")
            return False, None, "Failed to convert Alt/Az to RA/Dec"
        
        # Verify coordinates against target if available
        valid_coords = True
        if target_coords:
            valid_coords, separation, _ = verify_coordinates(
                ra, dec, target_coords, max_separation_deg, target_name=None
            )
            if not valid_coords:
                logger.warning(f"Coordinates don't match target, separation: {separation:.2f}°")
                return False, None, f"Coordinates don't match target (separation: {separation:.2f}°)"
            else:
                logger.info(f"Position OK - {separation:.2f}° from target")
        
        # Determine output path
        output_dir = config['processing'].get('default_output_dir', 'lights')
        new_path, error = get_new_filepath(fits_path, output_dir)
        if error:
            logger.error(f"Error determining output path: {error}")
            return False, None, f"Error determining output path: {error}"
        
        # Make sure output directory exists
        if new_path:
            new_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Annotate FITS file with JSON metadata and astrometry
        use_astrometry = not config['astrometry'].get('no_astrometry', 'False').lower() == 'true'
        success = annotate_fits_from_json(json_path, fits_path, use_astrometry, config)
        if not success:
            logger.error(f"Failed to annotate {fits_path}")
            return False, None, "Failed to annotate FITS file"
        
        # Copy/move to output location if new_path is defined
        if new_path:
            try:
                # Create parent directory if it doesn't exist
                new_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file to new location
                with open(fits_path, 'rb') as src, open(new_path, 'wb') as dst:
                    dst.write(src.read())
                logger.info(f"Copied {fits_path} to {new_path}")
                return True, new_path, "Success"
            except Exception as e:
                logger.error(f"Error copying file: {e}")
                return False, None, f"Error copying file: {e}"
        else:
            logger.info(f"Annotated {fits_path} in-place")
            return True, fits_path, "Success (in-place)"
            
    except Exception as e:
        logger.error(f"Error processing {json_path} / {fits_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, f"Error: {str(e)}"

def find_matching_files(src_dir, config):
    """Find matching JSON and FITS files"""
    src_path = Path(src_dir)
    logger = logging.getLogger(__name__)
    
    # Get file patterns from config
    json_pattern = config['stellina']['pattern_json']
    fits_pattern = config['stellina']['pattern_fits']
    json_regex = config['stellina']['json_regex']
    fits_regex = config['stellina']['fits_regex']
    
    # Compile regex patterns
    json_re = re.compile(json_regex)
    fits_re = re.compile(fits_regex)
    
    # Find all JSON and FITS files
    json_files = list(src_path.glob(json_pattern))
    fits_files = list(src_path.glob(fits_pattern))
    
    logger.info(f"Found {len(json_files)} JSON files and {len(fits_files)} FITS files")
    
    # Create mapping of indices to files
    json_map = {}
    for json_file in json_files:
        match = json_re.search(str(json_file.name))
        if match:
            index = match.group(1)
            json_map[index] = json_file
    
    fits_map = {}
    for fits_file in fits_files:
        match = fits_re.search(str(fits_file.name))
        if match:
            index = match.group(1)
            fits_map[index] = fits_file
    
    # Find matching pairs
    matches = []
    for index in json_map:
        if index in fits_map:
            matches.append((json_map[index], fits_map[index]))
    
    logger.info(f"Found {len(matches)} matching JSON/FITS pairs")
    return matches

def process_directory_parallel(src_dir, base_dir='lights', target_name=None, max_separation_deg=5.0, 
                              dry_run=True, use_astrometry=True, config=None, max_workers=8):
    """Process all matching files in a directory in parallel with coordinate verification and astrometry"""
    logger = logging.getLogger(__name__)
    
    # Convert to Path objects
    src_dir = Path(src_dir)
    base_dir = Path(base_dir)
    
    # Check if source directory exists
    if not src_dir.exists():
        logger.error(f"Source directory does not exist: {src_dir}")
        return 0, 0, 1
    
    # Update config with parameters
    if config is None:
        # Create a minimal config
        config = {
            'processing': {'default_output_dir': str(base_dir)},
            'astrometry': {'no_astrometry': str(not use_astrometry)}
        }
    else:
        # Update existing config
        config['processing']['default_output_dir'] = str(base_dir)
        config['astrometry']['no_astrometry'] = str(not use_astrometry)
    
    # Get target coordinates if provided
    target_coords = None
    if target_name:
        target_coords = get_object_coordinates(target_name)
        if not target_coords:
            logger.error(f"Could not find coordinates for target: {target_name}")
            # Continue without target verification
    
    # Find matching JSON and FITS files
    matches = find_matching_files(src_dir, config)
    
    if dry_run:
        logger.info(f"DRY RUN MODE: Found {len(matches)} file pairs to process")
        # Just report what would be done
        for json_path, fits_path in matches:
            logger.info(f"Would process: {json_path.name} / {fits_path.name}")
        return len(matches), 0, 0
    
    # Prepare arguments for parallel processing
    process_args = [(json_path, fits_path, target_coords, config, max_separation_deg) 
                   for json_path, fits_path in matches]
    
    # Process files in parallel
    processed = 0
    skipped = 0
    errors = 0
    
    # Determine number of workers (default to 8 or CPU count, whichever is smaller)
    num_workers = min(max_workers, multiprocessing.cpu_count())
    logger.info(f"Processing {len(matches)} file pairs using {num_workers} worker processes")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file_pair, args): args[0].name for args in process_args}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                success, output_path, message = future.result()
                if success:
                    processed += 1
                    logger.info(f"Successfully processed {file_name} -> {output_path}")
                else:
                    errors += 1
                    logger.error(f"Failed to process {file_name}: {message}")
            except Exception as e:
                errors += 1
                logger.error(f"Exception processing {file_name}: {str(e)}")
    
    logger.info(f"Parallel processing complete: {processed} processed, {skipped} skipped, {errors} errors")
    return processed, skipped, errors

# Replace the original process_directory function in main.py with this call to the parallel version
def process_directory(src_dir, base_dir='lights', target_name=None, max_separation_deg=5.0, 
                     dry_run=True, use_astrometry=True, config=None):
    """Wrapper to maintain compatibility with original function signature"""
    # Get max_workers from config or default to 8
    max_workers = 8
    if config and 'processing' in config:
        max_workers = int(config['processing'].get('max_workers', '8'))
    
    return process_directory_parallel(
        src_dir, base_dir, target_name, max_separation_deg, 
        dry_run, use_astrometry, config, max_workers
    )
