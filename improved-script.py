import os
import sys
from pathlib import Path
import json
from astropy.io import fits
import re
from datetime import datetime
import ephem
import logging
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Tuple
import astrometry

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.error(f"RA parse error '{fmt}': {e}")
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
        logger.error(f"Dec parse error '{fmt}': {e}")
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
                mag_v = float(mag_str) if mag_str else None
                
                logger.info(f"Parsed: RA={ra_deg:.4f}°, Dec={dec_deg:.4f}°")
                return SimbadResult(identifier, ra_deg, dec_deg, mag_v)
        
        logger.error("Could not find coordinate data in response")
        return None
        
    except Exception as e:
        logger.error(f"SIMBAD query error: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_object_coordinates(name: str) -> Optional[SkyCoord]:
    """Get coordinates from SIMBAD with failover to hardcoded values"""
    result = query_simbad(name)
    if result:
        logger.info(f"Found {result.identifier}: RA={result.ra_deg:.4f}°, Dec={result.dec_deg:.4f}°" + 
              (f", V={result.mag_v:.1f}" if result.mag_v is not None else ""))
        return SkyCoord(result.ra_deg * u.deg, result.dec_deg * u.deg)
        
    # Fallback for M51 if SIMBAD fails
    if name.upper() == 'M51':
        coords = SkyCoord('13h29m52.7s', '+47d11m43s')
        logger.info(f"Using hardcoded coordinates for M51: RA={coords.ra.deg:.4f}°, Dec={coords.dec.deg:.4f}°")
        return coords
        
    return None

def alt_az_to_radec(alt, az, date_obs, lat=52.2, lon=0.12):
    """Convert Alt/Az to RA/Dec"""
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.elevation = 20
    
    # Convert ISO format to ephem date format
    try:
        # Parse ISO format datetime
        dt = datetime.strptime(date_obs, '%Y-%m-%dT%H:%M:%S')
        # Convert to ephem date format
        observer.date = ephem.Date(dt)
        
        ra, dec = observer.radec_of(az * ephem.pi/180, alt * ephem.pi/180)
        return float(ra) * 180/ephem.pi, float(dec) * 180/ephem.pi
    except Exception as e:
        logger.error(f"Error converting coordinates: {str(e)}")
        return None, None

def solve_with_astrometry(fits_path, ra_hint=None, dec_hint=None, radius_hint=None):
    """
    Solve plate astrometry using astrometry library instead of command line
    
    Parameters:
    fits_path: Path to FITS file
    ra_hint: RA hint in degrees (optional)
    dec_hint: Dec hint in degrees (optional)
    radius_hint: Search radius hint in degrees (optional)
    
    Returns:
    (success, wcs_header or error_message)
    """
    try:
        logger.info(f"Plate solving {fits_path}")
        
        # Ensure fits_path is a Path object
        fits_path = Path(fits_path).resolve()
        
        # Load image data
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            
        # Single index file optimized for Stellina's scale (~1.2 arcsec/pixel)
        index_file = Path("/opt/homebrew/Cellar/astrometry-net/0.97/data/index-4108.fits").resolve()
        
        # Create astrometry solver with specified index file
        with astrometry.Solver(index_files=[str(index_file)]) as solver:
            # Create position hint if coordinates are provided
            position_hint = None
            if ra_hint is not None and dec_hint is not None:
                search_radius = 2.0  # Use tighter 2 degree search radius around expected position
                position_hint = astrometry.PositionHint(
                    ra_deg=ra_hint,
                    dec_deg=dec_hint,
                    radius_deg=search_radius
                )
                logger.debug(f"Searching around RA={ra_hint:.4f}°, Dec={dec_hint:.4f}° with {search_radius}° radius")
            
            # Set up size hint for Stellina telescope (known pixel scale ~1.2 arcsec/pixel)
            size_hint = astrometry.SizeHint(
                lower_arcsec_per_pixel=1.1,
                upper_arcsec_per_pixel=1.3,
                z_scale=2.0  # Set -z2 parameter for Bayer CCD
            )
            
            # Solve the field
            solution = solver.solve(
                image_data=data,
                size_hint=size_hint,
                position_hint=position_hint,
                downsample=2  # Downsample for faster solving
            )
            
            if solution is None:
                return False, "No solution found"
            
            # Create a new WCS header
            wcs_header = solution.get_wcs_header()
            
            # Return the WCS header
            return True, wcs_header
            
    except Exception as e:
        logger.error(f"Error in plate solving: {e}")
        return False, str(e)

def annotate_fits_from_json(json_path, fits_path, use_astrometry=True):
    try:
        # Read JSON data
        with open(json_path) as f:
            data = json.load(f)
            
        # Open FITS file in update mode
        with fits.open(fits_path, mode='update') as hdul:
            header = hdul[0].header
            
            # Get timestamp for RA/DEC calculation
            date_obs = header.get('DATE-OBS')
            alt = data['motors']['ALT']
            az = data['motors']['AZ']
            
            # Calculate approximate RA/DEC for hints
            ra_hint, dec_hint = alt_az_to_radec(alt, az, date_obs)
            
            # Use astrometry library to get precise WCS if requested
            if use_astrometry and ra_hint is not None and dec_hint is not None:
                success, result = solve_with_astrometry(fits_path, ra_hint, dec_hint, 5.0)
                if success:
                    # Update header with WCS information from astrometry solution
                    for key, value in result.items():
                        header[key] = value
                    logger.info(f"Successfully added WCS from astrometry solution")
                else:
                    logger.warning(f"Astrometry solve failed: {result}. Using approximate coordinates.")
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
            header['TELID'] = (data['telescopeId'], 'Telescope identifier')
            header['BOOTCNT'] = (data['bootCount'], 'Telescope boot count')
            
            # Add motor positions
            header['ALT'] = (data['motors']['ALT'], '[deg] Altitude')
            header['AZ'] = (data['motors']['AZ'], '[deg] Azimuth')
            header['DER'] = (data['motors']['DER'], '[deg] Derotator angle')
            header['MAP'] = (data['motors']['MAP'], 'Motor map position')
            
            # Add stacking information
            header['IMGIDX'] = (data['index'], 'Image index')
            header['MEAN'] = (data['mean'], 'Mean pixel value')
            header['ACQTIME'] = (data['acqTime'], 'Acquisition time counter')
            
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
                        header['REGDX'] = (reg['correction']['x'], 'Registration X correction')
                        header['REGDY'] = (reg['correction']['y'], 'Registration Y correction')
                        header['REGDROT'] = (reg['correction']['rot'], 'Registration rotation correction')
            
            hdul.flush()
            
        return True
        
    except Exception as e:
        logger.error(f"Error annotating {fits_path}: {str(e)}")
        return False                

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

def verify_coordinates(calc_ra, calc_dec, target_name, max_separation_deg=1.0):
    """
    Verify calculated coordinates are close to target object
    Returns: (is_valid, separation_deg, target_coords)
    """
    target_coords = get_object_coordinates(target_name)
    if target_coords is None:
        return False, None, None
        
    # Create SkyCoord for calculated position
    calc_coords = SkyCoord(calc_ra * u.deg, calc_dec * u.deg)
    
    # Calculate separation
    separation = calc_coords.separation(target_coords)
    separation_deg = float(separation.deg)
    
    is_valid = separation_deg <= max_separation_deg
    
    if not is_valid:
        logger.warning(f"Calculated position is {separation_deg:.2f}° from {target_name}")
        logger.warning(f"Calculated: RA={calc_ra:.4f}°, Dec={calc_dec:.4f}°")
        logger.warning(f"Expected:   RA={target_coords.ra.deg:.4f}°, Dec={target_coords.dec.deg:.4f}°")
    else:
        logger.info(f"Position OK - {separation_deg:.2f}° from {target_name}")
    
    return is_valid, separation_deg, target_coords

def process_directory(src_dir, base_dir='lights', target_name=None, max_separation_deg=1.0, 
                     dry_run=True, use_astrometry=True):
    """Process files with coordinate verification and optional astrometry solving"""
    src_dir = Path(src_dir)
    base_dir = Path(base_dir)
    
    logger.info(f"Scanning directory: {src_dir}")
    
    # Get target coordinates once at the start
    target_coords = None
    if target_name:
        target_coords = get_object_coordinates(target_name)
        if target_coords is None:
            logger.error(f"Could not get coordinates for {target_name}")
            return 0, 0, 0
        logger.info(f"Verifying coordinates against target: {target_name}")
        logger.info(f"Maximum allowed separation: {max_separation_deg}°")
        logger.info(f"Target coordinates: RA={target_coords.ra.deg:.4f}°, Dec={target_coords.dec.deg:.4f}°")

    # Find all matching pairs
    json_files = {}
    for f in src_dir.glob('img-*-stacking.json'):
        m = re.match(r'img-(\d{4})-stacking\.json', f.name)
        if m:
            json_files[m.group(1)] = f
    logger.info(f"Found {len(json_files)} JSON files")

    pairs = []
    fits_count = 0
    for f in src_dir.glob('img-*.fits'):
        fits_count += 1
        m = re.match(r'img-(\d{4})r?\.fits', f.name)
        if m and m.group(1) in json_files:
            pairs.append({
                'index': m.group(1),
                'json': json_files[m.group(1)],
                'fits': Path(f)  # Convert to Path here
            })
    logger.info(f"Found {fits_count} FITS files")
    logger.info(f"Matched {len(pairs)} JSON/FITS pairs")

    if dry_run:
        logger.info("DRY RUN - no files will be modified")

    processed = 0
    skipped = 0
    errors = 0
    
    for pair in pairs:
        logger.info(f"Processing index {pair['index']}:")
        logger.info(f"  JSON: {pair['json']}")
        logger.info(f"  FITS: {pair['fits']}")
        
        # Try to get coordinates first
        try:
            with fits.open(pair['fits']) as hdul:
                header = hdul[0].header
                date_obs = header.get('DATE-OBS')
                if not date_obs:
                    logger.error("No DATE-OBS in FITS header")
                    errors += 1
                    continue
                    
            with open(pair['json']) as f:
                data = json.load(f)
                logger.debug("JSON content structure:")
                logger.debug(json.dumps(data, indent=2))
                
                # More flexible parsing of motor positions
                alt = None
                az = None
                
                # Try different possible JSON structures
                if 'motors' in data:
                    if isinstance(data['motors'], dict):
                        alt = data['motors'].get('ALT')
                        az = data['motors'].get('AZ')
                elif 'position' in data:
                    if isinstance(data['position'], dict):
                        alt = data['position'].get('altitude')
                        az = data['position'].get('azimuth')
                
                if alt is None or az is None:
                    logger.error(f"Could not find altitude/azimuth in JSON structure")
                    logger.error(f"Available keys: {list(data.keys())}")
                    errors += 1
                    continue
                    
                logger.info(f"  ALT/AZ: {alt:.2f}°, {az:.2f}°")
            
            ra, dec = alt_az_to_radec(alt, az, date_obs)
            if ra is not None and dec is not None:
                logger.info(f"  Calculated RA/Dec: {ra:.4f}°, {dec:.4f}°")
            else:
                logger.error("Could not calculate RA/Dec")
                errors += 1
                continue
                
                            # Verify coordinates with astrometry if target specified
            if target_coords:
                # Try to plate solve the image
                success, result = solve_with_astrometry(
                    pair['fits'], 
                    ra_hint=ra, 
                    dec_hint=dec
                )
                
                if success:
                    # Extract solved RA/Dec from WCS header
                    solved_ra = result.get('CRVAL1')
                    solved_dec = result.get('CRVAL2')
                    if solved_ra is not None and solved_dec is not None:
                        solved_coords = SkyCoord(solved_ra * u.deg, solved_dec * u.deg)
                        separation = float(solved_coords.separation(target_coords).deg)
                        
                        if separation > max_separation_deg:
                            logger.info(f"  Skipping - astrometry separation too large: {separation:.2f}°")
                            logger.debug(f"  Solved position: RA={solved_ra:.4f}°, Dec={solved_dec:.4f}°")
                            logger.debug(f"  Target position: RA={target_coords.ra.deg:.4f}°, Dec={target_coords.dec.deg:.4f}°")
                            skipped += 1
                            continue
                        logger.info(f"  Position verified by astrometry - {separation:.2f}° from {target_name}")
                    else:
                        logger.warning("  Could not extract coordinates from astrometry solution")
                        skipped += 1
                        continue
                else:
                    # Fall back to Alt/Az derived coordinates
                    logger.warning(f"  Astrometry solve failed, falling back to Alt/Az coordinates")
                    calc_coords = SkyCoord(ra * u.deg, dec * u.deg)
                    separation = float(calc_coords.separation(target_coords).deg)
                    
                    if separation > max_separation_deg:
                        logger.info(f"  Skipping - Alt/Az separation too large: {separation:.2f}°")
                        skipped += 1
                        continue
                    logger.info(f"  Position OK (from Alt/Az) - {separation:.2f}° from {target_name}")
            
            new_path, error = get_new_filepath(pair['fits'], base_dir)
            if error:
                logger.error(f"Error determining new path: {error}")
                errors += 1
                continue
                
            logger.info(f"  Target path: {new_path}")
            
            if not dry_run:
                try:
                    # Create directory if needed
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if not new_path.exists():
                        # Copy and annotate
                        fits.writeto(new_path, fits.getdata(pair['fits']), 
                                   fits.getheader(pair['fits']), overwrite=False)
                        annotate_fits_from_json(str(pair['json']), str(new_path), use_astrometry)
                        logger.info(f"  Created and annotated {new_path}")
                        processed += 1
                    else:
                        logger.info(f"  Skipped - file already exists")
                        skipped += 1
                        
                except Exception as e:
                    logger.error(f"Error processing: {e}")
                    errors += 1
                    
        except Exception as e:
            logger.error(f"Error reading files: {str(e)}")
            errors += 1

    logger.info(f"Summary:")
    logger.info(f"  Processed: {processed}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Errors: {errors}")
    
    return processed, skipped, errors

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process Stellina capture directory')
    parser.add_argument('directory', help='Directory containing capture files')
    parser.add_argument('--output', default='lights', help='Base output directory')
    parser.add_argument('--target', help='Target object name for coordinate verification')
    parser.add_argument('--max-separation', type=float, default=5.0,
                      help='Maximum allowed separation from target in degrees')
    parser.add_argument('--dry-run', action='store_true', help='Validate without modifying files')
    parser.add_argument('--lat', type=float, default=52.2, help='Observatory latitude in degrees (N positive)')
    parser.add_argument('--lon', type=float, default=0.12, help='Observatory longitude in degrees (E positive)')
    parser.add_argument('--no-astrometry', action='store_true', help='Disable astrometry solving')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.target:
        # Look up coordinates immediately so we can fail fast if target not found
        target_coords = get_object_coordinates(args.target)
        if target_coords:
            logger.info(f"Target {args.target}: RA={target_coords.ra.deg:.4f}°, Dec={target_coords.dec.deg:.4f}°")
        else:
            logger.error(f"Could not find coordinates for {args.target}")
            sys.exit(1)
    
    processed, skipped, errors = process_directory(
        args.directory, 
        args.output, 
        args.target, 
        args.max_separation,
        args.dry_run,
        not args.no_astrometry
    )
