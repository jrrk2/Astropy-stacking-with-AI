import os
import sys
from pathlib import Path
import json
from astropy.io import fits
import re
from datetime import datetime
import ephem
from subprocess import run, PIPE
import threading
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np  # Added for transformation matrices

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
        print(f"RA parse error '{fmt}': {e}")
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
        print(f"Dec parse error '{fmt}': {e}")
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
        
        print(f"Query URL: {response.url}")
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Register namespace
        ns = {'v': 'http://www.ivoa.net/xml/VOTable/v1.2'}
        
        # First look for error response
        info = root.find('.//v:INFO[@name="Error"]', ns)
        if info is not None:
            print(f"SIMBAD Error: {info.get('value')}")
            return None
            
        # Find the TABLEDATA section
        tabledata = root.find('.//v:TABLEDATA', ns)
        if tabledata is None:
            print("No TABLEDATA found in response")
            print("Full response:")
            print(response.content.decode())
            return None
            
        # Get first row
        tr = tabledata.find('v:TR', ns)
        if tr is None:
            print("No data row found")
            return None
            
        # Get cells
        cells = tr.findall('v:TD', ns)
        print(f"Found {len(cells)} data cells")
        for i, cell in enumerate(cells):
            print(f"Cell {i}: {cell.text}")
        
        if len(cells) >= 3:
            identifier = cells[0].text
            ra_str = cells[1].text
            dec_str = cells[2].text
            mag_str = cells[3].text if len(cells) > 3 else None
            
            print(f"Raw coordinates: RA='{ra_str}', Dec='{dec_str}'")
            
            if ra_str and dec_str:
                ra_deg = parse_ra(ra_str)
                dec_deg = parse_dec(dec_str)
                mag_v = float(mag_str) if mag_str else None
                
                print(f"Parsed: RA={ra_deg:.4f}°, Dec={dec_deg:.4f}°")
                return SimbadResult(identifier, ra_deg, dec_deg, mag_v)
        
        print("Could not find coordinate data in response")
        return None
        
    except Exception as e:
        print(f"SIMBAD query error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Usage example:
def get_object_coordinates(name: str) -> Optional[SkyCoord]:
    """Get coordinates from SIMBAD with failover to hardcoded values"""
    result = query_simbad(name)
    if result:
        print(f"Found {result.identifier}: RA={result.ra_deg:.4f}°, Dec={result.dec_deg:.4f}°" + 
              (f", V={result.mag_v:.1f}" if result.mag_v is not None else ""))
        return SkyCoord(result.ra_deg * u.deg, result.dec_deg * u.deg)
        
    # Fallback for M51 if SIMBAD fails
    if name.upper() == 'M51':
        coords = SkyCoord('13h29m52.7s', '+47d11m43s')
        print(f"Using hardcoded coordinates for M51: RA={coords.ra.deg:.4f}°, Dec={coords.dec.deg:.4f}°")
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
        print(f"Error converting coordinates: {str(e)}")
        return None, None

def parse_observation_json(log_file_path):
    """Extract transformation matrices from the Stellina observation log file"""
    print(f"Parsing observation file: {log_file_path}")
    print(f"File exists: {os.path.exists(log_file_path)}")
    print(f"File size: {os.path.getsize(log_file_path) if os.path.exists(log_file_path) else 'N/A'} bytes")
    
    try:
        print(f"Opening observation.json...")
        with open(log_file_path, 'r') as f:
            log_content = f.read()
            print(f"Read {len(log_content)} bytes from observation.json")
        
        # Parse JSON
        print("Parsing JSON content...")
        log_data = json.loads(log_content)
        print("Successfully parsed JSON")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print("Check the observation.json file format.")
        return []
    except FileNotFoundError:
        print(f"File not found: {log_file_path}")
        return []
    except Exception as e:
        print(f"Error reading observation.json: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    transformations = []
    
    # Extract successful registration results
    if 'data' in log_data and 'corrections' in log_data['data']:
        for correction in log_data['data']['corrections']:
            if 'postCorrections' in correction:
                for post_correction in correction['postCorrections']:
                    if 'acqFile' in post_correction and 'stackingData' in post_correction['acqFile']:
                        stacking_data = post_correction['acqFile']['stackingData']
                        
                        if stacking_data.get('error') is None and 'liveRegistrationResult' in stacking_data:
                            reg_result = stacking_data['liveRegistrationResult']
                            
                            if 'matrix' in reg_result and reg_result.get('statusMessage') == 'StackingOk':
                                idx = reg_result['idx']
                                matrix = reg_result['matrix']
                                roundness = reg_result.get('roundness', 0)
                                stars_used = reg_result.get('starsUsed', 0)
                                
                                # Construct the transformation matrix
                                transform_matrix = np.array([
                                    [matrix[0], matrix[1], matrix[2]],
                                    [matrix[3], matrix[4], matrix[5]],
                                    [matrix[6], matrix[7], matrix[8]]
                                ])
                                
                                # Get file information
                                file_index = post_correction['acqFile'].get('index', -1)
                                file_path = post_correction['acqFile'].get('path', '')
                                mean_value = post_correction['acqFile'].get('mean', 0)
                                
                                # Get temperature metadata if available
                                motors = post_correction['acqFile'].get('motors', {})
                                metadata = post_correction['acqFile'].get('metadata', {})
                                
                                transformations.append({
                                    'idx': idx,
                                    'matrix': transform_matrix,
                                    'roundness': roundness,
                                    'stars_used': stars_used,
                                    'file_index': file_index,
                                    'file_path': file_path,
                                    'mean_value': mean_value,
                                    'motors': motors,
                                    'metadata': metadata
                                })
    
    # Sort by the original stacking index
    transformations.sort(key=lambda x: x['idx'])
    print(f"Extracted {len(transformations)} valid transformations")
    return transformations

def find_observation_json(fits_path):
    """
    Search for observation.json file in the same directory as the FITS file
    
    Args:
        fits_path (Path): Path to the FITS file
        
    Returns:
        Path or None: Path to observation.json if found, None otherwise
    """
    # Get directory of the FITS file
    fits_dir = Path(fits_path).parent
    
    # Look for observation.json in the same directory
    observation_json_path = fits_dir / "observation.json"
    
    if observation_json_path.exists():
        print(f"Found observation.json: {observation_json_path}")
        return observation_json_path
    else:
        print(f"No observation.json found in directory: {fits_dir}")
        return None

def annotate_fits_with_registration(fits_path, transformations, original_fits_path=None):
    """
    Annotate FITS file with registration attributes from observation.json
    
    Args:
        fits_path (Path): Path to the FITS file to annotate
        transformations (list): List of transformation dictionaries
        original_fits_path (Path, optional): Path to the original FITS file for matching
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Get the filename to match with transformations
    fits_filename = Path(fits_path).name
    original_filename = Path(original_fits_path).name if original_fits_path else None
    
    print(f"Looking for transformation match for: {fits_filename}")
    if original_filename:
        print(f"Original filename: {original_filename}")
    
    # First try to match by the original filename if provided
    matching_transform = None
    if original_filename:
        for transform in transformations:
            # Extract just the filename from the file_path
            transform_filename = os.path.basename(transform.get('file_path', ''))
            if transform_filename == original_filename:
                matching_transform = transform
                print(f"Found match by original filename: {transform_filename}")
                break
    
    # If no match found by original filename, try matching by index
    if not matching_transform:
        # Extract index from original filename if available
        index_match = None
        if original_filename:
            match = re.search(r'img-(\d{4})', original_filename)
            if match:
                index_match = match.group(1)
                print(f"Extracted index from original filename: {index_match}")
        
        # Try matching by index
        if index_match:
            for transform in transformations:
                file_index = transform.get('file_index', -1)
                if file_index != -1 and str(file_index).zfill(4) == index_match:
                    matching_transform = transform
                    print(f"Found match by index: {file_index}")
                    break
            
            # Also try matching by idx
            if not matching_transform:
                for transform in transformations:
                    idx = transform.get('idx', -1)
                    if idx != -1 and str(idx).zfill(4) == index_match:
                        matching_transform = transform
                        print(f"Found match by idx: {idx}")
                        break
    
    if not matching_transform:
        print(f"No matching transformation found for {fits_filename}")
        return False
    
    try:
        # Open FITS file in update mode
        with fits.open(fits_path, mode='update') as hdul:
            header = hdul[0].header
            
            # Add transformation matrix
            matrix = matching_transform['matrix']
            for i in range(3):
                for j in range(3):
                    key = f'TRNSFRM{i+1}{j+1}'
                    header[key] = float(matrix[i, j])
                    header.comments[key] = f'Registration matrix element [{i},{j}]'
            
            # Add other registration metadata
            header['REGINDX'] = matching_transform['idx']
            header.comments['REGINDX'] = 'Registration index'
            
            header['REGRNDS'] = matching_transform['roundness']
            header.comments['REGRNDS'] = 'Star roundness during registration'
            
            header['REGSTARS'] = matching_transform['stars_used']
            header.comments['REGSTARS'] = 'Number of stars used for registration'
            
            header['REGMEAN'] = matching_transform['mean_value']
            header.comments['REGMEAN'] = 'Mean pixel value during registration'
            
            # Add motor and metadata information if available
            motors = matching_transform.get('motors', {})
            if motors:
                for key, value in motors.items():
                    header_key = f'MOTOR_{key.upper()}'[:8]  # FITS keys limited to 8 chars
                    header[header_key] = value
                    header.comments[header_key] = f'Motor {key} information'
            
            meta = matching_transform.get('metadata', {})
            if meta:
                for key, value in meta.items():
                    header_key = f'META_{key.upper()}'[:8]  # FITS keys limited to 8 chars
                    if isinstance(value, (int, float, str, bool)):
                        header[header_key] = value
                        header.comments[header_key] = f'Metadata {key} information'
            
            # Save changes
            hdul.flush()
        
        print(f"Successfully annotated {fits_filename} with registration attributes")
        return True
    
    except Exception as e:
        print(f"Error annotating FITS with registration data: {e}")
        import traceback
        traceback.print_exc()
        return False

def annotate_fits_from_json(json_path, fits_path):
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
            
            # Calculate RA/DEC
            ra, dec = alt_az_to_radec(alt, az, date_obs)
            
            # Only add WCS if coordinate conversion succeeded
            if ra is not None and dec is not None:
                header['CTYPE1'] = 'RA---TAN'
                header['CTYPE2'] = 'DEC--TAN'
                header['CRVAL1'] = ra
                header['CRVAL2'] = dec
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
        print(f"Error annotating {fits_path}: {str(e)}")
        return False                

def get_new_filepath(fits_path, base_dir='lights'):
    """Determine new filepath based on temperature and DATE-OBS"""
    try:
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
        
        return Path(base_dir) / temp_dir / filename, None
        
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
        print(f"  Warning: Calculated position is {separation_deg:.2f}° from {target_name}")
        print(f"    Calculated: RA={calc_ra:.4f}°, Dec={calc_dec:.4f}°")
        print(f"    Expected:   RA={target_coords.ra.deg:.4f}°, Dec={target_coords.dec.deg:.4f}°")
    else:
        print(f"  Position OK - {separation_deg:.2f}° from {target_name}")
    
    return is_valid, separation_deg, target_coords

def process_directory(src_dir, base_dir='lights', target_name=None, max_separation_deg=1.0, 
                     dry_run=True, solve=False, add_registration=True, observation_json_path=None):
    """Process files with coordinate verification"""
    src_dir = Path(src_dir)
    base_dir = Path(base_dir)
    
    print(f"\nScanning directory: {src_dir}")
    
    # Get target coordinates once at the start
    target_coords = None
    if target_name:
        target_coords = get_object_coordinates(target_name)
        if target_coords is None:
            print(f"Error: Could not get coordinates for {target_name}")
            return 0, 0, 0
        print(f"\nVerifying coordinates against target: {target_name}")
        print(f"Maximum allowed separation: {max_separation_deg}°")
        print(f"Target coordinates: RA={target_coords.ra.deg:.4f}°, Dec={target_coords.dec.deg:.4f}°")

    # Find all matching pairs
    json_files = {}
    for f in src_dir.glob('img-*-stacking.json'):
        m = re.match(r'img-(\d{4})-stacking\.json', f.name)
        if m:
            json_files[m.group(1)] = f
    print(f"Found {len(json_files)} JSON files")

    pairs = []
    fits_count = 0
    for f in src_dir.glob('img-*.fits'):
        fits_count += 1
        m = re.match(r'img-(\d{4})r?\.fits', f.name)
        if m and m.group(1) in json_files:
            pairs.append({
                'index': m.group(1),
                'json': json_files[m.group(1)],
                'fits': f
            })
    print(f"Found {fits_count} FITS files")
    print(f"Matched {len(pairs)} JSON/FITS pairs")

    if dry_run:
        print("\nDRY RUN - no files will be modified")

    # Check for observation.json and load transformation data
    print("\n=== Observation.json Location Check ===")
    transformations = None
    
    # If a specific observation.json path is provided, use it directly
    if observation_json_path and add_registration:
        observation_json_path = Path(observation_json_path)
        print(f"Using specified observation.json path: {observation_json_path}")
        if observation_json_path.exists():
            print(f"Custom observation.json file exists: {observation_json_path}")
            transformations = parse_observation_json(observation_json_path)
            if transformations:
                print(f"Loaded {len(transformations)} transformations from custom observation.json")
            else:
                print("No valid transformations found in custom observation.json")
        else:
            print(f"ERROR: Specified observation.json does not exist: {observation_json_path}")
    
    # Otherwise look in default locations
    elif add_registration:
        # First check the source directory
        default_path = Path(src_dir) / "observation.json"
        print(f"Looking for observation.json at: {default_path}")
        
        if default_path.exists():
            print(f"Found observation.json: {default_path}")
            transformations = parse_observation_json(default_path)
            if transformations:
                print(f"Loaded {len(transformations)} transformations from observation.json")
            else:
                print("No valid transformations found in observation.json")
        else:
            # Try looking in the parent directory (one level up)
            parent_path = Path(src_dir).parent / "observation.json"
            print(f"Looking for observation.json in parent directory: {parent_path}")
            
            if parent_path.exists():
                print(f"Found observation.json in parent directory: {parent_path}")
                transformations = parse_observation_json(parent_path)
                if transformations:
                    print(f"Loaded {len(transformations)} transformations from parent directory observation.json")
                else:
                    print("No valid transformations found in parent directory observation.json")
            else:
                print("No observation.json found in either current or parent directory")
    
    print("=== End of Observation.json Check ===\n")

    processed = 0
    skipped = 0
    errors = 0
    
    for pair in pairs:
        print(f"\nProcessing index {pair['index']}:")
        print(f"  JSON: {pair['json']}")
        print(f"  FITS: {pair['fits']}")
        
        # Try to get coordinates first
        try:
            with fits.open(pair['fits']) as hdul:
                header = hdul[0].header
                date_obs = header.get('DATE-OBS')
                if not date_obs:
                    print("  Error: No DATE-OBS in FITS header")
                    errors += 1
                    continue
                    
            with open(pair['json']) as f:
                data = json.load(f)
                alt = data['motors']['ALT']
                az = data['motors']['AZ']
                print(f"  ALT/AZ: {alt:.2f}°, {az:.2f}°")
            
            ra, dec = alt_az_to_radec(alt, az, date_obs)
            if ra is not None and dec is not None:
                print(f"  Calculated RA/Dec: {ra:.4f}°, {dec:.4f}°")
            else:
                print("  Error: Could not calculate RA/Dec")
                errors += 1
                continue
                
            # Verify coordinates if target specified
            if target_coords:
                calc_coords = SkyCoord(ra * u.deg, dec * u.deg)
                separation = float(calc_coords.separation(target_coords).deg)
                
                if separation > max_separation_deg:
                    print(f"  Skipping - separation too large: {separation:.2f}°")
                    skipped += 1
                    continue
                print(f"  Position OK - {separation:.2f}° from {target_name}")
            
            new_path, error = get_new_filepath(pair['fits'], base_dir)
            if error:
                print(f"  Error determining new path: {error}")
                errors += 1
                continue
                
            print(f"  Target path: {new_path}")
            
            if not dry_run:
                try:
                    # Create directory if needed
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if not new_path.exists():
                        # Copy and annotate
                        fits.writeto(new_path, fits.getdata(pair['fits']), 
                                   fits.getheader(pair['fits']), overwrite=False)
                        annotate_fits_from_json(str(pair['json']), str(new_path))
                        
                        # Add registration attributes if available
                        if transformations and add_registration:
                            reg_success = annotate_fits_with_registration(new_path, transformations, original_fits_path=pair['fits'])
                            if reg_success:
                                print(f"  Added registration attributes from observation.json")
                            else:
                                print(f"  Failed to add registration attributes")
                        
                        print(f"  Created and annotated {new_path}")
                        processed += 1
                    else:
                        print(f"  Skipped - file already exists")
                        skipped += 1
                        
                except Exception as e:
                    print(f"  Error processing: {e}")
                    errors += 1
                    
        except Exception as e:
            print(f"  Error reading files: {str(e)}")
            errors += 1

    print(f"\nSummary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    
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
    parser.add_argument('--no-registration', action='store_true', 
                      help='Skip adding registration attributes from observation.json')
    parser.add_argument('--observation-json', 
                      help='Specific path to observation.json file if not in default location')
    
    args = parser.parse_args()
    if args.target:
        # Look up coordinates immediately so we can fail fast if target not found
        target_coords = get_object_coordinates(args.target)
        if target_coords:
            print(f"Target {args.target}: RA={target_coords.ra.deg:.4f}°, Dec={target_coords.dec.deg:.4f}°")
        else:
            print(f"Error: Could not find coordinates for {args.target}")
            sys.exit(1)
    
    # Handle observation.json path
    if args.observation_json:
        observation_json_path = args.observation_json
        print(f"Using custom observation.json path: {observation_json_path}")
        if not os.path.exists(observation_json_path):
            print(f"Warning: Specified observation.json path does not exist: {observation_json_path}")
            if not args.dry_run:
                print("This might cause issues when processing files. Consider checking the path.")
    else:
        observation_json_path = None
    
    processed, skipped, errors = process_directory(
        args.directory, 
        args.output, 
        args.target, 
        args.max_separation,
        args.dry_run,
        solve=False,
        add_registration=not args.no_registration,
        observation_json_path=observation_json_path
    )
