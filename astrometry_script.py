#!/usr/bin/env python3
"""
Stellina astrophotography image processing main script

This script processes Stellina telescope captures by:
1. Finding matching JSON/FITS pairs
2. Calculating celestial coordinates from alt/az data
3. Verifying coordinates against known objects
4. Solving plate astrometry
5. Adding metadata to FITS headers
6. Organizing files into output directories

Authors: Original by Jonathan, improved version with debugging fixes
"""

import os
import sys
from pathlib import Path
import re
import logging
import traceback
import json
from typing import List, Tuple, Dict, Any, Optional
import glob

# Import modules from our project files
from command_line import setup_argparser, setup_logging, update_config_from_args
from config_handling import load_config
from json_parser import load_and_parse_json, extract_alt_az_from_json
from coordinate_verification import verify_coordinates
from astrometry_solver import solve_with_astrometry
from parallel_processing import process_directory_parallel, get_object_coordinates

# Import necessary dependencies
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import ephem
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass

def process_directory(src_dir, base_dir='lights', target_name=None, max_separation_deg=5.0, 
                     dry_run=True, use_astrometry=True, config=None):
    """Process all matching files in a directory with coordinate verification and astrometry"""
    logger = logging.getLogger(__name__)
    
    # Convert to Path objects
    src_dir = Path(src_dir)
    base_dir = Path(base_dir)
    
    # Check if source directory exists
    if not src_dir.exists():
        logger.error(f"Source directory does not exist: {src_dir}")
        return 0, 0, 1
    
    # Get target coordinates if provided
    target_coords = None
    if target_name:
        target_coords = get_object_coordinates(target_name)
        if not target_coords:
            logger.error(f"Could not find coordinates for target: {target_name}")
            # Continue without target verification
    
    # Find matching JSON and FITS files
    matches = find_matching_files(src_dir, config)
    
    # Process each matching pair
    processed = 0
    skipped = 0
    errors = 0
    
    for json_path, fits_path in matches:
        try:
            logger.info(f"Processing pair: {json_path.name} / {fits_path.name}")
            
            # Load JSON data and extract Alt/Az
            json_data, alt, az = load_and_parse_json(str(json_path))
            if json_data is None or alt is None or az is None:
                logger.error(f"Could not extract data from JSON: {json_path}")
                errors += 1
                continue
            
            # Get timestamp from FITS header
            try:
                with fits.open(fits_path) as hdul:
                    date_obs = hdul[0].header.get('DATE-OBS')
                    if not date_obs:
                        logger.error(f"Missing DATE-OBS in FITS header: {fits_path}")
                        errors += 1
                        continue
            except Exception as e:
                logger.error(f"Error reading FITS header: {e}")
                errors += 1
                continue
            
            # Convert Alt/Az to RA/Dec
            ra, dec = alt_az_to_radec(alt, az, date_obs, config=config)
            if ra is None or dec is None:
                logger.error(f"Failed to convert Alt/Az to RA/Dec")
                errors += 1
                continue
            
            # Verify coordinates against target if available
            valid_coords = True
            if target_coords:
                valid_coords, separation, _ = verify_coordinates(
                    ra, dec, target_coords, max_separation_deg, target_name
                )
                if not valid_coords:
                    logger.warning(f"Coordinates don't match target {target_name}, separation: {separation:.2f}°")
                    # Continue with processing but mark as questionable
            
            # Determine output path
            new_path, error = get_new_filepath(fits_path, base_dir)
            if error:
                logger.error(f"Error determining output path: {error}")
                errors += 1
                continue
            
            # In dry run mode, just report what would be done
            if dry_run:
                logger.info(f"DRY RUN: Would annotate {fits_path} and save to {new_path}")
                processed += 1
                continue
            
            # Make sure output directory exists
            if new_path:
                new_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Annotate FITS file with JSON metadata and astrometry
            success = annotate_fits_from_json(json_path, fits_path, use_astrometry, config)
            if not success:
                logger.error(f"Failed to annotate {fits_path}")
                errors += 1
                continue
            
            # Copy/move to output location if new_path is defined
            if new_path:
                try:
                    # Create parent directory if it doesn't exist
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file to new location
                    with open(fits_path, 'rb') as src, open(new_path, 'wb') as dst:
                        dst.write(src.read())
                    logger.info(f"Copied {fits_path} to {new_path}")
                    processed += 1
                except Exception as e:
                    logger.error(f"Error copying file: {e}")
                    errors += 1
                    continue
            else:
                logger.info(f"Annotated {fits_path} in-place")
                processed += 1
        except Exception as e:
            logger.error(f"Error processing {json_path} / {fits_path}: {e}")
            traceback.print_exc()
            errors += 1
    
    return processed, skipped, errors

def main():
    """Main entry point of the script"""
    # Parse command line arguments
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args)
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config_from_args(config, args)
    
    # Check for target coordinates
    target_coords = None
    if args.target:
        target_coords = get_object_coordinates(args.target)
        if target_coords:
            logger.info(f"Target {args.target}: RA={target_coords.ra.deg:.4f}°, Dec={target_coords.dec.deg:.4f}°")
        else:
            logger.error(f"Could not find coordinates for {args.target}")
            sys.exit(1)
    
    # Run the processing
    processed, skipped, errors = process_directory_parallel(
        args.directory, 
        args.output, 
        args.target,
        float(config['processing']['max_separation']),
        args.dry_run,
        not args.no_astrometry,
        config
    )
    
    # Print summary
    logger.info(f"Processing complete:")
    logger.info(f"  Processed: {processed}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Errors: {errors}")
    
    # Return error code if any failures
    if errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
