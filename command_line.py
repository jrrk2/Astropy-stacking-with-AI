import argparse
import logging
import sys
from pathlib import Path

def setup_argparser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(description='Process Stellina astrophotography images')
    
    # Main arguments
    parser.add_argument('directory', help='Directory containing capture files')
    parser.add_argument('--output', default='lights', help='Base output directory')
    
    # Target and coordinate verification
    parser.add_argument('--target', help='Target object name for coordinate verification')
    parser.add_argument('--max-separation', type=float, default=5.0,
                      help='Maximum allowed separation from target in degrees')
    
    # Observatory location
    parser.add_argument('--lat', type=float, help='Observatory latitude in degrees (N positive)')
    parser.add_argument('--lon', type=float, help='Observatory longitude in degrees (E positive)')
    parser.add_argument('--elevation', type=float, help='Observatory elevation in meters')
    
    # Astrometry options
    parser.add_argument('--no-astrometry', action='store_true', help='Disable astrometry solving')
    parser.add_argument('--astrometry-index', help='Path to astrometry index file')
    parser.add_argument('--pixel-scale-lower', type=float, help='Lower bound for pixel scale in arcsec/pixel')
    parser.add_argument('--pixel-scale-upper', type=float, help='Upper bound for pixel scale in arcsec/pixel')
    
    # File patterns
    parser.add_argument('--json-pattern', help='Glob pattern for JSON files')
    parser.add_argument('--fits-pattern', help='Glob pattern for FITS files')
    
    # Processing options
    parser.add_argument('--dry-run', action='store_true', help='Validate without modifying files')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    
    # Configuration file
    parser.add_argument('--config', default='config.ini', help='Configuration file path')
    
    # Logging options
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--log-file', help='Log file path')
    
    return parser

def setup_logging(args):
    """Set up logging based on command line arguments"""
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    # Create handlers
    handlers = [logging.StreamHandler()]
    
    # Add file handler if specified
    if args.log_file:
        try:
            handlers.append(logging.FileHandler(args.log_file, mode='w'))
        except Exception as e:
            print(f"Warning: Could not set up log file: {e}")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    return logger

def update_config_from_args(config, args):
    """Update configuration with values from command line arguments"""
    # Update observatory section
    if args.lat is not None:
        config['observatory']['latitude'] = str(args.lat)
    if args.lon is not None:
        config['observatory']['longitude'] = str(args.lon)
    if args.elevation is not None:
        config['observatory']['elevation'] = str(args.elevation)
    
    # Update astrometry section
    if args.astrometry_index:
        config['astrometry']['index_file'] = args.astrometry_index
    if args.pixel_scale_lower is not None:
        config['astrometry']['pixel_scale_lower'] = str(args.pixel_scale_lower)
    if args.pixel_scale_upper is not None:
        config['astrometry']['pixel_scale_upper'] = str(args.pixel_scale_upper)
    
    # Update stellina section
    if args.json_pattern:
        config['stellina']['pattern_json'] = args.json_pattern
    if args.fits_pattern:
        config['stellina']['pattern_fits'] = args.fits_pattern
    
    # Update processing section
    if args.max_separation:
        config['processing']['max_separation'] = str(args.max_separation)
    if args.output:
        config['processing']['default_output_dir'] = args.output
    
    return config

# Example usage in main script:
"""
if __name__ == "__main__":
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
    if args.target:
        target_coords = get_object_coordinates(args.target)
        if target_coords:
            logger.info(f"Target {args.target}: RA={target_coords.ra.deg:.4f}°, Dec={target_coords.dec.deg:.4f}°")
        else:
            logger.error(f"Could not find coordinates for {args.target}")
            sys.exit(1)
    
    # Run the processing
    processed, skipped, errors = process_directory(
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
"""
