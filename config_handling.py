import configparser
from pathlib import Path
import logging

# Default configuration values
DEFAULT_CONFIG = {
    'observatory': {
        'latitude': 52.2,
        'longitude': 0.12,
        'elevation': 20,
    },
    'astrometry': {
        'index_file': '/opt/homebrew/Cellar/astrometry-net/0.97/data/index-4108.fits',
        'pixel_scale_lower': 1.1,
        'pixel_scale_upper': 1.3,
        'search_radius': 2.0,
        'downsample': 2,
    },
    'stellina': {
        'pattern_json': 'img-*-stacking.json',
        'pattern_fits': 'img-*.fits',
        'fits_regex': r'img-(\d{4})r?\.fits',
        'json_regex': r'img-(\d{4})-stacking\.json',
    },
    'processing': {
        'max_separation': 5.0,
        'default_output_dir': 'lights',
    }
}

# Load configuration
def load_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    
    # Set default values
    for section, options in DEFAULT_CONFIG.items():
        if not config.has_section(section):
            config.add_section(section)
        for key, value in options.items():
            config.set(section, key, str(value))
    
    # Override with values from config file
    if Path(config_file).exists():
        try:
            config.read(config_file)
            logging.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logging.warning(f"Error loading config file: {e}, using defaults")
    else:
        logging.info(f"Config file {config_file} not found, using defaults")
        
        # Create default config file if it doesn't exist
        try:
            with open(config_file, 'w') as f:
                config.write(f)
            logging.info(f"Created default config file at {config_file}")
        except Exception as e:
            logging.warning(f"Could not create default config file: {e}")
            
    return config

# Add this to your main script:
# CONFIG = load_config()
