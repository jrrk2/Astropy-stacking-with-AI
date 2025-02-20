from astropy.io import fits
import numpy as np
import cv2
import argparse
from scipy import interpolate
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io.votable import parse_single_table
from astroquery.vizier import Vizier
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Processing with Background Gradient Elimination and Photometric Calibration')
    parser.add_argument('--method', type=str, default='none', 
                        choices=['none', 'median', 'polynomial', 'local', 'wavelet'],
                        help='Background gradient elimination method')
    parser.add_argument('--catalog', type=str, default='apass',
                        choices=['apass', 'sdss', 'panstarrs'],
                        help='Star catalog to use for photometric calibration')
    parser.add_argument('--mag-limit', type=float, default=16.0,
                        help='Magnitude limit for reference stars')
    return parser.parse_args()

def get_catalog_stars(wcs, catalog='apass', mag_limit=16.0):
    """
    Query online star catalog for reference stars in the image field
    """
    # Get image corners
    naxis1, naxis2 = wcs.pixel_shape
    corners = np.array([[0, 0], [0, naxis2], [naxis1, 0], [naxis1, naxis2]])
    ra_dec_corners = wcs.all_pix2world(corners, 0)
    
    # Calculate field center and radius
    center_ra = np.mean(ra_dec_corners[:, 0])
    center_dec = np.mean(ra_dec_corners[:, 1])
    radius = np.max(np.sqrt(
        (ra_dec_corners[:, 0] - center_ra)**2 +
        (ra_dec_corners[:, 1] - center_dec)**2
    )) * u.deg
    
    print(f"\nField center: RA = {center_ra:.4f}, Dec = {center_dec:.4f}")
    print(f"Search radius: {radius.value:.4f} degrees")
    
    center = SkyCoord(center_ra, center_dec, unit=(u.deg, u.deg))
    
    # Configure Vizier
    vizier = Vizier(columns=['*', '+_r'])  # Get all columns plus distance
    vizier.ROW_LIMIT = -1  # No row limit
    
    # Define catalog parameters
    catalog_params = {
        'apass': {
            'catalog': "II/336/apass9",
            'magnitude_column': 'r_mag',
            'mag_limit_column': 'r_mag',
            'ra_column': 'RAJ2000',
            'dec_column': 'DEJ2000'
        },
        'sdss': {
            'catalog': "V/147/sdss12",
            'magnitude_column': 'rmag',
            'mag_limit_column': 'rmag',
            'ra_column': 'RA_ICRS',
            'dec_column': 'DE_ICRS'
        },
        'panstarrs': {
            'catalog': "II/349/ps1",
            'magnitude_column': 'rmag',
            'mag_limit_column': 'rmag',
            'ra_column': 'RAJ2000',
            'dec_column': 'DEJ2000'
        }
    }