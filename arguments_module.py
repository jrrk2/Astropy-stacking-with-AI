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
