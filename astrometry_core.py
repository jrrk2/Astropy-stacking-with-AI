#!/usr/bin/env python3
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

import os
import datetime
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
import subprocess
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import platform
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from reproject import reproject_interp

# Configure NumPy for optimal performance on Apple Silicon
os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())

# Detect if running on Apple Silicon
IS_APPLE_SILICON = platform.processor() == 'arm' and platform.system() == 'Darwin'
if IS_APPLE_SILICON:
    print(f"Detected Apple Silicon - using {multiprocessing.cpu_count()} cores")

def convert_to_mmap(input_file, output_file=None):
    """Convert FITS file to memory-mappable format, handling BZERO/BSCALE"""
    if output_file is None:
        # Ensure we use underscores in the output filename
        input_path = Path(input_file)
        safe_stem = input_path.stem.replace('-', '_')
        output_file = str(input_path.parent / f"{safe_stem}_mmap.fits")
        
    try:
        # Read the original file without memory mapping
        with fits.open(input_file, memmap=False) as hdul:
            # Get the data and apply any BZERO/BSCALE scaling
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header.copy()
            
            # Remove scaling keywords as they're now applied
            for keyword in ['BZERO', 'BSCALE', 'BLANK']:
                if keyword in header:
                    del header[keyword]
            
        # Write in memory-mappable format
        fits.writeto(output_file, data, header, overwrite=True)
        return output_file
    except Exception as e:
        print(f"Error converting {input_file} to memory-mappable format: {str(e)}")
        return None

def ensure_mmap_format(filename, mmap_dir='mmap'):
    """Check if file needs conversion to memory-mappable format and store in mmap directory"""
    path = Path(filename)
    
    # Create mmap directory if it doesn't exist
    mmap_path = Path(mmap_dir)
    mmap_path.mkdir(exist_ok=True)
    
    # Ensure we use underscores in the mmap filename
    safe_stem = path.stem.replace('-', '_')
    mmap_file = mmap_path / f"{safe_stem}_mmap.fits"
    
    # If mmap version exists, use it
    if mmap_file.exists():
        try:
            with fits.open(str(mmap_file), memmap=True) as hdul:
                _ = hdul[0].data
            return str(mmap_file)
        except Exception:
            print(f"Existing mmap file {mmap_file} is corrupt, will reconvert")
            
    # Try to memory map original file
    try:
        with fits.open(filename, memmap=True) as hdul:
            _ = hdul[0].data
        return filename
    except Exception:
        print(f"Converting {filename} to memory-mappable format...")
        try:
            # Read the original file without memory mapping
            with fits.open(filename, memmap=False) as hdul:
                # Get the data and apply any BZERO/BSCALE scaling
                data = hdul[0].data.astype(np.float32)
                header = hdul[0].header.copy()
                
                # Remove scaling keywords as they're now applied
                for keyword in ['BZERO', 'BSCALE', 'BLANK']:
                    if keyword in header:
                        del header[keyword]
                
            # Write in memory-mappable format to mmap directory
            fits.writeto(mmap_file, data, header, overwrite=True)
            return str(mmap_file)
        except Exception as e:
            print(f"Error converting {filename} to memory-mappable format: {str(e)}")
            return None
