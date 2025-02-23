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

# Import our modules
from dark_manager import DarkFrameManager
from mmap_stacker import MmapStacker
from astrometry_core import ensure_mmap_format, IS_APPLE_SILICON

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stack frames using memory-mapped processing')
    parser.add_argument('pattern', help='Input file pattern (e.g., "img-*.fits")')
    parser.add_argument('--dark-dir', required=True, help='Directory containing master dark frames')
    parser.add_argument('--mmap-dir', default='mmap', help='Directory for memory-mapped files')
    parser.add_argument('--max-dark-frames', type=int, default=256, 
                       help='Maximum number of dark frames to stack per temperature')
    
    args = parser.parse_args()
    
    import glob
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files found matching pattern: {args.pattern}")
        exit(1)
    
    print(f"Found {len(files)} files to process")
    
    # Create mmap directory if it doesn't exist
    mmap_dir = Path(args.mmap_dir)
    mmap_dir.mkdir(exist_ok=True)
    
    # Ensure all files are in memory-mappable format
    mmap_files = []
    for file in tqdm(files, desc="Checking/converting files"):
        mmap_file = ensure_mmap_format(file, args.mmap_dir)
        if mmap_file:
            mmap_files.append(mmap_file)
        else:
            print(f"Warning: Skipping {file} due to conversion failure")
            
    if not mmap_files:
        print("No valid files to process after conversion check")
        exit(1)
        
    print(f"Processing {len(mmap_files)} valid files")
    print(f"Memory-mapped files stored in: {args.mmap_dir}/")
    
    stacker = MmapStacker(args.dark_dir)
    stacker.process_files(mmap_files)

if __name__ == "__main__":
    main()
