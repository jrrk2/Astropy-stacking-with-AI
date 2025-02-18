#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from astropy.io import fits
from tqdm import tqdm

def convert_to_mmappable(filename):
    """Convert FITS file to memory-mappable format"""
    base = Path(filename).stem
    output = f"{base}_mmap.fits"
    
    with fits.open(filename) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        
        # Remove BZERO/BSCALE which prevent memory mapping
        if 'BZERO' in header:
            # Apply BZERO before removing
            data = data.astype(np.float32) + header['BZERO']
            del header['BZERO']
        if 'BSCALE' in header:
            if header['BSCALE'] != 1:
                data = data * header['BSCALE']
            del header['BSCALE']
            
        # Write mmap-compatible FITS
        fits.writeto(output, data.astype(np.float32), header, overwrite=True)
        
    return output

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert FITS files to mmap-compatible format')
    parser.add_argument('pattern', help='Input file pattern (e.g., "img-*.fits")')
    
    args = parser.parse_args()
    
    import glob
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files found matching pattern: {args.pattern}")
        exit(1)
        
    print(f"Converting {len(files)} files to mmap format...")
    converted = []
    for f in tqdm(files):
        converted.append(convert_to_mmappable(f))
    
    print("\nConverted files:")
    for f in converted:
        print(f"  {f}")
        
    return converted

if __name__ == "__main__":
    main()
