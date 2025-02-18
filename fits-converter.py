#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from astropy.io import fits

def convert_fits_to_mmap_compatible(input_file, output_file=None):
    """
    Convert FITS file to memory-map compatible format
    
    Args:
        input_file (str): Path to input FITS file
        output_file (str, optional): Path to output FITS file. 
                                     If None, generates filename based on input
    
    Returns:
        str: Path to converted FITS file
    """
    # If no output file specified, create one
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_mmap{ext}"
    
    # Open the input FITS file
    with fits.open(input_file, memmap=False) as hdul:
        # Extract primary data
        primary_data = hdul[0].data
        primary_header = hdul[0].header
        
        # Remove BZERO/BSCALE to make memory-mappable
        if 'BZERO' in primary_header:
            del primary_header['BZERO']
        if 'BSCALE' in primary_header:
            del primary_header['BSCALE']
        
        # Convert data to standard integer type if needed
        if primary_data.dtype == np.dtype('int16'):
            # If already int16, just create a copy
            data_to_save = primary_data.copy()
        else:
            # Convert to int16, scaling appropriately
            data_min = primary_data.min()
            data_max = primary_data.max()
            
            # Scale to full int16 range
            scaled_data = ((primary_data - data_min) / 
                           (data_max - data_min) * 65535).astype(np.int16)
            
            data_to_save = scaled_data
        
        # Create new FITS file with mmap-compatible data
        primary_hdu = fits.PrimaryHDU(data=data_to_save, header=primary_header)
        primary_hdu.writeto(output_file, overwrite=True)
    
    return output_file

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Convert FITS files to memory-map compatible format')
    parser.add_argument('files', nargs='+', help='Input FITS files to convert')
    parser.add_argument('-o', '--output-dir', 
                        help='Output directory for converted files (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Process each file
    converted_files = []
    for input_file in args.files:
        try:
            # Determine output path
            if args.output_dir:
                base_filename = os.path.basename(input_file)
                output_file = os.path.join(args.output_dir, 
                                           f"mmap_{base_filename}")
            else:
                output_file = None
            
            # Convert file
            converted_file = convert_fits_to_mmap_compatible(
                input_file, 
                output_file
            )
            
            converted_files.append(converted_file)
            
            if args.verbose:
                print(f"Converted {input_file} to {converted_file}")
        
        except Exception as e:
            print(f"Error converting {input_file}: {e}")
    
    # Print summary
    print(f"\nConverted {len(converted_files)} files.")
    if args.verbose:
        for f in converted_files:
            print(f"  {f}")

if __name__ == '__main__':
    main()
