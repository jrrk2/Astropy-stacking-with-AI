#!/usr/bin/env python3
"""
Diagnostic script for Stellina FITS files
"""

import os
import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def examine_file(file_path):
    """
    Examine a file to determine its format and content
    """
    print(f"\nExamining: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    # Try to read as FITS with various options
    try:
        with fits.open(file_path) as hdul:
            print(f"Successfully opened as standard FITS")
            print(f"Number of HDUs: {len(hdul)}")
            
            # Print header info for first HDU
            print("\nFirst HDU header:")
            for k, v in list(hdul[0].header.items())[:10]:  # First 10 header items
                print(f"  {k}: {v}")
            
            # Show data shape and type
            data = hdul[0].data
            if data is not None:
                print(f"\nData shape: {data.shape}")
                print(f"Data type: {data.dtype}")
                print(f"Data min/max: {np.min(data)}/{np.max(data)}")
                
                # Create a simple plot of the data
                plt.figure(figsize=(10, 8))
                plt.imshow(data, cmap='viridis', origin='lower')
                plt.colorbar(label='Pixel Value')
                plt.title(f"Image from {os.path.basename(file_path)}")
                plt.savefig(f"diagnostic_{os.path.basename(file_path)}.png")
                plt.close()
                print(f"Saved image preview as diagnostic_{os.path.basename(file_path)}.png")
            else:
                print("No data in the primary HDU")
    except Exception as e:
        print(f"Failed to open as standard FITS: {str(e)}")
        
        # Try with ignore_missing_simple
        try:
            with fits.open(file_path, ignore_missing_simple=True) as hdul:
                print(f"Successfully opened with ignore_missing_simple=True")
                print(f"Number of HDUs: {len(hdul)}")
                
                # Print header info for first HDU
                print("\nFirst HDU header:")
                for k, v in list(hdul[0].header.items())[:10]:  # First 10 header items
                    print(f"  {k}: {v}")
                
                # Show data shape and type
                data = hdul[0].data
                if data is not None:
                    print(f"\nData shape: {data.shape}")
                    print(f"Data type: {data.dtype}")
                    print(f"Data min/max: {np.min(data)}/{np.max(data)}")
                    
                    # Create a simple plot of the data
                    plt.figure(figsize=(10, 8))
                    plt.imshow(data, cmap='viridis', origin='lower')
                    plt.colorbar(label='Pixel Value')
                    plt.title(f"Image from {os.path.basename(file_path)} (ignore_missing_simple)")
                    plt.savefig(f"diagnostic_ignore_{os.path.basename(file_path)}.png")
                    plt.close()
                    print(f"Saved image preview as diagnostic_ignore_{os.path.basename(file_path)}.png")
                else:
                    print("No data in the primary HDU")
        except Exception as e2:
            print(f"Failed with ignore_missing_simple=True: {str(e2)}")
    
    # Try to read first few bytes to determine format
    try:
        with open(file_path, 'rb') as f:
            header_bytes = f.read(16)
            print("\nFirst 16 bytes:")
            print(" ".join(f"{b:02X}" for b in header_bytes))
            
            # Check for common file signatures
            if header_bytes.startswith(b'SIMPLE  ='):
                print("File appears to be standard FITS (starts with 'SIMPLE  =')")
            elif header_bytes.startswith(b'\x89PNG'):
                print("File appears to be PNG format")
            elif header_bytes.startswith(b'\xFF\xD8\xFF'):
                print("File appears to be JPEG format")
            elif header_bytes[0:2] == b'BM':
                print("File appears to be BMP format")
            elif header_bytes.startswith(b'P6') or header_bytes.startswith(b'P5'):
                print("File appears to be PPM/PGM format")
    except Exception as e:
        print(f"Error examining raw file content: {str(e)}")

def main():
    """
    Main function
    """
    if len(sys.argv) < 2:
        print("Usage: python stellina_diag.py <directory_or_file>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        # Get all files in directory
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        
        # Only examine first few files
        max_files = 3
        print(f"Found {len(files)} files, examining first {min(max_files, len(files))}")
        
        for file_path in files[:max_files]:
            examine_file(file_path)
    else:
        # Single file
        examine_file(path)

if __name__ == "__main__":
    main()
