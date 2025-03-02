#!/usr/bin/env python3
"""
Efficient FITS image stacking script based on Stellina telescope log data.
This script applies transformations directly from the log file and
uses minimal processing for speed.
"""

import os
import json
import glob
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
from skimage import transform
from tqdm import tqdm
import argparse
import time

def parse_log_file(log_file_path):
    """Extract transformation matrices from the Stellina log file"""
    print(f"Parsing log file: {log_file_path}")
    
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    # Parse JSON
    try:
        log_data = json.loads(log_content)
    except json.JSONDecodeError:
        print("Error parsing JSON. Check the log file format.")
        return []
    
    transformations = []
    
    # Extract successful registration results
    if 'data' in log_data and 'corrections' in log_data['data']:
        for correction in log_data['data']['corrections']:
            if 'postCorrections' in correction:
                for post_correction in correction['postCorrections']:
                    if 'acqFile' in post_correction and 'stackingData' in post_correction['acqFile']:
                        stacking_data = post_correction['acqFile']['stackingData']
                        
                        if stacking_data.get('error') is None and 'liveRegistrationResult' in stacking_data:
                            reg_result = stacking_data['liveRegistrationResult']
                            
                            if 'matrix' in reg_result and reg_result.get('statusMessage') == 'StackingOk':
                                idx = reg_result['idx']
                                matrix = reg_result['matrix']
                                roundness = reg_result.get('roundness', 0)
                                stars_used = reg_result.get('starsUsed', 0)
                                
                                # Construct the transformation matrix
                                transform_matrix = np.array([
                                    [matrix[0], matrix[1], matrix[2]],
                                    [matrix[3], matrix[4], matrix[5]],
                                    [matrix[6], matrix[7], matrix[8]]
                                ])
                                
                                # Get file information
                                file_index = post_correction['acqFile'].get('index', -1)
                                file_path = post_correction['acqFile'].get('path', '')
                                
                                transformations.append({
                                    'idx': idx,
                                    'matrix': transform_matrix,
                                    'roundness': roundness,
                                    'stars_used': stars_used,
                                    'file_index': file_index,
                                    'file_path': file_path
                                })
    
    # Sort by the original stacking index
    transformations.sort(key=lambda x: x['idx'])
    print(f"Extracted {len(transformations)} valid transformations")
    return transformations

def apply_transformation(image, matrix):
    """Apply a 3x3 transformation matrix to an image"""
    # The matrices in the log are already in the correct format for image registration
    tform = transform.AffineTransform(matrix=matrix)
    
    # Apply the transformation - use linear interpolation for speed
    transformed = transform.warp(image, inverse_map=tform.inverse, 
                                preserve_range=True, 
                                order=1,  # Use linear interpolation (faster than cubic)
                                mode='constant', cval=0)
    
    return transformed.astype(image.dtype)

def stack_images(images, method='mean'):
    """Stack multiple images together efficiently"""
    if not images:
        return None
        
    # Convert to float32 (not float64) for better memory usage
    float_images = [img.astype(np.float32) for img in images]
    
    # Stack the images
    print(f"Stacking {len(float_images)} images...")
    if method == 'mean':
        stacked = np.mean(float_images, axis=0)
    elif method == 'median':
        stacked = np.median(float_images, axis=0)
    elif method == 'sum':
        stacked = np.sum(float_images, axis=0)
    else:
        raise ValueError(f"Unknown stacking method: {method}")
    
    # Convert back to original dtype 
    return stacked.astype(images[0].dtype)

def process_fits_files(fits_directory, transformations, output_file, dark_frame_path=None, 
                      stacking_method='mean', quality_threshold=0.4, 
                      max_files=None):
    """Process FITS files using the extracted transformations"""
    start_time = time.time()
    print(f"Processing FITS files from: {fits_directory}")
    
    # Find all FITS files in the directory
    fits_files = sorted(glob.glob(os.path.join(fits_directory, "*.fits")))
    
    if not fits_files:
        print(f"No FITS files found in {fits_directory}")
        return
    
    num_files = len(fits_files)
    print(f"Found {num_files} FITS files")
    
    # Load the master dark frame if provided
    master_dark = None
    if dark_frame_path and os.path.exists(dark_frame_path):
        try:
            with fits.open(dark_frame_path) as hdul:
                master_dark = hdul[0].data
            print(f"Loaded master dark frame from {dark_frame_path}")
        except Exception as e:
            print(f"Error loading master dark frame: {e}")
            print("Continuing without dark subtraction")
    
    # Read the first FITS file to get reference header
    with fits.open(fits_files[0]) as hdul:
        reference_header = hdul[0].header.copy()
    
    # Filter transformations by quality threshold
    valid_transforms = [t for t in transformations if t['roundness'] >= quality_threshold]
    print(f"Using {len(valid_transforms)} transformations after quality filtering")
    
    # Limit the number of files if requested
    if max_files and max_files > 0:
        valid_transforms = valid_transforms[:max_files]
        print(f"Limited to {len(valid_transforms)} files as requested")
    
    # Initialize list to hold transformed images
    transformed_images = []
    
    # Track which files we successfully processed
    processed_files = []
    
    # Process each transformation
    for transform_info in tqdm(valid_transforms, desc="Applying transformations"):
        # Find the corresponding FITS file
        file_index = transform_info['file_index']
        file_path = transform_info['file_path']
        
        # Try to match the file by index in the filename
        matching_files = [f for f in fits_files if f"_{file_index:04d}" in f or f"_{file_index}" in f]
        
        # Try different ways to find the file
        if matching_files:
            file_to_use = matching_files[0]
        elif len(fits_files) > file_index >= 0:
            # Use index as position in the sorted list
            file_to_use = fits_files[file_index]
        else:
            # Use original filename from log if possible
            basename = os.path.basename(file_path)
            matching_files = [f for f in fits_files if basename in f]
            if matching_files:
                file_to_use = matching_files[0]
            else:
                print(f"Could not find FITS file for index {file_index}")
                continue
        
        # Read the FITS file
        try:
            with fits.open(file_to_use) as hdul:
                image_data = hdul[0].data.astype(np.float32)
        except Exception as e:
            print(f"Error reading file {file_to_use}: {e}")
            continue
        
        # Apply dark frame subtraction if available
        if master_dark is not None:
            if master_dark.shape == image_data.shape:
                image_data = np.maximum(image_data - master_dark, 0)
            else:
                print(f"Warning: Dark frame shape {master_dark.shape} doesn't match image shape {image_data.shape}")
        
        # Apply the transformation from the log file
        matrix = transform_info['matrix']
        
        try:
            transformed = apply_transformation(image_data, matrix)
            transformed_images.append(transformed)
            processed_files.append(os.path.basename(file_to_use))
        except Exception as e:
            print(f"Error applying transformation to {file_to_use}: {e}")
            continue
        
    if not transformed_images:
        print("No images could be transformed. Check your files and transformation data.")
        return
        
    # Stack the transformed images
    stacked_image = stack_images(transformed_images, method=stacking_method)
    
    # Save the result
    new_hdu = fits.PrimaryHDU(data=stacked_image, header=reference_header)
    new_hdu.header['HISTORY'] = f'Stacked {len(transformed_images)} images using {stacking_method} method'
    new_hdu.writeto(output_file, overwrite=True)
    
    # Display a preview
    plt.figure(figsize=(10, 10))
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(stacked_image)
    plt.imshow(stacked_image, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar()
    plt.title(f'Stacked Image ({len(transformed_images)} frames)')
    plt.grid(True, alpha=0.3, color='white')
    preview_file = output_file.replace('.fits', '.png')
    plt.savefig(preview_file)
    
    # Save the list of processed files
    with open(output_file.replace('.fits', '_files.txt'), 'w') as f:
        for filename in processed_files:
            f.write(f"{filename}\n")
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.1f} seconds")
    print(f"Saved stacked image to {output_file}")
    print(f"Saved preview image to {preview_file}")

def main():
    parser = argparse.ArgumentParser(description='Efficiently stack FITS images based on Stellina log data')
    parser.add_argument('log_file', help='Path to the Stellina log file')
    parser.add_argument('fits_directory', help='Directory containing FITS files')
    parser.add_argument('--output', '-o', default='stacked.fits', help='Output file path')
    parser.add_argument('--dark', '-d', help='Path to master dark frame')
    parser.add_argument('--method', '-m', choices=['mean', 'median', 'sum'], default='mean',
                       help='Stacking method (default: mean)')
    parser.add_argument('--quality', '-q', type=float, default=0.4,
                       help='Quality threshold for frame roundness (default: 0.4)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    
    args = parser.parse_args()
    
    # Parse the log file
    transformations = parse_log_file(args.log_file)
    
    # Process the FITS files
    process_fits_files(
        args.fits_directory, 
        transformations, 
        args.output,
        dark_frame_path=args.dark,
        stacking_method=args.method, 
        quality_threshold=args.quality,
        max_files=args.max_files
    )

if __name__ == "__main__":
    main()
