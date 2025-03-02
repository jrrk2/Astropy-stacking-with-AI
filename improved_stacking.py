#!/usr/bin/env python3
"""
Improved FITS stacking script for Stellina telescope data with BGGR Bayer pattern.
This script learns from Astro Pixel Processor's approach and handles BGGR Bayer data
with 180-degree rotator offset.
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
                                mean_value = post_correction['acqFile'].get('mean', 0)
                                
                                transformations.append({
                                    'idx': idx,
                                    'matrix': transform_matrix,
                                    'roundness': roundness,
                                    'stars_used': stars_used,
                                    'file_index': file_index,
                                    'file_path': file_path,
                                    'mean_value': mean_value
                                })
    
    # Sort by the original stacking index
    transformations.sort(key=lambda x: x['idx'])
    print(f"Extracted {len(transformations)} valid transformations")
    return transformations

def select_reference_frame(transformations, fits_files, quality_threshold=0.4):
    """
    Select the best reference frame based on quality metrics
    Similar to how APP chooses a reference frame
    """
    # Filter by quality threshold
    good_frames = [t for t in transformations if t['roundness'] >= quality_threshold]
    
    if not good_frames:
        print("No frames meet the quality threshold. Using first transformation.")
        return transformations[0] if transformations else None
    
    # Sort by number of stars used for registration (higher is better)
    good_frames.sort(key=lambda x: x['stars_used'], reverse=True)
    
    # Take the top 25% of frames based on star count
    top_frames = good_frames[:max(1, len(good_frames) // 4)]
    
    # From these, select the one with best roundness
    top_frames.sort(key=lambda x: x['roundness'], reverse=True)
    
    print(f"Selected reference frame idx={top_frames[0]['idx']} with "
          f"{top_frames[0]['stars_used']} stars and roundness={top_frames[0]['roundness']:.3f}")
    
    return top_frames[0]

def apply_transformation(image, matrix, rotation_180=False, order=3):
    """
    Apply a 3x3 transformation matrix to an image
    
    Parameters:
    -----------
    image : 2D array
        Input image
    matrix : 3x3 array
        Transformation matrix
    rotation_180 : bool
        Whether to apply 180 degree rotation to handle rotator offset
    order : int
        Interpolation order (1=linear, 3=cubic, 5=lanczos)
    """
    # Apply 180 degree rotation if needed
    if rotation_180:
        # Create rotation matrix for 180 degrees
        rot_matrix = np.array([
            [-1, 0, image.shape[1]],  # Flip X and translate
            [0, -1, image.shape[0]],  # Flip Y and translate
            [0, 0, 1]
        ])
        
        # Combine with the existing transformation
        matrix = np.dot(rot_matrix, matrix)
    
    # Create the transformation
    tform = transform.ProjectiveTransform(matrix=matrix)
    
    # Apply the transformation with specified interpolation order
    transformed = transform.warp(image, inverse_map=tform.inverse, 
                                preserve_range=True, 
                                order=order,     # Interpolation order
                                mode='constant', 
                                cval=0)
    
    return transformed.astype(image.dtype)

def debayer_simple(bayer_img):
    """
    Simple debayering of BGGR pattern (just for preview, not for stacking)
    Note: For actual stacking, we process the raw Bayer data directly
    """
    h, w = bayer_img.shape
    result = np.zeros((h, w, 3), dtype=bayer_img.dtype)
    
    # BGGR pattern: B at (0,0), G at (0,1) and (1,0), R at (1,1)
    # Blue channel (0, 0)
    result[0::2, 0::2, 2] = bayer_img[0::2, 0::2]
    
    # Green channel (0, 1) and (1, 0)
    result[0::2, 1::2, 1] = bayer_img[0::2, 1::2]
    result[1::2, 0::2, 1] = bayer_img[1::2, 0::2]
    
    # Red channel (1, 1)
    result[1::2, 1::2, 0] = bayer_img[1::2, 1::2]
    
    # Simple nearest-neighbor interpolation for missing values
    # Blue
    result[0::2, 1::2, 2] = result[0::2, 0::2, 2]
    result[1::2, :, 2] = result[0::2, :, 2]
    
    # Green (already has half the pixels filled)
    result[1::2, 1::2, 1] = result[1::2, 0::2, 1]
    
    # Red
    result[0::2, :, 0] = result[1::2, :, 0]
    result[1::2, 0::2, 0] = result[1::2, 1::2, 0]
    
    return result

def stack_images(images, method='mean'):
    """Stack multiple images together efficiently"""
    if not images:
        return None
        
    # Convert to float32 (not float64) for better memory usage
    float_images = [img.astype(np.float32) for img in images]
    
    # Stack the images
    print(f"Stacking {len(float_images)} images using {method} method...")
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

def process_fits_files(fits_directory, transformations, output_file, 
                      dark_frame_path=None, stacking_method='mean', 
                      quality_threshold=0.4, max_files=None,
                      interpolation_order=3, rotate_180=True):
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
    
    # Select a reference frame (similar to APP approach)
    reference_info = select_reference_frame(transformations, fits_files, quality_threshold)
    if not reference_info:
        print("Unable to select a reference frame. Exiting.")
        return
    
    # Read the reference frame to get header information and as alignment target
    reference_file_index = reference_info['file_index']
    matching_files = [f for f in fits_files if f"_{reference_file_index:04d}" in f 
                     or f"_{reference_file_index}" in f]
    
    if not matching_files and len(fits_files) > reference_file_index >= 0:
        reference_file = fits_files[reference_file_index]
    elif matching_files:
        reference_file = matching_files[0]
    else:
        print(f"Could not locate reference file with index {reference_file_index}")
        print("Using first available file as reference")
        reference_file = fits_files[0]
    
    print(f"Using {reference_file} as reference frame")
    
    # Load the reference frame
    try:
        with fits.open(reference_file) as hdul:
            reference_data = hdul[0].data.astype(np.float32)
            reference_header = hdul[0].header.copy()
            
            # Apply dark subtraction to reference if needed
            if master_dark is not None and master_dark.shape == reference_data.shape:
                reference_data = np.maximum(reference_data - master_dark, 0)
                
            # Handle Bayer pattern data - check dimensions
            if len(reference_data.shape) == 2:
                print("Detected 2D data (likely raw Bayer pattern)")
                is_bayer = True
            else:
                print(f"Detected {len(reference_data.shape)}D data")
                is_bayer = False
    except Exception as e:
        print(f"Error reading reference file: {e}")
        return
    
    # Filter transformations by quality threshold
    valid_transforms = [t for t in transformations if t['roundness'] >= quality_threshold]
    print(f"Using {len(valid_transforms)} transformations after quality filtering")
    
    # Limit the number of files if requested
    if max_files and max_files > 0:
        valid_transforms = valid_transforms[:max_files]
        print(f"Limited to {len(valid_transforms)} files as requested")
    
    # Calculate transformation to reference frame for each frame
    reference_matrix = reference_info['matrix']
    reference_matrix_inv = np.linalg.inv(reference_matrix)
    
    # Initialize list to hold transformed images
    transformed_images = []
    
    # Track which files we successfully processed
    processed_files = []
    
    # Process each transformation
    for transform_info in tqdm(valid_transforms, desc="Applying transformations"):
        # Skip reference frame to avoid duplication
        if transform_info['idx'] == reference_info['idx']:
            continue
            
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
        
        # Calculate transformation relative to reference frame
        # This is similar to how APP would align all frames to a reference
        frame_matrix = transform_info['matrix']
        
        # Combine transformations: frame -> reference
        # The transformation needed is: frame -> world -> reference
        # So we use: reference_matrix_inv @ frame_matrix
        combined_matrix = np.dot(reference_matrix_inv, frame_matrix)
        
        # Apply the combined transformation
        try:
            transformed = apply_transformation(
                image_data, 
                combined_matrix, 
                rotation_180=rotate_180,
                order=interpolation_order
            )
            transformed_images.append(transformed)
            processed_files.append(os.path.basename(file_to_use))
        except Exception as e:
            print(f"Error applying transformation to {file_to_use}: {e}")
            continue
    
    # Add reference frame itself
    if rotate_180:
        # If we're rotating everything else, we need to rotate the reference too
        rot_matrix = np.array([
            [-1, 0, reference_data.shape[1]],
            [0, -1, reference_data.shape[0]],
            [0, 0, 1]
        ])
        reference_rotated = apply_transformation(
            reference_data, rot_matrix, rotation_180=False, order=interpolation_order
        )
        transformed_images.append(reference_rotated)
    else:
        transformed_images.append(reference_data)
        
    processed_files.append(os.path.basename(reference_file))
    
    if not transformed_images:
        print("No images could be transformed. Check your files and transformation data.")
        return
        
    # Stack the transformed images
    stacked_image = stack_images(transformed_images, method=stacking_method)
    
    # Save the result
    new_hdu = fits.PrimaryHDU(data=stacked_image, header=reference_header)
    
    # Add APP-like headers
    new_hdu.header['REGMODEL'] = 'projective'
    new_hdu.header['REFERENC'] = os.path.basename(reference_file)
    new_hdu.header['REGMODE'] = 'normal'
    new_hdu.header['INTERPOL'] = f'order-{interpolation_order}'
    new_hdu.header['ROTATE180'] = str(rotate_180)
    new_hdu.header['CFAIMAGE'] = 'yes' if is_bayer else 'no'
    if is_bayer:
        new_hdu.header['BAYERPAT'] = 'BGGR'
    new_hdu.header['HISTORY'] = f'Stacked {len(transformed_images)} images using {stacking_method} method'
    
    new_hdu.writeto(output_file, overwrite=True)
    
    # Display a preview
    plt.figure(figsize=(10, 10))
    
    # For Bayer data, create a simple debayered preview
    if is_bayer:
        preview_data = debayer_simple(stacked_image)
        plt.imshow(preview_data)
        plt.title(f'Stacked & Debayered Preview ({len(transformed_images)} frames)')
    else:
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(stacked_image)
        plt.imshow(stacked_image, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        plt.title(f'Stacked Image ({len(transformed_images)} frames)')
    
    plt.colorbar()
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
    parser = argparse.ArgumentParser(description='Stack FITS images with BGGR Bayer pattern based on Stellina log data')
    parser.add_argument('log_file', help='Path to the Stellina log file')
    parser.add_argument('fits_directory', help='Directory containing FITS files')
    parser.add_argument('--output', '-o', default='stacked.fits', help='Output file path')
    parser.add_argument('--dark', '-d', help='Path to master dark frame')
    parser.add_argument('--method', '-m', choices=['mean', 'median', 'sum'], default='mean',
                       help='Stacking method (default: mean)')
    parser.add_argument('--quality', '-q', type=float, default=0.4,
                       help='Quality threshold for frame roundness (default: 0.4)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--interpolation', '-i', type=int, choices=[1, 3, 5], default=3,
                      help='Interpolation order: 1=linear, 3=cubic, 5=lanczos (default: 3)')
    parser.add_argument('--no-rotation', action='store_true',
                      help='Disable 180-degree rotation correction')
    
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
        max_files=args.max_files,
        interpolation_order=args.interpolation,
        rotate_180=not args.no_rotation
    )

if __name__ == "__main__":
    main()
