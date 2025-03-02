#!/usr/bin/env python3
"""
Complete FITS stacking script for Stellina telescope data with proper dark subtraction 
and APP-like registration to a common reference frame.
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
import sys
from pathlib import Path

# Import dark_manager - must be in the same directory
try:
    from dark_manager import DarkFrameManager
    DARK_MANAGER_AVAILABLE = True
except ImportError:
    print("Warning: dark_manager.py not found. Dark calibration functionality will be limited.")
    DARK_MANAGER_AVAILABLE = False

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
                                
                                # Get temperature metadata if available
                                motors = post_correction['acqFile'].get('motors', {})
                                metadata = post_correction['acqFile'].get('metadata', {})
                                
                                transformations.append({
                                    'idx': idx,
                                    'matrix': transform_matrix,
                                    'roundness': roundness,
                                    'stars_used': stars_used,
                                    'file_index': file_index,
                                    'file_path': file_path,
                                    'mean_value': mean_value,
                                    'motors': motors,
                                    'metadata': metadata
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

def calculate_padded_dimensions(fits_files, transformations, margin_percent=10):
    """
    Calculate the dimensions for a padded output image that can hold all transformed frames
    based on APP's approach of using a larger canvas
    """
    # First find the max dimensions of the input images
    max_width, max_height = 0, 0
    
    for fits_file in fits_files:
        try:
            with fits.open(fits_file) as hdul:
                height, width = hdul[0].data.shape[-2:]
                max_width = max(max_width, width)
                max_height = max(max_height, height)
        except Exception as e:
            print(f"Error reading dimensions from {fits_file}: {e}")
    
    # Add margin to ensure all transformed frames fit
    margin_width = int(max_width * margin_percent / 100)
    margin_height = int(max_height * margin_percent / 100)
    
    padded_width = max_width + 2 * margin_width
    padded_height = max_height + 2 * margin_height
    
    # Ensure dimensions are even (sometimes helps with certain algorithms)
    padded_width += padded_width % 2
    padded_height += padded_height % 2
    
    print(f"Original max dimensions: {max_width}x{max_height}")
    print(f"Padded dimensions: {padded_width}x{padded_height} (with {margin_percent}% margin)")
    
    return padded_width, padded_height, margin_width, margin_height

def apply_transformation(image, matrix, output_shape=None, rotation_180=False, order=3):
    """
    Apply a 3x3 transformation matrix to an image with optional padding and rotation
    
    Parameters:
    -----------
    image : 2D array
        Input image
    matrix : 3x3 array
        Transformation matrix
    output_shape : tuple
        (height, width) of the output image (for padding)
    rotation_180 : bool
        Whether to apply 180 degree rotation to handle rotator offset
    order : int
        Interpolation order (1=linear, 3=cubic)
    """
    h, w = image.shape
    
    # If no output shape is provided, use the input shape
    if output_shape is None:
        output_shape = (h, w)
    
    # Calculate translation to center the image in the padded canvas
    if output_shape[0] > h or output_shape[1] > w:
        # Calculate the center shift
        y_shift = (output_shape[0] - h) / 2
        x_shift = (output_shape[1] - w) / 2
        
        # Create a translation matrix to center the image
        center_matrix = np.array([
            [1, 0, x_shift],
            [0, 1, y_shift],
            [0, 0, 1]
        ])
        
        # Apply centering first
        matrix = np.dot(center_matrix, matrix)
    
    # Apply 180 degree rotation if needed
    if rotation_180:
        # Create rotation matrix for 180 degrees (around the center of the output)
        rot_matrix = np.array([
            [-1, 0, output_shape[1]],  # Flip X and translate
            [0, -1, output_shape[0]],  # Flip Y and translate
            [0, 0, 1]
        ])
        
        # Combine with the existing transformation
        matrix = np.dot(rot_matrix, matrix)
    
    # Create the transformation
    tform = transform.ProjectiveTransform(matrix=matrix)
    
    # Apply the transformation with specified output shape and interpolation order
    transformed = transform.warp(image, inverse_map=tform.inverse, 
                                output_shape=output_shape,
                                order=order,     # Interpolation order
                                mode='constant', 
                                cval=0,
                                preserve_range=True)
    
    return transformed.astype(image.dtype)

def stack_images(images, method='mean'):
    """Stack multiple images together efficiently"""
    if not images:
        return None
        
    # Convert to float32 for efficiency
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

def get_dark_frame(fits_file, dark_frame_path=None, dark_manager=None):
    """
    Get appropriate dark frame for calibration
    
    Parameters:
    -----------
    fits_file : str
        Path to the FITS file to be calibrated
    dark_frame_path : str
        Path to a fixed dark frame file
    dark_manager : DarkFrameManager
        Initialized dark frame manager for temperature-matched calibration
    
    Returns:
    --------
    dark_data : ndarray or None
        Dark frame data if available, or None
    """
    # If no dark frame sources, return None
    if dark_frame_path is None and dark_manager is None:
        return None, "No dark frame source provided"
    
    # Try to read the FITS file to get metadata
    try:
        with fits.open(fits_file) as hdul:
            light_data = hdul[0].data
            header = hdul[0].header
    except Exception as e:
        return None, f"Error reading {fits_file}: {e}"
    
    # Option 1: Use dark manager with temperature matching
    if dark_manager is not None:
        # Get temp from header or estimate
        temp_c = header.get('CCD-TEMP', 20)  # Default to 20°C if not found
        
        # Determine Bayer pattern
        pattern = header.get('BAYERPAT', 'RGGB').strip()
        
        # Get dark frame from manager
        try:
            dark_data, message = dark_manager.get_dark_frame(temp_c, pattern, light_data)
            return dark_data, message
        except Exception as e:
            print(f"Error getting dark frame from manager: {e}")
            # Fall back to fixed dark frame if available
    
    # Option 2: Use fixed dark frame
    if dark_frame_path:
        try:
            with fits.open(dark_frame_path) as hdul:
                dark_data = hdul[0].data
                return dark_data, f"Using fixed dark frame: {dark_frame_path}"
        except Exception as e:
            return None, f"Error loading dark frame {dark_frame_path}: {e}"
    
    return None, "Failed to get appropriate dark frame"

def process_fits_files(fits_directory, transformations, output_file, 
                      dark_frame_path=None, dark_directory=None,
                      stacking_method='mean', quality_threshold=0.4, max_files=None,
                      interpolation_order=3, rotate_180=True, margin_percent=10,
                      normalize_output=False, save_calibrated=False):
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
    
    # Initialize dark manager if directory provided
    dark_manager = None
    if dark_directory and DARK_MANAGER_AVAILABLE:
        try:
            dark_manager = DarkFrameManager(dark_directory)
            print(f"Initialized dark manager from {dark_directory}")
        except Exception as e:
            print(f"Error initializing dark manager: {e}")
            print("Falling back to fixed dark frame if provided")
    
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
            
            # Check for Bayer pattern data
            if len(reference_data.shape) == 2:
                print("Detected 2D data (likely raw Bayer pattern)")
                is_bayer = True
                pattern = reference_header.get('BAYERPAT', 'RGGB').strip()
                print(f"Bayer pattern: {pattern}")
            else:
                print(f"Detected {len(reference_data.shape)}D data")
                is_bayer = False
                pattern = None
    except Exception as e:
        print(f"Error reading reference file: {e}")
        return
    
    # Calculate padded dimensions for the output (like APP does)
    padded_width, padded_height, margin_width, margin_height = calculate_padded_dimensions(
        fits_files, transformations, margin_percent
    )
    output_shape = (padded_height, padded_width)
    
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
    dark_messages = []
    
    # Process each transformation
    for transform_info in tqdm(valid_transforms, desc="Processing frames"):
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
        
        # Create calibrated output directories if requested
        if save_calibrated:
            # Create output directories for calibrated and transformed frames
            calibrated_dir = os.path.join(os.path.dirname(output_file), "calibrated_frames")
            transform_dir = os.path.join(os.path.dirname(output_file), "transformed_frames")
            os.makedirs(calibrated_dir, exist_ok=True)
            os.makedirs(transform_dir, exist_ok=True)
            
            # Save original frame
            orig_output = os.path.join(calibrated_dir, f"orig_{os.path.basename(file_to_use)}")
            try:
                fits.PrimaryHDU(data=image_data).writeto(orig_output, overwrite=True)
                print(f"Saved original frame: {orig_output}")
            except Exception as e:
                print(f"Error saving original frame: {e}")
        
        # Apply dark frame subtraction
        dark_data, message = get_dark_frame(file_to_use, dark_frame_path, dark_manager)
        if dark_data is not None:
            if dark_data.shape == image_data.shape:
                # Apply dark subtraction
                image_data = np.maximum(image_data - dark_data, 0)
                dark_messages.append(f"{os.path.basename(file_to_use)}: {message}")
            else:
                print(f"Warning: Dark frame shape {dark_data.shape} doesn't match image shape {image_data.shape}")
        
        # Save calibrated frame after dark subtraction (if applicable)
        if save_calibrated:
            cal_output = os.path.join(calibrated_dir, f"cal_{os.path.basename(file_to_use)}")
            try:
                fits.PrimaryHDU(data=image_data).writeto(cal_output, overwrite=True)
                print(f"Saved calibrated frame: {cal_output}")
            except Exception as e:
                print(f"Error saving calibrated frame: {e}")
        
        # Calculate transformation relative to reference frame
        # This is similar to how APP would align all frames to a reference
        frame_matrix = transform_info['matrix']
        
        # Combine transformations: frame -> world -> reference
        # The transformation needed is: frame -> world -> reference
        # So we use: reference_matrix_inv @ frame_matrix
        combined_matrix = np.dot(reference_matrix_inv, frame_matrix)
        
        # Apply the combined transformation to the padded output shape
        try:
            transformed = apply_transformation(
                image_data, 
                combined_matrix,
                output_shape=output_shape,
                rotation_180=rotate_180,
                order=interpolation_order
            )
            transformed_images.append(transformed)
            processed_files.append(os.path.basename(file_to_use))
            
            # Save transformed frame if requested
            if save_calibrated:
                trans_output = os.path.join(transform_dir, f"trans_{os.path.basename(file_to_use)}")
                try:
                    fits.PrimaryHDU(data=transformed).writeto(trans_output, overwrite=True)
                    print(f"Saved transformed frame: {trans_output}")
                except Exception as e:
                    print(f"Error saving transformed frame: {e}")
        except Exception as e:
            print(f"Error applying transformation to {file_to_use}: {e}")
            continue
    
    # Add reference frame itself (transformed to the padded output)
    # Get dark frame for reference
    ref_dark_data, message = get_dark_frame(reference_file, dark_frame_path, dark_manager)
    
    # Save original reference frame if requested
    if save_calibrated:
        orig_ref_output = os.path.join(calibrated_dir, f"orig_{os.path.basename(reference_file)}")
        try:
            fits.PrimaryHDU(data=reference_data).writeto(orig_ref_output, overwrite=True)
            print(f"Saved original reference frame: {orig_ref_output}")
        except Exception as e:
            print(f"Error saving original reference frame: {e}")
    
    # Apply dark subtraction to reference frame if available
    if ref_dark_data is not None and ref_dark_data.shape == reference_data.shape:
        reference_data = np.maximum(reference_data - ref_dark_data, 0)
        dark_messages.append(f"{os.path.basename(reference_file)}: {message}")
    
    # Save calibrated reference frame if requested
    if save_calibrated:
        cal_ref_output = os.path.join(calibrated_dir, f"cal_{os.path.basename(reference_file)}")
        try:
            fits.PrimaryHDU(data=reference_data).writeto(cal_ref_output, overwrite=True)
            print(f"Saved calibrated reference frame: {cal_ref_output}")
        except Exception as e:
            print(f"Error saving calibrated reference frame: {e}")
    
    # Apply identity transform to reference (just to center it in the padded canvas)
    identity_matrix = np.eye(3)
    reference_padded = apply_transformation(
        reference_data,
        identity_matrix, 
        output_shape=output_shape,
        rotation_180=rotate_180,
        order=interpolation_order
    )
    transformed_images.append(reference_padded)
    processed_files.append(os.path.basename(reference_file))
    
    # Save transformed reference frame if requested
    if save_calibrated:
        trans_ref_output = os.path.join(transform_dir, f"trans_{os.path.basename(reference_file)}")
        try:
            fits.PrimaryHDU(data=reference_padded).writeto(trans_ref_output, overwrite=True)
            print(f"Saved transformed reference frame: {trans_ref_output}")
        except Exception as e:
            print(f"Error saving transformed reference frame: {e}")
    
    if not transformed_images:
        print("No images could be transformed. Check your files and transformation data.")
        return
        
    # Stack the transformed images
    stacked_image = stack_images(transformed_images, method=stacking_method)
    
    # Normalize to 0-1 range if requested (like APP does)
    if normalize_output:
        print("Normalizing output to 0-1 range")
        min_val = np.min(stacked_image)
        max_val = np.max(stacked_image)
        if max_val > min_val:
            stacked_image = (stacked_image - min_val) / (max_val - min_val)
    
    # Save the result
    new_hdu = fits.PrimaryHDU(data=stacked_image, header=reference_header)
    
    # Add APP-like headers
    new_hdu.header['REGMODEL'] = 'projective'
    new_hdu.header['REFERENC'] = os.path.basename(reference_file)
    new_hdu.header['REGMODE'] = 'normal'
    new_hdu.header['INTERPOL'] = f'order-{interpolation_order}'
    new_hdu.header['ROTATE180'] = str(rotate_180)
    new_hdu.header['CFAIMAGE'] = 'yes' if is_bayer else 'no'
    new_hdu.header['NFRAMES'] = len(transformed_images)
    if is_bayer:
        new_hdu.header['BAYERPAT'] = pattern
    new_hdu.header['HISTORY'] = f'Stacked {len(transformed_images)} images using {stacking_method} method'
    
    # Write output
    try:
        new_hdu.writeto(output_file, overwrite=True)
    except Exception as e:
        print(f"Error writing output file: {e}")
        alternative_path = os.path.join(os.path.dirname(output_file), f"alt_{os.path.basename(output_file)}")
        print(f"Trying alternative path: {alternative_path}")
        try:
            new_hdu.writeto(alternative_path, overwrite=True)
            output_file = alternative_path
        except Exception as e2:
            print(f"Could not save output: {e2}")
            return
    
    # Display a preview
    plt.figure(figsize=(10, 10))
    
    # For visualization
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(stacked_image)
    plt.imshow(stacked_image, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    plt.title(f'Stacked Image ({len(transformed_images)} frames)')
    plt.colorbar()
    plt.grid(True, alpha=0.3, color='white')
    
    # Save the preview to same directory as output
    preview_file = output_file.replace('.fits', '.png')
    plt.savefig(preview_file)
    
    # Save processing information
    info_file = output_file.replace('.fits', '_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"Stacking Information\n")
        f.write(f"==================\n\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Stacking method: {stacking_method}\n")
        f.write(f"Number of frames: {len(transformed_images)}\n")
        f.write(f"Reference frame: {os.path.basename(reference_file)}\n")
        f.write(f"Output dimensions: {padded_width}x{padded_height}\n")
        f.write(f"Interpolation order: {interpolation_order}\n")
        f.write(f"180-degree rotation: {rotate_180}\n")
        f.write(f"Normalized output: {normalize_output}\n\n")
        
        f.write(f"Processed files:\n")
        for filename in processed_files:
            f.write(f"- {filename}\n")
            
        f.write(f"\nDark calibration:\n")
        for msg in dark_messages:
            f.write(f"- {msg}\n")
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.1f} seconds")
    print(f"Saved stacked image to {output_file}")
    print(f"Saved preview image to {preview_file}")
    print(f"Saved processing information to {info_file}")

def main():
    parser = argparse.ArgumentParser(description='Stack FITS images with APP-like registration based on Stellina log data')
    parser.add_argument('log_file', help='Path to the Stellina log file')
    parser.add_argument('fits_directory', help='Directory containing FITS files')
    parser.add_argument('--output', '-o', default='stacked.fits', help='Output file path')
    parser.add_argument('--dark', '-d', help='Path to master dark frame')
    parser.add_argument('--dark-dir', help='Directory with temperature-sorted dark frames')
    parser.add_argument('--method', '-m', choices=['mean', 'median', 'sum'], default='mean',
                       help='Stacking method (default: mean)')
    parser.add_argument('--quality', '-q', type=float, default=0.4,
                       help='Quality threshold for frame roundness (default: 0.4)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--interpolation', '-i', type=int, choices=[1, 3, 5], default=3,
                      help='Interpolation order: 1=linear, 3=cubic, 5=lanczos (default: 3)')
    parser.add_argument('--no-rotation', action='store_true',
                      help='Disable 180-degree rotation correction')
    parser.add_argument('--margin', type=int, default=10,
                      help='Margin percentage for output padding (default: 10)')
    parser.add_argument('--normalize', action='store_true',
                      help='Normalize output to 0-1 range like APP')
    parser.add_argument('--save-calibrated', action='store_true',
                      help='Save individual calibrated and transformed frames')
    
    args = parser.parse_args()
    
    # Parse the log file
    transformations = parse_log_file(args.log_file)
    
    # Process the FITS files
    process_fits_files(
        args.fits_directory, 
        transformations, 
        args.output,
        dark_frame_path=args.dark,
        dark_directory=args.dark_dir,
        stacking_method=args.method, 
        quality_threshold=args.quality,
        max_files=args.max_files,
        interpolation_order=args.interpolation,
        rotate_180=not args.no_rotation,
        margin_percent=args.margin,
        normalize_output=args.normalize,
        save_calibrated=args.save_calibrated
    )

if __name__ == "__main__":
    main()
