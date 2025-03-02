#!/usr/bin/env python3
"""
FITS stacking script that only uses frames with valid transformation matrices
in their headers, ignoring frames that don't have matrix annotations.
"""

import os
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

def check_for_transformation_matrix(fits_file):
    """Check if a FITS file has a valid transformation matrix in its header"""
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            
            # Check for shortened version first (TRNSFM)
            has_short_transform = all(f'TRNSFM{i+1}{j+1}' in header 
                                 for i in range(3) for j in range(3))
                                 
            # Check for full version next (TRNSFRM)
            has_long_transform = all(f'TRNSFRM{i+1}{j+1}' in header 
                                for i in range(3) for j in range(3))
            
            has_transform = has_short_transform or has_long_transform
            key_prefix = 'TRNSFM' if has_short_transform else 'TRNSFRM' if has_long_transform else None
            
            if has_transform and key_prefix:
                # Make sure matrix has non-zero determinant
                matrix = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        matrix[i, j] = header[f'{key_prefix}{i+1}{j+1}']
                
                # Check determinant
                det = np.linalg.det(matrix)
                if abs(det) < 1e-10:
                    print(f"Warning: Near-singular matrix in {os.path.basename(fits_file)} (det={det})")
                    return False, None
                    
                return True, key_prefix
            else:
                return False, None
                
    except Exception as e:
        print(f"Error reading {fits_file}: {e}")
        return False, None

def extract_transformations_from_fits(fits_files):
    """Extract transformation matrices from FITS headers"""
    print(f"Extracting transformations from {len(fits_files)} FITS files")
    
    transformations = []
    valid_files = []
    key_prefixes = {}  # Store the key prefix for each file
    
    for file_idx, fits_file in enumerate(fits_files):
        try:
            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                
                # Get the key prefix (TRNSFM or TRNSFRM) for this file
                has_matrix, key_prefix = check_for_transformation_matrix(fits_file)
                
                if has_matrix and key_prefix:
                    # Store the key prefix for later use
                    key_prefixes[fits_file] = key_prefix
                    
                    # Extract the 3x3 transformation matrix
                    matrix = np.zeros((3, 3))
                    for i in range(3):
                        for j in range(3):
                            matrix[i, j] = header[f'{key_prefix}{i+1}{j+1}']
                    
                    # Debug: Print the first matrix we find
                    if len(transformations) == 0:
                        print(f"\nFirst transformation matrix from {os.path.basename(fits_file)} using key prefix {key_prefix}:")
                        print(matrix)
                        print(f"Determinant: {np.linalg.det(matrix)}")
                        print(f"Shape: {matrix.shape}")
                        print()
                    
                    # Get registration quality metrics if available
                    roundness = header.get('REGRNDS', 0.5)  # Default to middle value if not found
                    stars_used = header.get('REGSTARS', 0)
                    mean_value = header.get('REGMEAN', 0)
                    reg_index = header.get('REGINDX', file_idx)
                    
                    transformations.append({
                        'idx': reg_index,
                        'matrix': matrix,
                        'roundness': roundness,
                        'stars_used': stars_used,
                        'file_index': file_idx,
                        'file_path': fits_file,
                        'mean_value': mean_value,
                        'key_prefix': key_prefix  # Store the key prefix
                    })
                    valid_files.append(fits_file)
                    print(f"  Extracted transformation from {os.path.basename(fits_file)} with {stars_used} stars and roundness {roundness:.3f}")
                else:
                    print(f"  Skipping {os.path.basename(fits_file)} - no transformation matrix found")
                    
        except Exception as e:
            print(f"Error reading {fits_file}: {e}")
    
    # Sort by the original stacking index
    transformations.sort(key=lambda x: x['idx'])
    print(f"Extracted {len(transformations)} valid transformations from {len(fits_files)} files")
    return transformations, valid_files, key_prefixes

def select_reference_frame(transformations, fits_files, quality_threshold=0.4):
    """
    Select the best reference frame based on quality metrics
    Similar to how APP chooses a reference frame
    """
    if not transformations:
        print("No transformations available to select reference frame.")
        return None
    
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
    # Use 'constant' mode with black (0) background
    transformed = transform.warp(image, inverse_map=tform.inverse, 
                                output_shape=output_shape,
                                order=order,     # Interpolation order
                                mode='constant', 
                                cval=0,          # Black background
                                preserve_range=True)
    
    return transformed.astype(image.dtype)

def stack_images(images, method='mean', memory_efficient=False, chunk_size=10):
    """Stack multiple images together efficiently"""
    if not images:
        return None
    
    # Memory-efficient stacking for large datasets
    if memory_efficient and len(images) > chunk_size:
        print(f"Using memory-efficient stacking with {len(images)} images in chunks of {chunk_size}...")
        
        # Process in smaller chunks
        num_chunks = (len(images) + chunk_size - 1) // chunk_size
        chunk_results = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(images))
            chunk = images[start_idx:end_idx]
            
            # Stack this chunk
            print(f"Stacking chunk {i+1}/{num_chunks} ({len(chunk)} images)...")
            chunk_result = stack_images(chunk, method, False)
            chunk_results.append(chunk_result)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Now stack the chunk results
        return stack_images(chunk_results, method, False)
        
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

def process_fits_files(fits_directory, output_file, 
                      stacking_method='median', quality_threshold=0.4, max_files=None,
                      interpolation_order=1, rotate_180=True, margin_percent=15,
                      normalize_output=False, save_transformed=False, stretch_contrast=True,
                      background_subtraction=True, memory_efficient=True):
    """Process FITS files using transformations from FITS headers"""
    start_time = time.time()
    print(f"Processing FITS files from: {fits_directory}")
    
    # Find all FITS files in the directory
    all_fits_files = sorted(glob.glob(os.path.join(fits_directory, "*.fits")))
    
    if not all_fits_files:
        print(f"No FITS files found in {fits_directory}")
        return
    
    print(f"Found {len(all_fits_files)} FITS files")
    
    # Filter files to only include those with valid transformation matrices
    print("Filtering files to only include those with valid transformation matrices...")
    files_with_matrices = []
    key_prefixes = {}
    
    # Use a memory-efficient approach to check files
    for fits_file in tqdm(all_fits_files, desc="Checking files"):
        has_matrix, key_prefix = check_for_transformation_matrix(fits_file)
        if has_matrix:
            files_with_matrices.append(fits_file)
            key_prefixes[fits_file] = key_prefix
    
    if not files_with_matrices:
        print("No files with valid transformation matrices found. Cannot proceed.")
        return
    
    print(f"Found {len(files_with_matrices)} files with valid transformation matrices")
    
    # Extract transformations from FITS headers
    transformations, valid_files, key_prefix_dict = extract_transformations_from_fits(files_with_matrices)
    # Merge the key_prefix dictionaries
    key_prefixes.update(key_prefix_dict)
    
    if not transformations:
        print("No valid transformations found in FITS headers. Cannot proceed.")
        return
    
    # Use only valid files for processing
    fits_files = valid_files
    
    # Select a reference frame
    reference_info = select_reference_frame(transformations, fits_files, quality_threshold)
    if not reference_info:
        print("Unable to select a reference frame. Exiting.")
        return
    
    # Get the reference file
    reference_file = reference_info['file_path']
    print(f"Using {os.path.basename(reference_file)} as reference frame")
    
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
    
    # Calculate padded dimensions for the output
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
    
    # Create output directory for transformed frames if needed
    if save_transformed:
        transform_dir = os.path.join(os.path.dirname(output_file), "transformed_frames")
        os.makedirs(transform_dir, exist_ok=True)
        print(f"Created directory for transformed files: {transform_dir}")
    
    # Process each transformation
    for transform_info in tqdm(valid_transforms, desc="Processing frames"):
        # Skip reference frame to avoid duplication 
        if transform_info['idx'] == reference_info['idx']:
            continue
            
        # Get the file path for this transform
        file_to_use = transform_info['file_path']
        
        # Read the FITS file
        try:
            with fits.open(file_to_use) as hdul:
                image_data = hdul[0].data.astype(np.float32)
        except Exception as e:
            print(f"Error reading file {file_to_use}: {e}")
            continue
        
        # Calculate transformation relative to reference frame
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
            if save_transformed:
                trans_output = os.path.join(transform_dir, f"trans_{os.path.basename(file_to_use)}")
                try:
                    fits.PrimaryHDU(data=transformed).writeto(trans_output, overwrite=True)
                except Exception as e:
                    print(f"Error saving transformed frame: {e}")
        except Exception as e:
            print(f"Error applying transformation to {file_to_use}: {e}")
            continue
    
    # Add reference frame itself (transformed to the padded output)
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
    if save_transformed:
        trans_ref_output = os.path.join(transform_dir, f"trans_{os.path.basename(reference_file)}")
        try:
            fits.PrimaryHDU(data=reference_padded).writeto(trans_ref_output, overwrite=True)
        except Exception as e:
            print(f"Error saving transformed reference frame: {e}")
    
    if not transformed_images:
        print("No images could be transformed. Check your files and transformation data.")
        return
        
    # Stack the transformed images
    stacked_image = stack_images(transformed_images, method=stacking_method, memory_efficient=memory_efficient)
    
    # Background subtraction (helps match APP's dark background)
    if background_subtraction:
        print("Performing background subtraction...")
        from scipy import ndimage
        # Use a large kernel to estimate background
        bg_kernel_size = min(stacked_image.shape) // 10
        bg_kernel_size = max(bg_kernel_size, 31)  # Ensure it's at least 31 pixels
        bg_kernel_size = bg_kernel_size + (1 - bg_kernel_size % 2)  # Ensure odd size
        print(f"Using background kernel size: {bg_kernel_size}")
        
        # Estimate background using median filter
        background = ndimage.median_filter(stacked_image, size=bg_kernel_size)
        
        # Subtract background with floor at zero
        stacked_image = np.maximum(stacked_image - background, 0)
        print("Background subtraction complete")
    
    # Apply contrast stretch to enhance visibility (similar to what APP does)
    if stretch_contrast:
        print("Applying contrast stretch...")
        # Calculate histogram
        hist, bin_edges = np.histogram(stacked_image.flatten(), bins=1000)
        cumulative_hist = np.cumsum(hist)
        
        # Find black and white point (similar to APP's default settings)
        # Black point at 0.1% of total pixels
        black_threshold = 0.001 * cumulative_hist[-1]
        black_idx = np.argmax(cumulative_hist > black_threshold)
        black_level = bin_edges[black_idx]
        
        # White point at 99.9% of pixels
        white_threshold = 0.999 * cumulative_hist[-1]
        white_idx = np.argmax(cumulative_hist > white_threshold)
        white_level = bin_edges[white_idx]
        
        print(f"Stretch levels: black={black_level:.6f}, white={white_level:.6f}")
        
        # Apply the stretch with clipping
        stacked_image = np.clip((stacked_image - black_level) / (white_level - black_level), 0, 1)
        
        # Convert back to original data type range
        if issubclass(transformed_images[0].dtype.type, np.integer):
            max_val = np.iinfo(transformed_images[0].dtype).max
            stacked_image = (stacked_image * max_val).astype(transformed_images[0].dtype)
    
    # Normalize to 0-1 range if requested
    elif normalize_output:
        print("Normalizing output to 0-1 range")
        min_val = np.min(stacked_image)
        max_val = np.max(stacked_image)
        if max_val > min_val:
            stacked_image = (stacked_image - min_val) / (max_val - min_val)
    
    # Save the result
    new_hdu = fits.PrimaryHDU(data=stacked_image, header=reference_header)
    
    # Add headers
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
    plt.close()
    
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
        f.write(f"Background subtraction: {background_subtraction}\n")
        f.write(f"Contrast stretching: {stretch_contrast}\n")
        f.write(f"Normalized output: {normalize_output}\n\n")
        
        f.write(f"Processed files ({len(processed_files)}):\n")
        for filename in processed_files:
            f.write(f"- {filename}\n")
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.1f} seconds")
    print(f"Saved stacked image to {output_file}")
    print(f"Saved preview image to {preview_file}")
    print(f"Saved processing information to {info_file}")

def main():
    parser = argparse.ArgumentParser(description='Stack FITS images using only files with valid transformation matrices')
    parser.add_argument('fits_directory', help='Directory containing FITS files')
    parser.add_argument('--output', '-o', default='stacked.fits', help='Output file path')
    parser.add_argument('--method', '-m', choices=['mean', 'median', 'sum'], default='median',
                       help='Stacking method (default: median)')
    parser.add_argument('--quality', '-q', type=float, default=0.4,
                       help='Quality threshold for frame roundness (default: 0.4)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--interpolation', '-i', type=int, choices=[1, 3, 5], default=1,
                      help='Interpolation order: 1=linear, 3=cubic, 5=lanczos (default: 1)')
    parser.add_argument('--no-rotation', action='store_true',
                      help='Disable 180-degree rotation correction')
    parser.add_argument('--margin', type=int, default=15,
                      help='Margin percentage for output padding (default: 15)')
    parser.add_argument('--normalize', action='store_true',
                      help='Normalize output to 0-1 range')
    parser.add_argument('--save-transformed', action='store_true',
                      help='Save individual transformed frames')
    parser.add_argument('--no-stretch', action='store_true',
                      help='Disable contrast stretching')
    parser.add_argument('--no-bg-subtraction', action='store_true',
                      help='Disable background subtraction')
    parser.add_argument('--memory-efficient', action='store_true',
                      help='Process files in a memory-efficient way')
    
    args = parser.parse_args()
    
    # Process the FITS files
    process_fits_files(
        args.fits_directory, 
        args.output,
        stacking_method=args.method, 
        quality_threshold=args.quality,
        max_files=args.max_files,
        interpolation_order=args.interpolation,
        rotate_180=not args.no_rotation,
        margin_percent=args.margin,
        normalize_output=args.normalize,
        save_transformed=args.save_transformed,
        stretch_contrast=not args.no_stretch,
        background_subtraction=not args.no_bg_subtraction,
        memory_efficient=args.memory_efficient
    )

if __name__ == "__main__":
    main()
