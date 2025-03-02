#!/usr/bin/env python3
"""
Improved AstroPy-based FITS image stacking script based on Stellina telescope log data.
This script uses cross-correlation and FFT-based methods to verify and refine
image alignment before stacking.
"""

import os
import json
import re
import glob
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
import matplotlib.pyplot as plt
from skimage import transform, registration, feature, filters, exposure
from scipy import signal, ndimage
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="skimage")

def parse_log_file(log_file_path):
    """
    Extract transformation matrices and other relevant information from the Stellina log file
    """
    print(f"Parsing log file: {log_file_path}")
    
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    # Parse the JSON content from the log file
    try:
        log_data = json.loads(log_content)
    except json.JSONDecodeError:
        # If the file doesn't start with valid JSON, try to extract JSON from it
        json_pattern = r'(\{.*\})'
        match = re.search(json_pattern, log_content, re.DOTALL)
        if match:
            try:
                log_data = json.loads(match.group(1))
            except json.JSONDecodeError:
                print("Could not parse JSON from log file")
                return []
        else:
            print("Could not find JSON content in log file")
            return []
    
    # Extract the stacking data
    transformations = []
    
    # Navigate through the 'corrections' entries in the log to find the stacking information
    if 'data' in log_data and 'corrections' in log_data['data']:
        for correction in log_data['data']['corrections']:
            if 'postCorrections' in correction:
                for post_correction in correction['postCorrections']:
                    if 'acqFile' in post_correction and 'stackingData' in post_correction['acqFile']:
                        stacking_data = post_correction['acqFile']['stackingData']
                        
                        # Check if this frame was successfully stacked
                        if stacking_data.get('error') is None and 'liveRegistrationResult' in stacking_data:
                            reg_result = stacking_data['liveRegistrationResult']
                            
                            # Extract the frame index and transformation matrix
                            if 'matrix' in reg_result and 'idx' in reg_result:
                                idx = reg_result['idx']
                                matrix = reg_result['matrix']
                                roundness = reg_result.get('roundness', 0)
                                stars_used = reg_result.get('starsUsed', 0)
                                
                                # Convert the matrix to proper numpy format (3x3)
                                transform_matrix = np.array([
                                    [matrix[0], matrix[1], matrix[2]],
                                    [matrix[3], matrix[4], matrix[5]],
                                    [matrix[6], matrix[7], matrix[8]]
                                ])
                                
                                frame_info = {
                                    'idx': idx,
                                    'matrix': transform_matrix,
                                    'roundness': roundness,
                                    'stars_used': stars_used,
                                    'file_index': post_correction['acqFile'].get('index', -1),
                                    'file_path': post_correction['acqFile'].get('path', '')
                                }
                                
                                transformations.append(frame_info)
    
    print(f"Extracted {len(transformations)} frame transformations")
    return transformations

def preprocess_image(image, dark_frame=None):
    """
    Preprocess an image: subtract dark frame, normalize, and enhance features
    """
    # Convert to float for processing
    image_float = image.astype(np.float32)
    
    # Subtract dark frame if provided
    if dark_frame is not None:
        if dark_frame.shape == image_float.shape:
            image_float = np.maximum(image_float - dark_frame, 0)
    
    # Basic normalization (0-1 scale)
    if np.max(image_float) > 0:
        norm_image = image_float / np.max(image_float)
    else:
        norm_image = image_float
        
    # Apply Gaussian filter to reduce noise
    smoothed = filters.gaussian(norm_image, sigma=1.0)
    
    # Enhance features
    enhanced = exposure.equalize_adapthist(smoothed, clip_limit=0.03)
    
    return enhanced, image_float

def detect_stars(image, threshold_sigma=3.0, min_size=5):
    """
    Detect stars in an image using a simple threshold-based method
    """
    # Apply a median filter to reduce noise
    filtered = ndimage.median_filter(image, size=3)
    
    # Calculate threshold based on image statistics
    mean = np.mean(filtered)
    std = np.std(filtered)
    threshold = mean + threshold_sigma * std
    
    # Apply threshold
    binary = filtered > threshold
    
    # Remove small objects
    labeled, num_features = ndimage.label(binary)
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    mask = np.zeros_like(binary, dtype=bool)
    for i, size in enumerate(sizes):
        if size >= min_size:
            mask = mask | (labeled == i + 1)
    
    # Get star coordinates
    coords = np.column_stack(np.where(mask))
    
    return coords, mask

def estimate_transformation_from_stars(src_stars, dst_stars, method='ransac'):
    """
    Estimate transformation matrix from star coordinates
    """
    if len(src_stars) < 3 or len(dst_stars) < 3:
        return None
    
    # Use RANSAC to estimate the transformation
    model = transform.AffineTransform()
    if method == 'ransac':
        # Limit the number of stars to make RANSAC more efficient
        max_stars = min(100, len(src_stars), len(dst_stars))
        src_subset = src_stars[:max_stars]
        dst_subset = dst_stars[:max_stars]
        
        try:
            model_robust, inliers = transform.ransac(
                (src_subset, dst_subset),
                transform.AffineTransform,
                min_samples=3,
                residual_threshold=2,
                max_trials=100
            )
            if model_robust is not None:
                return model_robust
        except:
            pass
    
    # Fallback to direct estimation
    try:
        model.estimate(src_stars, dst_stars)
        return model
    except:
        return None

def phase_cross_correlation(image1, image2):
    """
    Use phase cross-correlation to estimate translation between images
    """
    # Apply window to reduce edge effects
    h, w = image1.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    
    # Apply window and compute cross-correlation
    shift, error, diffphase = registration.phase_cross_correlation(
        image1 * window, 
        image2 * window,
        upsample_factor=10
    )
    
    return shift

def apply_transformation(image, matrix):
    """
    Apply a 3x3 transformation matrix to an image
    """
    # Create the affine transformation from the matrix
    tform = transform.AffineTransform(matrix=matrix)
    
    # Apply the transformation
    transformed = transform.warp(image, inverse_map=tform.inverse, preserve_range=True)
    
    return transformed.astype(image.dtype)

def verify_alignment(image1, image2, threshold=0.3):
    """
    Verify image alignment using normalized cross-correlation
    """
    # Calculate normalized cross-correlation
    corr = signal.correlate2d(
        image1, image2, mode='same', boundary='symm'
    )
    
    # Normalize correlation
    norm_factor = np.sqrt(np.sum(image1**2) * np.sum(image2**2))
    if norm_factor > 0:
        corr = corr / norm_factor
    
    # Calculate maximum correlation
    max_corr = np.max(corr)
    
    return max_corr > threshold, max_corr

def refine_alignment_correlation(base_image, moving_image, initial_matrix=None):
    """
    Refine image alignment using phase cross-correlation
    """
    # Apply initial transformation if provided
    if initial_matrix is not None:
        transformed = apply_transformation(moving_image, initial_matrix)
    else:
        transformed = moving_image.copy()
    
    # Preprocess images for correlation (normalize and enhance features)
    base_processed, _ = preprocess_image(base_image)
    moving_processed, _ = preprocess_image(transformed)
    
    # Calculate shift using phase cross-correlation
    shift = phase_cross_correlation(base_processed, moving_processed)
    
    # Create transformation matrix for the shift
    shift_matrix = np.array([
        [1, 0, -shift[1]],  # Note the reversal of x,y coordinates
        [0, 1, -shift[0]],
        [0, 0, 1]
    ])
    
    # Combine with initial matrix if provided
    if initial_matrix is not None:
        # Matrix multiplication to combine transformations
        combined_matrix = np.dot(shift_matrix, initial_matrix)
        return combined_matrix
    else:
        return shift_matrix

def stack_images(images, method='mean'):
    """
    Stack multiple images together
    """
    if not images:
        return None
        
    # Convert to float64 for processing
    float_images = [img.astype(np.float64) for img in images]
    
    # Stack the images
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
                      correlation_threshold=0.3, use_log_transforms=True,
                      refine_alignment=True):
    """
    Process FITS files according to the extracted transformations
    """
    print(f"Processing FITS files from: {fits_directory}")
    
    # Sort the transformations by their idx
    transformations.sort(key=lambda x: x['idx'])
    
    # Find all FITS files in the directory
    fits_files = sorted(glob.glob(os.path.join(fits_directory, "*.fits")))
    
    if not fits_files:
        print(f"No FITS files found in {fits_directory}")
        return
    
    print(f"Found {len(fits_files)} FITS files")
    
    # Load the master dark frame if provided
    master_dark = None
    if dark_frame_path:
        try:
            with fits.open(dark_frame_path) as hdul:
                master_dark = hdul[0].data
            print(f"Loaded master dark frame from {dark_frame_path}")
        except Exception as e:
            print(f"Error loading master dark frame: {e}")
            print("Continuing without dark subtraction")
    
    # Read the first FITS file to get the image shape
    with fits.open(fits_files[0]) as hdul:
        reference_data = hdul[0].data
        reference_header = hdul[0].header
    
    # Initialize an empty reference image for alignment verification
    reference_image = None
    
    # Initialize list to hold transformed images
    transformed_images = []
    correlation_scores = []
    
    # Keep track of processed files for verification
    processed_files = []
    
    # Process each transformation
    for transform_info in tqdm(transformations, desc="Processing frames"):
        # Skip frames with low quality metrics
        if transform_info['roundness'] < quality_threshold:
            print(f"Skipping frame idx={transform_info['idx']} with roundness={transform_info['roundness']}")
            continue
            
        # Find the corresponding FITS file
        file_index = transform_info['file_index']
        
        # Try to find the file by index in the filename
        matching_files = [f for f in fits_files if f"_{file_index:04d}" in f or f"_{file_index}" in f]
        
        if not matching_files and len(fits_files) > file_index:
            # If we can't find by index in the name, try using the position in the sorted list
            file_to_use = fits_files[file_index]
        elif matching_files:
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
        
        # Apply the transformation from the log file if requested
        if use_log_transforms:
            # Get the transformation matrix from the log
            matrix = transform_info['matrix']
            transformed = apply_transformation(image_data, matrix)
        else:
            transformed = image_data
        
        # Set the first valid frame as reference if not yet set
        if reference_image is None:
            reference_image = transformed.copy()
            transformed_images.append(transformed)
            correlation_scores.append(1.0)  # Perfect correlation with itself
            processed_files.append(file_to_use)
            continue
        
        # Verify alignment using cross-correlation
        _, processed_transformed = preprocess_image(transformed)
        _, processed_reference = preprocess_image(reference_image)
        
        is_aligned, correlation = verify_alignment(
            processed_reference, processed_transformed, threshold=correlation_threshold
        )
        
        # Refine alignment if needed and possible
        if refine_alignment:
            matrix = refine_alignment_correlation(
                reference_image, transformed, 
                initial_matrix=transform_info['matrix'] if use_log_transforms else None
            )
            transformed = apply_transformation(image_data, matrix)
            
            # Check alignment again after refinement
            _, processed_transformed = preprocess_image(transformed)
            is_aligned, correlation = verify_alignment(
                processed_reference, processed_transformed, threshold=correlation_threshold
            )
        
        # Only add well-aligned images
        if is_aligned:
            transformed_images.append(transformed)
            correlation_scores.append(correlation)
            processed_files.append(file_to_use)
        else:
            print(f"Skipping misaligned frame: {file_to_use}, correlation={correlation:.3f}")
        
    if not transformed_images:
        print("No images could be transformed. Check your files and transformation data.")
        return
        
    # Stack the transformed images
    print(f"Stacking {len(transformed_images)} images using {stacking_method} method")
    stacked_image = stack_images(transformed_images, method=stacking_method)
    
    # Save the result
    new_hdu = fits.PrimaryHDU(data=stacked_image, header=reference_header)
    new_hdu.header['HISTORY'] = f'Stacked {len(transformed_images)} images using {stacking_method} method'
    new_hdu.writeto(output_file, overwrite=True)
    
    print(f"Saved stacked image to {output_file}")
    
    # Display a preview using asinh scaling for better visualization of astronomical details
    plt.figure(figsize=(10, 10))
    norm = ImageNormalize(stacked_image, stretch=AsinhStretch())
    plt.imshow(stacked_image, cmap='gray', norm=norm, origin='lower')
    plt.colorbar()
    plt.title(f'Stacked Image ({len(transformed_images)} frames)')
    plt.grid(True, alpha=0.3, color='white')
    preview_file = output_file.replace('.fits', '_preview.png')
    plt.savefig(preview_file)
    print(f"Saved preview image to {preview_file}")
    
    # Generate diagnostic plot showing correlations
    plt.figure(figsize=(10, 6))
    plt.plot(correlation_scores, 'o-')
    plt.axhline(y=correlation_threshold, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Frame Number')
    plt.ylabel('Correlation Score')
    plt.title('Alignment Correlation Scores')
    plt.grid(True)
    corr_file = output_file.replace('.fits', '_correlations.png')
    plt.savefig(corr_file)
    
    # Save correlation data
    with open(output_file.replace('.fits', '_correlations.csv'), 'w') as f:
        f.write("File,Correlation\n")
        for idx, (file_path, corr) in enumerate(zip(processed_files, correlation_scores)):
            f.write(f"{os.path.basename(file_path)},{corr:.6f}\n")
    
    # Generate a diagnostic image showing a sample of the frames
    sample_size = min(4, len(transformed_images))
    if sample_size > 0:
        fig, axes = plt.subplots(1, sample_size, figsize=(15, 4))
        if sample_size == 1:
            axes = [axes]
        
        sample_indices = np.linspace(0, len(transformed_images)-1, sample_size, dtype=int)
        
        for i, idx in enumerate(sample_indices):
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(transformed_images[idx])
            axes[i].imshow(transformed_images[idx], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
            axes[i].set_title(f"Frame {idx}\nCorr: {correlation_scores[idx]:.3f}")
            axes[i].grid(True, alpha=0.3, color='white')
        
        plt.tight_layout()
        frames_file = output_file.replace('.fits', '_frames.png')
        plt.savefig(frames_file)
        print(f"Saved frame samples to {frames_file}")

def main():
    parser = argparse.ArgumentParser(description='Stack FITS images based on Stellina log data')
    parser.add_argument('log_file', help='Path to the Stellina log file')
    parser.add_argument('fits_directory', help='Directory containing FITS files')
    parser.add_argument('--output', '-o', default='stacked.fits', help='Output file path')
    parser.add_argument('--dark', '-d', help='Path to master dark frame')
    parser.add_argument('--method', '-m', choices=['mean', 'median', 'sum'], default='mean',
                       help='Stacking method (default: mean)')
    parser.add_argument('--quality', '-q', type=float, default=0.4,
                       help='Quality threshold for frame roundness (default: 0.4)')
    parser.add_argument('--correlation', '-c', type=float, default=0.3,
                       help='Correlation threshold for alignment verification (default: 0.3)')
    parser.add_argument('--no-log-transforms', action='store_true',
                       help='Skip using transformations from log file')
    parser.add_argument('--no-refine', action='store_true',
                       help='Skip alignment refinement')
    
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
        correlation_threshold=args.correlation,
        use_log_transforms=not args.no_log_transforms,
        refine_alignment=not args.no_refine
    )

if __name__ == "__main__":
    main()
