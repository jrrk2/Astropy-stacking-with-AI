#!/usr/bin/env python3
"""
FITS Matrix Stacker - Stack astronomical images using transformation matrices

This script stacks FITS images by applying the transformation matrices stored 
in their headers, without requiring WCS information. It's designed for situations
where affine transformations provide better alignment than WCS-based methods.
"""

import os
import argparse
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from skimage import transform
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

def get_matrix_from_fits(fits_file):
    """Extract transformation matrix from FITS header"""
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            
            # Check for shortened version first (TRNSFM)
            has_short_transform = all(f'TRNSFM{i+1}{j+1}' in header 
                                 for i in range(3) for j in range(3))
                                 
            # Check for full version next (TRNSFRM)
            has_long_transform = all(f'TRNSFRM{i+1}{j+1}' in header 
                                for i in range(3) for j in range(3))
            
            if has_short_transform:
                key_prefix = 'TRNSFM'
            elif has_long_transform:
                key_prefix = 'TRNSFRM'
            else:
                return None
            
            # Extract the matrix
            matrix = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    matrix[i, j] = header[f'{key_prefix}{i+1}{j+1}']
            
            return matrix
            
    except Exception as e:
        print(f"Error reading matrix from {fits_file}: {e}")
        return None

def apply_transformation(image, matrix, output_shape=None, order=1):
    """Apply a transformation matrix to an image"""
    h, w = image.shape
    
    # If no output shape is provided, use the input shape
    if output_shape is None:
        output_shape = (h, w)
    
    # Create the transformation
    tform = transform.ProjectiveTransform(matrix=matrix)
    
    # Apply the transformation with specified output shape and interpolation order
    transformed = transform.warp(image, inverse_map=tform.inverse, 
                                output_shape=output_shape,
                                order=order,
                                mode='constant', 
                                cval=0,
                                preserve_range=True)
    
    return transformed.astype(image.dtype)

def stack_images(fits_files, reference_file=None, output_dir=None, interpolation_order=1, 
                 padding_percent=10, rejection_threshold=None, normalize=True, weight_by_snr=False):
    """
    Stack FITS images using their transformation matrices
    
    Args:
        fits_files: List of FITS files to stack
        reference_file: Reference file (default: first file in list)
        output_dir: Directory to save output (default: current directory)
        interpolation_order: Interpolation order for transformation (default: 1)
        padding_percent: Extra padding to add to output image (default: 10%)
        rejection_threshold: Sigma threshold for pixel rejection (default: None)
        normalize: Whether to normalize images before stacking (default: True)
        weight_by_snr: Whether to weight images by their SNR (default: False)
        
    Returns:
        stacked_data: The stacked image data
        output_header: The output FITS header
    """
    if len(fits_files) < 2:
        print("Need at least 2 FITS files to stack")
        return None, None
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set reference file if not specified
    if reference_file is None:
        reference_file = fits_files[0]
    elif reference_file not in fits_files:
        fits_files.insert(0, reference_file)
    else:
        # Move reference file to the front
        fits_files.remove(reference_file)
        fits_files.insert(0, reference_file)
    
    print(f"Using {os.path.basename(reference_file)} as reference")
    
    # Get reference matrix
    ref_matrix = get_matrix_from_fits(reference_file)
    if ref_matrix is None:
        print(f"No transformation matrix found in reference file. Trying another...")
        for f in fits_files[1:]:
            ref_matrix = get_matrix_from_fits(f)
            if ref_matrix is not None:
                reference_file = f
                # Move new reference to front
                fits_files.remove(reference_file)
                fits_files.insert(0, reference_file)
                print(f"Using {os.path.basename(reference_file)} as reference instead")
                break
                
        if ref_matrix is None:
            print("No transformation matrices found. Cannot proceed.")
            return None, None
    
    # Load reference image
    with fits.open(reference_file) as hdul:
        ref_data = hdul[0].data
        if len(ref_data.shape) > 2:
            print(f"Reference has {len(ref_data.shape)} dimensions, using first channel")
            ref_data = ref_data[0]
        ref_header = hdul[0].header.copy()
    
    # Calculate padded dimensions
    h, w = ref_data.shape
    margin_h = int(h * padding_percent / 100)
    margin_w = int(w * padding_percent / 100)
    padded_h = h + 2 * margin_h
    padded_w = w + 2 * margin_w
    output_shape = (padded_h, padded_w)
    
    print(f"Original dimensions: {w}x{h}")
    print(f"Padded output dimensions: {padded_w}x{padded_h} (with {padding_percent}% margin)")
    
    # Initialize arrays for stacking
    stack_sum = np.zeros(output_shape, dtype=np.float32)
    weight_sum = np.zeros(output_shape, dtype=np.float32)
    
    if rejection_threshold is not None:
        # For sigma rejection, we need to track all transformed images
        all_transformed = np.zeros((len(fits_files), padded_h, padded_w), dtype=np.float32)
        all_weights = np.zeros((len(fits_files), padded_h, padded_w), dtype=np.float32)
    
    # Create identity matrix for the reference
    identity = np.eye(3)
    
    # Add translation to center the reference in padded output
    center_matrix = np.array([
        [1, 0, margin_w],
        [0, 1, margin_h],
        [0, 0, 1]
    ])
    
    # Process each file
    print("Transforming and stacking images...")
    for i, fits_file in enumerate(tqdm(fits_files)):
        try:
            # Load the image
            with fits.open(fits_file) as hdul:
                image_data = hdul[0].data
                if len(image_data.shape) > 2:
                    image_data = image_data[0]
                
                # For the first image (reference), just center it in the output
                if i == 0:
                    matrix = center_matrix.copy()
                else:
                    # Get the file's transformation matrix
                    file_matrix = get_matrix_from_fits(fits_file)
                    if file_matrix is None:
                        print(f"Skipping {os.path.basename(fits_file)} - no transformation matrix")
                        continue
                    
                    # Calculate transformation relative to the reference
                    # inv(ref_matrix) @ file_matrix - transforms from file's frame to reference frame
                    ref_matrix_inv = np.linalg.inv(ref_matrix)
                    relative_matrix = np.dot(ref_matrix_inv, file_matrix)
                    
                    # Apply centering to the relative transformation
                    matrix = np.dot(center_matrix, relative_matrix)
                
                # Apply the transformation
                transformed = apply_transformation(
                    image_data, matrix, output_shape=output_shape,
                    order=interpolation_order
                )
                
                # Calculate weight (can be enhanced with more sophisticated methods)
                if weight_by_snr:
                    # Simple SNR estimate: mean / std where signal is above background
                    mask = transformed > np.median(transformed)
                    if np.sum(mask) > 0:
                        signal = np.mean(transformed[mask])
                        noise = np.std(transformed[~mask])
                        if noise > 0:
                            weight = signal / noise
                        else:
                            weight = 1.0
                    else:
                        weight = 1.0
                else:
                    weight = 1.0
                
                # Normalize if requested
                if normalize:
                    # Estimate background and scale
                    bg = np.median(transformed)
                    scale = 1.0
                    if bg > 0:
                        # Simple scaling to match reference background level
                        ref_bg = np.median(ref_data)
                        if ref_bg > 0:
                            scale = ref_bg / bg
                    
                    # Apply scaling
                    transformed = transformed * scale
                
                if rejection_threshold is not None:
                    # Store for later sigma rejection
                    all_transformed[i] = transformed
                    all_weights[i] = weight
                else:
                    # Add to the stack directly
                    stack_sum += transformed * weight
                    weight_sum += weight * (transformed > 0)  # Only count weights where pixels are valid
        
        except Exception as e:
            print(f"Error processing {fits_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Perform sigma rejection if requested
    if rejection_threshold is not None and len(fits_files) > 2:
        print("Performing sigma rejection...")
        
        # For each pixel position
        for y in tqdm(range(padded_h)):
            for x in range(padded_w):
                # Get values at this position from all images
                values = all_transformed[:, y, x]
                weights = all_weights[:, y, x]
                
                # Skip if all zeros
                if np.all(values == 0):
                    continue
                
                # Compute mean and std of non-zero values
                valid_mask = values > 0
                if np.sum(valid_mask) > 1:
                    valid_values = values[valid_mask]
                    valid_weights = weights[valid_mask]
                    
                    # Compute weighted mean and std
                    mean = np.average(valid_values, weights=valid_weights)
                    # Weighted standard deviation
                    variance = np.average((valid_values - mean)**2, weights=valid_weights)
                    std = np.sqrt(variance)
                    
                    # Reject outliers
                    if std > 0:
                        reject_mask = np.abs(valid_values - mean) > rejection_threshold * std
                        good_mask = ~reject_mask
                        
                        if np.sum(good_mask) > 0:
                            # Add only non-rejected pixels to the stack
                            stack_sum[y, x] = np.sum(valid_values[good_mask] * valid_weights[good_mask])
                            weight_sum[y, x] = np.sum(valid_weights[good_mask])
                else:
                    # If only one valid value, use it
                    stack_sum[y, x] = np.sum(values * weights)
                    weight_sum[y, x] = np.sum(weights * (values > 0))
    
    # Normalize by weights
    stacked_data = np.zeros_like(stack_sum)
    valid_mask = weight_sum > 0
    stacked_data[valid_mask] = stack_sum[valid_mask] / weight_sum[valid_mask]
    
    # Add stacking info to header
    ref_header['STACKED'] = True
    ref_header['NUMSTACK'] = len(fits_files)
    ref_header['STACKREF'] = os.path.basename(reference_file)
    
    if output_dir:
        # Save the stacked image
        output_file = os.path.join(output_dir, 'stacked.fits')
        
        # Create a new HDU with the stacked data
        hdu = fits.PrimaryHDU(stacked_data.astype(np.float32), header=ref_header)
        hdu.writeto(output_file, overwrite=True)
        print(f"Saved stacked image to {output_file}")
        
        # Save a preview image
        plt.figure(figsize=(10, 8))
        # Use ZScale for better visualization
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(stacked_data)
        plt.imshow(stacked_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Pixel Value')
        plt.title(f'Stacked Image ({len(fits_files)} frames)')
        plt.tight_layout()
        preview_file = os.path.join(output_dir, 'stacked_preview.png')
        plt.savefig(preview_file, dpi=300)
        plt.close()
        print(f"Saved preview to {preview_file}")
    
    return stacked_data, ref_header

def main():
    parser = argparse.ArgumentParser(description='Stack FITS images using transformation matrices')
    parser.add_argument('fits_directory', help='Directory containing FITS files or wildcard pattern')
    parser.add_argument('--output', '-o', default='stacked_output', help='Output directory')
    parser.add_argument('--reference', '-r', help='Reference file (default: first file)')
    parser.add_argument('--interpolation', '-i', type=int, choices=[0, 1, 3, 5], default=1,
                      help='Interpolation order: 0=nearest, 1=linear, 3=cubic (default: 1)')
    parser.add_argument('--padding', '-p', type=int, default=10, 
                      help='Padding percentage around output image (default: 10%%)')
    parser.add_argument('--sigma-clip', '-s', type=float, default=None,
                      help='Sigma threshold for pixel rejection (default: None)')
    parser.add_argument('--no-normalize', action='store_true',
                      help='Disable normalization of images before stacking')
    parser.add_argument('--weight-by-snr', action='store_true',
                      help='Weight images by estimated SNR')
    parser.add_argument('--pattern', default='*.fits',
                      help='File pattern to match (default: *.fits)')
    
    args = parser.parse_args()
    
    # Find all FITS files
    if os.path.isdir(args.fits_directory):
        fits_files = sorted(glob.glob(os.path.join(args.fits_directory, args.pattern)))
    else:
        # Assume it's a wildcard pattern
        fits_files = sorted(glob.glob(args.fits_directory))
    
    if not fits_files:
        print(f"No FITS files found matching the pattern")
        return
    
    print(f"Found {len(fits_files)} FITS files")
    
    # Stack the images
    stack_images(
        fits_files,
        reference_file=args.reference,
        output_dir=args.output,
        interpolation_order=args.interpolation,
        padding_percent=args.padding,
        rejection_threshold=args.sigma_clip,
        normalize=not args.no_normalize,
        weight_by_snr=args.weight_by_snr
    )

if __name__ == "__main__":
    main()
