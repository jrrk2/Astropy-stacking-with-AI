#!/usr/bin/env python3
"""
FITS Matrix Stacker - Part 2: Main stacking function

This file contains the main stacking function for the FITS Matrix/WCS Stacker.
It's separated from part 1 to handle longer code sections.
"""

import os
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def stack_images(fits_files, reference_file=None, output_dir=None, interpolation_order=1, 
                 padding_percent=10, rejection_threshold=None, normalize=True, weight_by_snr=False,
                 use_wcs=False, debug_mode=False):
    """
    Stack FITS images using their transformation matrices or WCS information
    
    Args:
        fits_files: List of FITS files to stack
        reference_file: Reference file (default: first file in list)
        output_dir: Directory to save output (default: current directory)
        interpolation_order: Interpolation order for transformation (default: 1)
        padding_percent: Extra padding to add to output image (default: 10%)
        rejection_threshold: Sigma threshold for pixel rejection (default: None)
        normalize: Whether to normalize images before stacking (default: True)
        weight_by_snr: Whether to weight images by their SNR (default: False)
        use_wcs: Use WCS instead of transformation matrices (default: False)
        debug_mode: Save intermediate transformation results for debugging (default: False)
        
    Returns:
        stacked_data: The stacked image data
        output_header: The output FITS header
    """
    # Import needed functions in a way that avoids circular imports
    from fits_matrix_stacker_part1 import get_matrix_and_wcs_from_fits, apply_transformation, apply_wcs_transformation
    
    if len(fits_files) < 2:
        print("Need at least 2 FITS files to stack")
        return None, None
    
    start_time = time.time()
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if debug_mode:
            debug_dir = os.path.join(output_dir, 'debug')
            os.makedirs(debug_dir, exist_ok=True)
    
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
    
    # Get reference matrix and WCS
    ref_matrix, ref_wcs = get_matrix_and_wcs_from_fits(reference_file)
    
    # Check if we have what we need based on stacking method
    if use_wcs and ref_wcs is None:
        print(f"No valid WCS found in reference file. Cannot use WCS stacking method.")
        if ref_matrix is not None:
            print("Falling back to transformation matrix method.")
            use_wcs = False
        else:
            print("No transformation matrix found either. Cannot proceed.")
            return None, None
    elif not use_wcs and ref_matrix is None:
        print(f"No transformation matrix found in reference file.")
        if ref_wcs is not None:
            print("Falling back to WCS method.")
            use_wcs = True
        else:
            print("No WCS found either. Cannot proceed.")
            return None, None
    
    # If we're still okay, load reference image
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
    print(f"Using {'WCS' if use_wcs else 'transformation matrix'} method for alignment")
    
    # Initialize arrays for stacking
    stack_sum = np.zeros(output_shape, dtype=np.float32)
    weight_sum = np.zeros(output_shape, dtype=np.float32)
    
    if rejection_threshold is not None:
        # For sigma rejection, we need to track all transformed images
        all_transformed = np.zeros((len(fits_files), padded_h, padded_w), dtype=np.float32)
        all_weights = np.zeros((len(fits_files), padded_h, padded_w), dtype=np.float32)
    
    # Create identity matrix for the reference (for matrix method)
    identity = np.eye(3)
    
    # Add translation to center the reference in padded output (for matrix method)
    center_matrix = np.array([
        [1, 0, margin_w],
        [0, 1, margin_h],
        [0, 0, 1]
    ])
    
    # Create target WCS for the reference centered in the padded output (for WCS method)
    if use_wcs and ref_wcs is not None:
        target_wcs = ref_wcs.deepcopy()
        # Update reference pixel to center the image in the padded output
        target_wcs.wcs.crpix[0] += margin_w
        target_wcs.wcs.crpix[1] += margin_h
    
    # Process each file
    print(f"Transforming and stacking {len(fits_files)} images...")
    successful_files = 0
    for i, fits_file in enumerate(tqdm(fits_files)):
        try:
            # Load the image
            with fits.open(fits_file) as hdul:
                image_data = hdul[0].data
                if len(image_data.shape) > 2:
                    image_data = image_data[0]
                
                # For the first image (reference), just center it in the output
                if i == 0:
                    if use_wcs:
                        # For WCS, use the reference WCS to transform to the target WCS (just centering)
                        transformed = apply_wcs_transformation(
                            image_data, target_wcs, ref_wcs,
                            output_shape=output_shape,
                            order=interpolation_order
                        )
                    else:
                        # For matrix method, apply the centering matrix
                        transformed = apply_transformation(
                            image_data, center_matrix, 
                            output_shape=output_shape,
                            order=interpolation_order
                        )
                else:
                    # Get the file's matrix and WCS
                    file_matrix, file_wcs = get_matrix_and_wcs_from_fits(fits_file)
                    
                    if use_wcs:
                        # Check if we have valid WCS
                        if file_wcs is None:
                            print(f"Skipping {os.path.basename(fits_file)} - no valid WCS")
                            continue
                        
                        # Apply WCS transformation
                        transformed = apply_wcs_transformation(
                            image_data, target_wcs, file_wcs,
                            output_shape=output_shape,
                            order=interpolation_order
                        )
                    else:
                        # Check if we have valid matrix
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
                            image_data, matrix, 
                            output_shape=output_shape,
                            order=interpolation_order
                        )
                
                # Save debug image if requested
                if debug_mode and output_dir:
                    debug_file = os.path.join(debug_dir, f'transformed_{i:03d}.png')
                    plt.figure(figsize=(10, 8))
                    zscale = ZScaleInterval()
                    vmin, vmax = zscale.get_limits(transformed)
                    plt.imshow(transformed, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                    plt.colorbar(label='Pixel Value')
                    plt.title(f'Transformed {os.path.basename(fits_file)}')
                    plt.tight_layout()
                    plt.savefig(debug_file, dpi=150)
                    plt.close()
                
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
                
                successful_files += 1
        
        except Exception as e:
            print(f"Error processing {fits_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Successfully processed {successful_files} out of {len(fits_files)} files")
    
    # Perform sigma rejection if requested
    if rejection_threshold is not None and successful_files > 2:
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
    
    elapsed_time = time.time() - start_time
    print(f"Stacking completed in {elapsed_time:.1f} seconds")
    
    # Add stacking info to header
    ref_header['STACKED'] = True
    ref_header['NUMSTACK'] = successful_files
    ref_header['STACKREF'] = os.path.basename(reference_file)
    ref_header['STACKMTD'] = 'WCS' if use_wcs else 'MATRIX'
    if rejection_threshold is not None:
        ref_header['SIGCLIP'] = rejection_threshold
    
    if output_dir:
        # Determine filename based on method
        method_str = 'wcs' if use_wcs else 'matrix'
        output_file = os.path.join(output_dir, f'stacked_{method_str}.fits')
        
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
        plt.title(f'Stacked Image ({successful_files} frames) - {method_str.upper()} method')
        plt.tight_layout()
        preview_file = os.path.join(output_dir, f'stacked_{method_str}_preview.png')
        plt.savefig(preview_file, dpi=300)
        plt.close()
        print(f"Saved preview to {preview_file}")
    
    return stacked_data, ref_header