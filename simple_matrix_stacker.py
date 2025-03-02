#!/usr/bin/env python3
"""
Simple Matrix Stacker - A focused approach to stack astronomical images using transformation matrices

This script uses a direct approach to apply transformation matrices from FITS headers,
focusing on accurate star shapes.
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

def get_transformation_matrix(fits_file):
    """Extract transformation matrix from FITS header"""
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            
            # Check for transformation matrix format
            if all(f'TRNSFM{i+1}{j+1}' in header for i in range(3) for j in range(3)):
                key_prefix = 'TRNSFM'
            elif all(f'TRNSFRM{i+1}{j+1}' in header for i in range(3) for j in range(3)):
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

def direct_transform(image, matrix, output_shape=None, order=3):
    """Apply transformation directly using the matrix, with special handling for astronomical images"""
    h, w = image.shape
    
    # If no output shape is provided, use the input shape
    if output_shape is None:
        output_shape = (h, w)
    
    # Create the transformation
    tform = transform.ProjectiveTransform(matrix=matrix)
    
    # Apply the transformation with specified output shape and interpolation order
    # Using reflect mode for better edge behavior
    transformed = transform.warp(image, inverse_map=tform.inverse, 
                                output_shape=output_shape,
                                order=order,
                                mode='constant', 
                                cval=0,
                                preserve_range=True)
    
    return transformed.astype(image.dtype)

def stack_images(fits_files, reference_file=None, output_dir="stacked_output", 
                 sigma_clip=3.0, padding=10, interpolation_order=3):
    """
    Simple, direct stacking of FITS images using transformation matrices
    """
    if len(fits_files) < 2:
        print("Need at least 2 FITS files to stack")
        return None
    
    # Create output directory
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
    ref_matrix = get_transformation_matrix(reference_file)
    if ref_matrix is None:
        print(f"No transformation matrix found in reference file. Cannot proceed.")
        return None
    
    # Load reference image
    with fits.open(reference_file) as hdul:
        ref_data = hdul[0].data
        if len(ref_data.shape) > 2:
            print(f"Reference has {len(ref_data.shape)} dimensions, using first channel")
            ref_data = ref_data[0]
        ref_header = hdul[0].header.copy()
    
    # Calculate padded dimensions
    h, w = ref_data.shape
    margin_h = int(h * padding / 100)
    margin_w = int(w * padding / 100)
    padded_h = h + 2 * margin_h
    padded_w = w + 2 * margin_w
    output_shape = (padded_h, padded_w)
    
    print(f"Original dimensions: {w}x{h}")
    print(f"Padded output dimensions: {padded_w}x{padded_h} (with {padding}% margin)")
    
    # Create arrays to store transformed images and masks
    transformed_images = []
    transformed_masks = []
    
    # Add translation to center the reference in padded output
    center_matrix = np.array([
        [1, 0, margin_w],
        [0, 1, margin_h],
        [0, 0, 1]
    ])
    
    # Process each file
    print(f"Transforming {len(fits_files)} images...")
    successful_files = 0
    
    for fits_file in tqdm(fits_files):
        try:
            # Load the image
            with fits.open(fits_file) as hdul:
                image_data = hdul[0].data
                if len(image_data.shape) > 2:
                    image_data = image_data[0]
                
                # Get the file's transformation matrix
                file_matrix = get_transformation_matrix(fits_file)
                if file_matrix is None:
                    print(f"Skipping {os.path.basename(fits_file)} - no transformation matrix")
                    continue
                
                # Calculate transformation directly to the reference frame
                if fits_file == reference_file:
                    # For reference, just apply centering
                    matrix = center_matrix
                else:
                    # Calculate: reference_frame <- world_frame <- image_frame
                    # This is: center_matrix @ inv(ref_matrix) @ file_matrix
                    ref_matrix_inv = np.linalg.inv(ref_matrix)
                    relative_matrix = np.dot(ref_matrix_inv, file_matrix)
                    matrix = np.dot(center_matrix, relative_matrix)
                
                # Apply the transformation
                transformed = direct_transform(
                    image_data, matrix, output_shape=output_shape,
                    order=interpolation_order
                )
                
                # Create a mask for valid pixels (non-zero)
                mask = (transformed > 0).astype(np.float32)
                
                # Store the transformed image and mask
                transformed_images.append(transformed)
                transformed_masks.append(mask)
                
                successful_files += 1
                
                # Save individual transformed image for inspection
                if successful_files <= 5 or successful_files % 10 == 0:  # First 5 and every 10th
                    plt.figure(figsize=(10, 8))
                    zscale = ZScaleInterval()
                    vmin, vmax = zscale.get_limits(transformed)
                    plt.imshow(transformed, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                    plt.colorbar(label='Pixel Value')
                    plt.title(f'Transformed {os.path.basename(fits_file)}')
                    plt.tight_layout()
                    debug_file = os.path.join(output_dir, f'transformed_{successful_files:03d}.png')
                    plt.savefig(debug_file, dpi=150)
                    plt.close()
        
        except Exception as e:
            print(f"Error processing {fits_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if successful_files < 2:
        print("Not enough images successfully processed for stacking")
        return None
    
    print(f"Successfully transformed {successful_files} images, now stacking...")
    
    # Convert to numpy arrays
    transformed_images = np.array(transformed_images)
    transformed_masks = np.array(transformed_masks)
    
    # Perform sigma clipping stack
    if sigma_clip > 0 and successful_files > 2:
        print(f"Applying {sigma_clip}-sigma clipping...")
        
        # Prepare output arrays
        stacked_data = np.zeros(output_shape, dtype=np.float32)
        weight_sum = np.zeros(output_shape, dtype=np.float32)
        
        # Process each pixel
        for y in tqdm(range(padded_h)):
            for x in range(padded_w):
                # Get all pixel values at this position
                values = transformed_images[:, y, x]
                masks = transformed_masks[:, y, x]
                
                # Filter for valid pixels
                valid = masks > 0
                if np.sum(valid) > 0:
                    valid_values = values[valid]
                    
                    if len(valid_values) > 2:
                        # Calculate mean and standard deviation
                        mean = np.mean(valid_values)
                        std = np.std(valid_values)
                        
                        # Identify outliers
                        if std > 0:
                            good_values = valid_values[np.abs(valid_values - mean) <= sigma_clip * std]
                            if len(good_values) > 0:
                                stacked_data[y, x] = np.mean(good_values)
                                weight_sum[y, x] = len(good_values)
                        else:
                            # No variance, use all valid values
                            stacked_data[y, x] = mean
                            weight_sum[y, x] = len(valid_values)
                    else:
                        # Not enough for statistics, use average
                        stacked_data[y, x] = np.mean(valid_values)
                        weight_sum[y, x] = len(valid_values)
    else:
        # Simple mean stack (no rejection)
        print("Performing simple mean stack...")
        stacked_data = np.sum(transformed_images, axis=0) / np.maximum(np.sum(transformed_masks, axis=0), 1)
        weight_sum = np.sum(transformed_masks, axis=0)
    
    # Set pixels with no contribution to zero
    stacked_data[weight_sum == 0] = 0
    
    # Save the stacked image
    output_file = os.path.join(output_dir, 'stacked.fits')
    
    # Update header
    ref_header['STACKED'] = True
    ref_header['NUMSTACK'] = successful_files
    ref_header['STACKREF'] = os.path.basename(reference_file)
    if sigma_clip > 0:
        ref_header['SIGCLIP'] = sigma_clip
    
    # Create a new HDU with the stacked data
    hdu = fits.PrimaryHDU(stacked_data.astype(np.float32), header=ref_header)
    hdu.writeto(output_file, overwrite=True)
    print(f"Saved stacked image to {output_file}")
    
    # Save a preview image
    plt.figure(figsize=(10, 8))
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(stacked_data)
    plt.imshow(stacked_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Pixel Value')
    plt.title(f'Stacked Image ({successful_files} frames)')
    plt.tight_layout()
    preview_file = os.path.join(output_dir, 'stacked_preview.png')
    plt.savefig(preview_file, dpi=300)
    plt.close()
    print(f"Saved preview to {preview_file}")
    
    # Save a weight map for inspection
    plt.figure(figsize=(10, 8))
    plt.imshow(weight_sum, cmap='viridis', origin='lower')
    plt.colorbar(label='Number of Contributing Frames')
    plt.title('Weight Map (number of frames contributing to each pixel)')
    plt.tight_layout()
    weight_file = os.path.join(output_dir, 'weight_map.png')
    plt.savefig(weight_file, dpi=300)
    plt.close()
    
    return stacked_data

def main():
    parser = argparse.ArgumentParser(description='Simple, direct stacking of FITS images using transformation matrices')
    parser.add_argument('fits_directory', help='Directory containing FITS files or wildcard pattern')
    parser.add_argument('--output', '-o', default='stacked_output', help='Output directory')
    parser.add_argument('--reference', '-r', help='Reference file (default: first file)')
    parser.add_argument('--sigma-clip', '-s', type=float, default=3.0, help='Sigma threshold for pixel rejection (default: 3.0)')
    parser.add_argument('--padding', '-p', type=int, default=10, help='Padding percentage (default: 10%%)')
    parser.add_argument('--interpolation', '-i', type=int, choices=[0, 1, 3, 5], default=3,
                      help='Interpolation order: 0=nearest, 1=linear, 3=cubic (default: 3)')
    parser.add_argument('--pattern', default='*.fits', help='File pattern to match (default: *.fits)')
    
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
        sigma_clip=args.sigma_clip,
        padding=args.padding,
        interpolation_order=args.interpolation
    )

if __name__ == "__main__":
    main()
