#!/usr/bin/env python3
"""
FITS Matrix Stacker - Stack astronomical images using transformation matrices or WCS

This script stacks FITS images by applying either the transformation matrices stored 
in their headers or by using WCS information for alignment. This allows direct comparison
between matrix-based and WCS-based stacking approaches.
"""

import os
import argparse
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy import wcs
from skimage import transform
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import time

def get_matrix_and_wcs_from_fits(fits_file):
    """Extract transformation matrix and WCS info from FITS header"""
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            
            # Get transformation matrix
            matrix = None
            # Check for shortened version first (TRNSFM)
            has_short_transform = all(f'TRNSFM{i+1}{j+1}' in header 
                                 for i in range(3) for j in range(3))
                                 
            # Check for full version next (TRNSFRM)
            has_long_transform = all(f'TRNSFRM{i+1}{j+1}' in header 
                                for i in range(3) for j in range(3))
            
            if has_short_transform:
                key_prefix = 'TRNSFM'
                matrix = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        matrix[i, j] = header[f'{key_prefix}{i+1}{j+1}']
            elif has_long_transform:
                key_prefix = 'TRNSFRM'
                matrix = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        matrix[i, j] = header[f'{key_prefix}{i+1}{j+1}']
            
            # Get WCS information if available
            wcs_info = None
            if all(key in header for key in ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']):
                try:
                    wcs_info = wcs.WCS(header)
                except Exception as e:
                    print(f"Error creating WCS object from {fits_file}: {e}")
            
            return matrix, wcs_info
            
    except Exception as e:
        print(f"Error reading from {fits_file}: {e}")
        return None, None

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

def apply_wcs_transformation(image, target_wcs, source_wcs, output_shape=None, order=1):
    """
    Transform an image from source WCS to target WCS
    
    This function resamples the image from the source WCS frame to the target WCS frame
    using the specified interpolation order.
    """
    # If no output shape is provided, use the input shape
    if output_shape is None:
        output_shape = image.shape
    
    # Create a grid of pixel coordinates in the target frame
    y, x = np.mgrid[:output_shape[0], :output_shape[1]]
    
    # Convert target pixel coordinates to world coordinates
    ra, dec = target_wcs.wcs_pix2world(x, y, 0)
    
    # Convert world coordinates to source pixel coordinates
    source_x, source_y = source_wcs.wcs_world2pix(ra, dec, 0)
    
    # Create a coordinate transformation
    coords = np.array([source_y, source_x])
    
    # Apply the transformation with specified interpolation order
    transformed = transform.warp(image, coords, order=order, mode='constant', 
                                cval=0, preserve_range=True,
                                output_shape=output_shape)
    
    return transformed.astype(image.dtype)

# Import the stacking function from the second part
from fits_matrix_stacker_part2 import stack_images

def main():
    parser = argparse.ArgumentParser(description='Stack FITS images using transformation matrices or WCS')
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
    parser.add_argument('--use-wcs', action='store_true',
                      help='Use WCS instead of transformation matrices for alignment')
    parser.add_argument('--debug', action='store_true',
                      help='Save intermediate transformations for debugging')
    parser.add_argument('--compare', action='store_true',
                      help='Run both matrix and WCS methods and compare')
    
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
    
    if args.compare:
        # Run both methods for comparison
        print("\n=== Running with transformation matrix method ===")
        stack_images(
            fits_files,
            reference_file=args.reference,
            output_dir=args.output,
            interpolation_order=args.interpolation,
            padding_percent=args.padding,
            rejection_threshold=args.sigma_clip,
            normalize=not args.no_normalize,
            weight_by_snr=args.weight_by_snr,
            use_wcs=False,
            debug_mode=args.debug
        )
        
        print("\n=== Running with WCS method ===")
        stack_images(
            fits_files,
            reference_file=args.reference,
            output_dir=args.output,
            interpolation_order=args.interpolation,
            padding_percent=args.padding,
            rejection_threshold=args.sigma_clip,
            normalize=not args.no_normalize,
            weight_by_snr=args.weight_by_snr,
            use_wcs=True,
            debug_mode=args.debug
        )
        
        # Create comparison image
        try:
            matrix_file = os.path.join(args.output, 'stacked_matrix.fits')
            wcs_file = os.path.join(args.output, 'stacked_wcs.fits')
            
            if os.path.exists(matrix_file) and os.path.exists(wcs_file):
                with fits.open(matrix_file) as hdul_matrix, fits.open(wcs_file) as hdul_wcs:
                    matrix_data = hdul_matrix[0].data
                    wcs_data = hdul_wcs[0].data
                    
                    plt.figure(figsize=(15, 10))
                    
                    # Use ZScale for better visualization
                    zscale = ZScaleInterval()
                    
                    # Plot matrix result
                    plt.subplot(2, 2, 1)
                    vmin, vmax = zscale.get_limits(matrix_data)
                    plt.imshow(matrix_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                    plt.title('Matrix Method')
                    plt.colorbar(label='Pixel Value')
                    
                    # Plot WCS result
                    plt.subplot(2, 2, 2)
                    vmin, vmax = zscale.get_limits(wcs_data)
                    plt.imshow(wcs_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                    plt.title('WCS Method')
                    plt.colorbar(label='Pixel Value')
                    
                    # Plot difference
                    plt.subplot(2, 2, 3)
                    diff = matrix_data - wcs_data
                    vmin, vmax = np.percentile(diff[~np.isnan(diff)], [1, 99])
                    plt.imshow(diff, cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
                    plt.title('Difference (Matrix - WCS)')
                    plt.colorbar(label='Pixel Value Difference')
                    
                    # Plot star profiles comparison
                    plt.subplot(2, 2, 4)
                    # Find a bright star for profile comparison
                    # Use a simple maximum finder
                    smoothed = np.copy(matrix_data)
                    # Simple box smoothing to reduce noise
                    for _ in range(2):
                        smoothed[1:-1, 1:-1] = (smoothed[:-2, 1:-1] + smoothed[2:, 1:-1] + 
                                              smoothed[1:-1, :-2] + smoothed[1:-1, 2:]) / 4
                    
                    # Find potential stars by looking for local maxima
                    threshold = np.percentile(smoothed, 99.5)
                    y, x = np.where((smoothed > threshold) & 
                                    (smoothed > np.roll(smoothed, 1, axis=0)) & 
                                    (smoothed > np.roll(smoothed, -1, axis=0)) & 
                                    (smoothed > np.roll(smoothed, 1, axis=1)) & 
                                    (smoothed > np.roll(smoothed, -1, axis=1)))
                    
                    if len(y) > 0:
                        # Use the brightest star
                        idx = np.argmax(smoothed[y, x])
                        star_y, star_x = y[idx], x[idx]
                        
                        # Extract profiles (11x11 patch)
                        size = 5
                        y_range = slice(max(0, star_y-size), min(matrix_data.shape[0], star_y+size+1))
                        x_range = slice(max(0, star_x-size), min(matrix_data.shape[1], star_x+size+1))
                        
                        matrix_patch = matrix_data[y_range, x_range]
                        wcs_patch = wcs_data[y_range, x_range]
                        
                        # Get central row and column profiles
                        matrix_row = matrix_patch[size if star_y-y_range.start >= size else star_y-y_range.start, :]
                        matrix_col = matrix_patch[:, size if star_x-x_range.start >= size else star_x-x_range.start]
                        wcs_row = wcs_patch[size if star_y-y_range.start >= size else star_y-y_range.start, :]
                        wcs_col = wcs_patch[:, size if star_x-x_range.start >= size else star_x-x_range.start]
                        
                        # Plot profiles
                        plt.plot(matrix_row, 'b-', label='Matrix (Horizontal)')
                        plt.plot(matrix_col, 'b--', label='Matrix (Vertical)')
                        plt.plot(wcs_row, 'r-', label='WCS (Horizontal)')
                        plt.plot(wcs_col, 'r--', label='WCS (Vertical)')
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.title(f'Star Profile at ({star_x}, {star_y})')
                    else:
                        plt.text(0.5, 0.5, "No bright stars found for profile", 
                                ha='center', va='center', transform=plt.gca().transAxes)
                    
                    plt.tight_layout()
                    comparison_file = os.path.join(args.output, 'method_comparison.png')
                    plt.savefig(comparison_file, dpi=300)
                    plt.close()
                    print(f"Saved method comparison to {comparison_file}")
        except Exception as e:
            print(f"Error creating comparison: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Run with the selected method
        stack_images(
            fits_files,
            reference_file=args.reference,
            output_dir=args.output,
            interpolation_order=args.interpolation,
            padding_percent=args.padding,
            rejection_threshold=args.sigma_clip,
            normalize=not args.no_normalize,
            weight_by_snr=args.weight_by_snr,
            use_wcs=args.use_wcs,
            debug_mode=args.debug
        )

if __name__ == "__main__":
    main()
