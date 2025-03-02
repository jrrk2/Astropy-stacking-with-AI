#!/usr/bin/env python3
"""
Script to validate frame alignments using cross-correlation
and compare with FITS matrix transformations.
"""

import os
import glob
import numpy as np
from astropy.io import fits
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

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

def extract_translation(matrix):
    """Extract translation components from a 3x3 transformation matrix"""
    if matrix is None:
        return None, None
    
    try:
        # For a projective transformation, the translation components are in the last column
        tx = matrix[0, 2]
        ty = matrix[1, 2]
        
        # But we need to normalize by the perspective component
        if matrix[2, 2] != 0:
            tx /= matrix[2, 2]
            ty /= matrix[2, 2]
            
        return tx, ty
    except:
        return None, None

def phase_cross_correlation(reference, image, upsample_factor=100):
    """
    Compute cross-correlation to find the shift between images
    
    Parameters:
    -----------
    reference : 2D array
        Reference image
    image : 2D array
        Image to register
    upsample_factor : int
        Upsampling factor for subpixel precision
        
    Returns:
    --------
    shift : (dy, dx)
        Estimated shift in y and x
    """
    # Ensure both images have the same shape
    min_shape = (min(reference.shape[0], image.shape[0]), 
                min(reference.shape[1], image.shape[1]))
    
    reference_crop = reference[:min_shape[0], :min_shape[1]]
    image_crop = image[:min_shape[0], :min_shape[1]]
    
    # Use normalized spectrum for better correlation
    reference_spectrum = np.fft.fft2(reference_crop - np.mean(reference_crop))
    image_spectrum = np.fft.fft2(image_crop - np.mean(image_crop))
    
    # Compute cross-power spectrum
    cross_power = reference_spectrum * np.conj(image_spectrum)
    cross_power = cross_power / np.abs(cross_power)
    
    # Compute inverse FFT and locate maximum
    cc_image = np.fft.ifft2(cross_power).real
    
    # Find the peak
    max_y, max_x = np.unravel_index(np.argmax(cc_image), cc_image.shape)
    
    # Extract subpixel shift
    if upsample_factor > 1:
        # If upsampling > 1, refine estimate with matrix multiply DFT
        # Adapted from scikit-image's phase_cross_correlation
        # Center the cross-correlation to prepare for DFT upsampling
        max_y_shift = max_y
        max_x_shift = max_x
        
        if max_y > reference_crop.shape[0] // 2:
            max_y_shift = max_y - reference_crop.shape[0]
        if max_x > reference_crop.shape[1] // 2:
            max_x_shift = max_x - reference_crop.shape[1]
            
        # Subpixel shift estimation
        # NOTE: This is a simplified version; scikit-image has a more robust implementation
        cc_image = np.roll(cc_image, (-max_y_shift, -max_x_shift), (0, 1))
        
        # Localize the peak with subpixel accuracy
        y, x = np.unravel_index(np.argmax(cc_image), cc_image.shape)
        
        # Calculate shifts as offsets from the image center
        shift_y = -(y - reference_crop.shape[0] // 2) / upsample_factor
        shift_x = -(x - reference_crop.shape[1] // 2) / upsample_factor
        
        return shift_y, shift_x
    else:
        # Calculate shifts from the image center
        shift_y = -((max_y + reference_crop.shape[0] // 2) % reference_crop.shape[0] - reference_crop.shape[0] // 2)
        shift_x = -((max_x + reference_crop.shape[1] // 2) % reference_crop.shape[1] - reference_crop.shape[1] // 2)
        
        return shift_y, shift_x

def direct_cross_correlation(reference, image, search_window=50):
    """
    Compute direct cross-correlation using a sliding window approach
    
    Parameters:
    -----------
    reference : 2D array
        Reference image
    image : 2D array
        Image to register
    search_window : int
        Maximum search radius
        
    Returns:
    --------
    shift : (dy, dx)
        Estimated shift
    """
    h, w = reference.shape
    
    # Crop both images to the same size, avoiding edges
    crop_size = min(h, w) - 2*search_window
    start_y = h // 2 - crop_size // 2
    start_x = w // 2 - crop_size // 2
    
    ref_crop = reference[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    # Initialize correlation matrix
    corr = np.zeros((2*search_window+1, 2*search_window+1))
    
    # Compute correlation for each possible shift
    for dy in range(-search_window, search_window+1):
        for dx in range(-search_window, search_window+1):
            y_start = start_y + dy
            x_start = start_x + dx
            
            if y_start < 0 or x_start < 0 or y_start + crop_size > h or x_start + crop_size > w:
                continue
                
            img_crop = image[y_start:y_start+crop_size, x_start:x_start+crop_size]
            
            # Compute normalized correlation coefficient
            ref_mean = np.mean(ref_crop)
            img_mean = np.mean(img_crop)
            
            ref_std = np.std(ref_crop)
            img_std = np.std(img_crop)
            
            if ref_std > 0 and img_std > 0:
                corr[dy+search_window, dx+search_window] = np.mean(
                    (ref_crop - ref_mean) * (img_crop - img_mean)
                ) / (ref_std * img_std)
    
    # Find the peak
    max_y, max_x = np.unravel_index(np.argmax(corr), corr.shape)
    
    # Convert to shift
    shift_y = max_y - search_window
    shift_x = max_x - search_window
    
    return shift_y, shift_x, corr

def compare_shifts(fits_files, output_dir, method='direct'):
    """
    Compare shifts detected by cross-correlation with FITS matrix transformations
    
    Parameters:
    -----------
    fits_files : list
        List of FITS file paths
    output_dir : str
        Output directory
    method : str
        Cross-correlation method ('fft' or 'direct')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if len(fits_files) < 2:
        print("Need at least 2 FITS files")
        return
    
    # Use first file as reference
    ref_file = fits_files[0]
    print(f"Using {os.path.basename(ref_file)} as reference")
    
    # Load reference image
    with fits.open(ref_file) as hdul:
        ref_data = hdul[0].data
        if len(ref_data.shape) > 2:
            print(f"Reference has {len(ref_data.shape)} dimensions, using first channel")
            ref_data = ref_data[0]
    
    # Get reference matrix
    ref_matrix = get_matrix_from_fits(ref_file)
    
    # Store results
    results = []
    
    # Process each non-reference file
    for i, fits_file in enumerate(fits_files[1:], 1):
        file_basename = os.path.basename(fits_file)
        print(f"\nProcessing {file_basename} ({i}/{len(fits_files)-1})")
        
        # Load image
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data
            if len(image_data.shape) > 2:
                image_data = image_data[0]
        
        # Get matrix transformation
        matrix = get_matrix_from_fits(fits_file)
        
        if matrix is not None and ref_matrix is not None:
            # Calculate transformation between frames
            # For shift between frames, we need to do:
            # ref_matrix_inv @ matrix
            try:
                ref_matrix_inv = np.linalg.inv(ref_matrix)
                combined_matrix = np.dot(ref_matrix_inv, matrix)
                
                # Extract translation
                matrix_dx, matrix_dy = extract_translation(combined_matrix)
                matrix_shift = (matrix_dx, matrix_dy)
                
                print(f"Matrix shift: dx={matrix_dx:.2f}, dy={matrix_dy:.2f}")
            except:
                matrix_dx, matrix_dy = None, None
                matrix_shift = None
                print("Could not compute matrix shift")
        else:
            matrix_dx, matrix_dy = None, None
            matrix_shift = None
            print("No matrix available")
        
        # Compute correlation-based shift
        try:
            if method == 'fft':
                # FFT-based method
                shift_y, shift_x = phase_cross_correlation(ref_data, image_data)
                corr_image = None
            else:
                # Direct correlation approach
                shift_y, shift_x, corr_image = direct_cross_correlation(ref_data, image_data)
                
            # Convert to dx, dy for consistency
            corr_dx, corr_dy = shift_x, shift_y
            corr_shift = (corr_dx, corr_dy)
            
            print(f"Correlation shift: dx={corr_dx:.2f}, dy={corr_dy:.2f}")
        except Exception as e:
            print(f"Error computing correlation: {e}")
            corr_dx, corr_dy = None, None
            corr_shift = None
            corr_image = None
        
        # Calculate difference if both available
        if matrix_shift is not None and corr_shift is not None:
            diff_dx = matrix_dx - corr_dx
            diff_dy = matrix_dy - corr_dy
            print(f"Difference: dx={diff_dx:.2f}, dy={diff_dy:.2f}")
        else:
            diff_dx, diff_dy = None, None
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Show images
        plt.subplot(2, 3, 1)
        plt.imshow(ref_data, cmap='gray', origin='lower', 
                  norm=plt.Normalize(*np.percentile(ref_data[ref_data > 0], [5, 99])))
        plt.title('Reference Image')
        
        plt.subplot(2, 3, 2)
        plt.imshow(image_data, cmap='gray', origin='lower',
                  norm=plt.Normalize(*np.percentile(image_data[image_data > 0], [5, 99])))
        plt.title(f'Image {i}')
        
        # Show correlation map if available
        if corr_image is not None:
            plt.subplot(2, 3, 3)
            plt.imshow(corr_image, cmap='viridis', origin='lower')
            plt.colorbar()
            plt.title('Correlation Map')
            
            # Mark the peak
            max_y, max_x = np.unravel_index(np.argmax(corr_image), corr_image.shape)
            plt.plot(max_x, max_y, 'r+', markersize=10)
            
        # Create difference visualization
        # Shift images based on matrix transform
        if matrix_shift is not None:
            # Create shifted image using matrix prediction
            h, w = ref_data.shape
            y, x = np.indices((h, w))
            
            # Apply reverse shift to coordinates
            y_shifted = y - matrix_dy
            x_shifted = x - matrix_dx
            
            # Enforce bounds
            valid = (y_shifted >= 0) & (y_shifted < h) & (x_shifted >= 0) & (x_shifted < w)
            
            # Create empty shifted image
            matrix_shifted = np.zeros_like(ref_data)
            
            # Fill in values from original at shifted positions
            y_valid = y[valid].astype(int)
            x_valid = x[valid].astype(int)
            y_shifted_valid = y_shifted[valid].astype(int)
            x_shifted_valid = x_shifted[valid].astype(int)
            
            matrix_shifted[y_valid, x_valid] = image_data[y_shifted_valid, x_shifted_valid]
            
            # Show matrix-shifted image
            plt.subplot(2, 3, 4)
            plt.imshow(matrix_shifted, cmap='gray', origin='lower',
                      norm=plt.Normalize(*np.percentile(matrix_shifted[matrix_shifted > 0], [5, 99])))
            plt.title('Image Shifted by Matrix')
            
            # Create difference image
            diff_image = np.zeros((h, w, 3))
            ref_norm = ref_data / np.percentile(ref_data, 99) if np.percentile(ref_data, 99) > 0 else ref_data
            shifted_norm = matrix_shifted / np.percentile(matrix_shifted, 99) if np.percentile(matrix_shifted, 99) > 0 else matrix_shifted
            
            diff_image[:, :, 0] = np.clip(ref_norm, 0, 1)  # Red = reference
            diff_image[:, :, 2] = np.clip(shifted_norm, 0, 1)  # Blue = shifted
            
            plt.subplot(2, 3, 5)
            plt.imshow(diff_image, origin='lower')
            plt.title('Difference (Red=Ref, Blue=Shifted)')
        
        # Add text with measurements
        ax = plt.subplot(2, 3, 6)
        ax.axis('off')
        
        y_pos = 0.9
        ax.text(0.5, y_pos, f"File: {file_basename}", ha='center', va='center', fontsize=10, transform=ax.transAxes)
        y_pos -= 0.1
        
        if matrix_shift is not None:
            ax.text(0.5, y_pos, f"Matrix shift: dx={matrix_dx:.2f}, dy={matrix_dy:.2f}", 
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
            y_pos -= 0.1
            
        if corr_shift is not None:
            ax.text(0.5, y_pos, f"Correlation shift: dx={corr_dx:.2f}, dy={corr_dy:.2f}", 
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
            y_pos -= 0.1
            
        if diff_dx is not None and diff_dy is not None:
            ax.text(0.5, y_pos, f"Difference: dx={diff_dx:.2f}, dy={diff_dy:.2f}", 
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
            
            # Calculate magnitude of difference
            diff_magnitude = np.sqrt(diff_dx**2 + diff_dy**2)
            y_pos -= 0.1
            ax.text(0.5, y_pos, f"Difference magnitude: {diff_magnitude:.2f} pixels", 
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'correlation_{i:03d}.png'))
        plt.close()
        
        # Store result
        results.append({
            'file': file_basename,
            'matrix_dx': matrix_dx,
            'matrix_dy': matrix_dy,
            'corr_dx': corr_dx,
            'corr_dy': corr_dy,
            'diff_dx': diff_dx,
            'diff_dy': diff_dy,
            'diff_magnitude': np.sqrt(diff_dx**2 + diff_dy**2) if diff_dx is not None and diff_dy is not None else None
        })
    
    # Save results
    if results:
        with open(os.path.join(output_dir, 'shift_comparison.csv'), 'w') as f:
            f.write("File,Matrix dX,Matrix dY,Correlation dX,Correlation dY,Difference dX,Difference dY,Difference Magnitude\n")
            
            for r in results:
                f.write(f"{r['file']},{r['matrix_dx']},{r['matrix_dy']},{r['corr_dx']},{r['corr_dy']},{r['diff_dx']},{r['diff_dy']},{r['diff_magnitude']}\n")
        
        # Create summary visualization
        plt.figure(figsize=(15, 10))
        
        # Plot shifts from both methods
        plt.subplot(2, 2, 1)
        matrix_dx = [r['matrix_dx'] for r in results if r['matrix_dx'] is not None]
        matrix_dy = [r['matrix_dy'] for r in results if r['matrix_dy'] is not None]
        corr_dx = [r['corr_dx'] for r in results if r['corr_dx'] is not None]
        corr_dy = [r['corr_dy'] for r in results if r['corr_dy'] is not None]
        
        if matrix_dx and matrix_dy:
            plt.scatter(matrix_dx, matrix_dy, c='red', label='Matrix')
        if corr_dx and corr_dy:
            plt.scatter(corr_dx, corr_dy, c='blue', label='Correlation')
        
        plt.xlabel('dX (pixels)')
        plt.ylabel('dY (pixels)')
        plt.title('Detected Shifts')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot difference magnitudes
        plt.subplot(2, 2, 2)
        diff_magnitudes = [r['diff_magnitude'] for r in results if r['diff_magnitude'] is not None]
        
        if diff_magnitudes:
            plt.bar(range(len(diff_magnitudes)), diff_magnitudes)
            plt.xlabel('Frame Index')
            plt.ylabel('Difference Magnitude (pixels)')
            plt.title('Difference Between Methods')
            plt.grid(True, alpha=0.3)
        
        # Plot comparison between methods
        plt.subplot(2, 2, 3)
        
        if corr_dx and matrix_dx:
            plt.scatter(matrix_dx, corr_dx)
            plt.xlabel('Matrix dX')
            plt.ylabel('Correlation dX')
            plt.title('dX Comparison')
            plt.grid(True, alpha=0.3)
            
            # Plot ideal line
            min_val = min(min(matrix_dx), min(corr_dx))
            max_val = max(max(matrix_dx), max(corr_dx))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        plt.subplot(2, 2, 4)
        
        if corr_dy and matrix_dy:
            plt.scatter(matrix_dy, corr_dy)
            plt.xlabel('Matrix dY')
            plt.ylabel('Correlation dY')
            plt.title('dY Comparison')
            plt.grid(True, alpha=0.3)
            
            # Plot ideal line
            min_val = min(min(matrix_dy), min(corr_dy))
            max_val = max(max(matrix_dy), max(corr_dy))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shift_summary.png'))
        plt.close()
        
        print(f"Results saved to {os.path.join(output_dir, 'shift_comparison.csv')}")

def main():
    parser = argparse.ArgumentParser(description='Compare shifts from cross-correlation with FITS matrices')
    parser.add_argument('fits_directory', help='Directory containing FITS files')
    parser.add_argument('--output', '-o', default='correlation_results', help='Output directory')
    parser.add_argument('--first_n', type=int, default=10, help='Process only first N files (default: 10)')
    parser.add_argument('--method', choices=['fft', 'direct'], default='direct',
                      help='Cross-correlation method (default: direct)')
    
    args = parser.parse_args()
    
    # Find all FITS files in the directory
    fits_files = sorted(glob.glob(os.path.join(args.fits_directory, "*.fits")))
    
    if not fits_files:
        print(f"No FITS files found in {args.fits_directory}")
        return
    
    print(f"Found {len(fits_files)} FITS files")
    
    # Filter for files with matrices
    files_with_matrices = []
    for fits_file in tqdm(fits_files, desc="Checking files"):
        matrix = get_matrix_from_fits(fits_file)
        if matrix is not None:
            files_with_matrices.append(fits_file)
    
    if not files_with_matrices:
        print("No files with matrices found. Cannot proceed.")
        return
    
    print(f"Found {len(files_with_matrices)} files with transformation matrices")
    
    # Limit to first N files
    if args.first_n and args.first_n < len(files_with_matrices):
        print(f"Limiting to first {args.first_n} files")
        files_with_matrices = files_with_matrices[:args.first_n]
    
    # Compare shifts
    compare_shifts(files_with_matrices, args.output, method=args.method)

if __name__ == "__main__":
    main()
