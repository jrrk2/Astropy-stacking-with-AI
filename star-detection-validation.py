#!/usr/bin/env python3
"""
Script to validate transformation matrices by comparing star positions 
with detection-based alignment using photutils and FFT registration.
"""

import os
import glob
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy import signal
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from skimage import transform
from tqdm import tqdm
import argparse

def detect_stars(image, fwhm=3.0, threshold=5.0, max_stars=100):
    """
    Detect stars in an image using photutils.
    
    Parameters:
    -----------
    image : 2D array
        Input image
    fwhm : float
        Full width at half maximum of the stars
    threshold : float
        Detection threshold in sigmas
    max_stars : int
        Maximum number of stars to return
        
    Returns:
    --------
    positions : list of (x, y) tuples
        Positions of detected stars (brightest stars first)
    """
    # Make sure we have a valid 2D image
    if image is None or image.ndim != 2:
        print(f"Error: Invalid image with shape {image.shape if image is not None else 'None'}")
        return []
    
    # Convert to float if needed
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)
    
    try:
        # Estimate background and noise
        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        
        # Check if we have valid statistics
        if np.isnan(median) or np.isnan(std) or std == 0:
            print(f"Warning: Invalid image statistics. Mean={mean}, Median={median}, Std={std}")
            # Try with a simple percentile estimation instead
            median = np.percentile(image, 50)
            std = np.std(image[image > median])
            if np.isnan(std) or std == 0:
                print("Cannot determine image statistics. Image may be blank or corrupt.")
                return []
        
        # Debug info
        print(f"Image stats: median={median:.2f}, std={std:.2f}")
        
        # Subtract background
        image_bg_sub = image - median
        
        # Find stars
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
        sources = daofind(image_bg_sub)
        
        if sources is None or len(sources) == 0:
            print("No stars detected. Trying with lower threshold...")
            daofind = DAOStarFinder(fwhm=fwhm, threshold=3.0*std)
            sources = daofind(image_bg_sub)
            
            if sources is None or len(sources) == 0:
                print("Still no stars detected. Trying with different FWHM...")
                daofind = DAOStarFinder(fwhm=5.0, threshold=3.0*std)
                sources = daofind(image_bg_sub)
                
                if sources is None or len(sources) == 0:
                    print("No stars detected after multiple attempts.")
                    return []
        
        # Sort by brightness (flux)
        sources.sort('flux', reverse=True)
        
        # Limit to max_stars
        if len(sources) > max_stars:
            sources = sources[:max_stars]
        
        # Return positions
        positions = [(x, y) for x, y in zip(sources['xcentroid'], sources['ycentroid'])]
        print(f"Detected {len(positions)} stars")
        
        return positions
        
    except Exception as e:
        print(f"Error in star detection: {e}")
        import traceback
        traceback.print_exc()
        return []

def match_stars(ref_positions, target_positions, max_distance=10.0):
    """
    Match stars between two lists of positions
    
    Parameters:
    -----------
    ref_positions : list of (x, y) tuples
        Reference star positions
    target_positions : list of (x, y) tuples
        Target star positions
    max_distance : float
        Maximum distance between matching stars
        
    Returns:
    --------
    matches : list of ((x1, y1), (x2, y2)) tuples
        Matching star positions
    """
    matches = []
    
    for ref_x, ref_y in ref_positions:
        best_match = None
        best_dist = max_distance
        
        for target_x, target_y in target_positions:
            dist = np.sqrt((ref_x - target_x)**2 + (ref_y - target_y)**2)
            if dist < best_dist:
                best_dist = dist
                best_match = (target_x, target_y)
        
        if best_match:
            matches.append(((ref_x, ref_y), best_match))
    
    print(f"Matched {len(matches)} stars")
    return matches

def estimate_transformation(matches):
    """
    Estimate transformation matrix from matching stars
    
    Parameters:
    -----------
    matches : list of ((x1, y1), (x2, y2)) tuples
        Matching star positions
        
    Returns:
    --------
    matrix : 3x3 array
        Transformation matrix
    """
    if len(matches) < 3:
        print("Not enough matches to estimate transformation")
        return None
    
    # Extract source and destination points
    src_points = np.array([match[0] for match in matches])
    dst_points = np.array([match[1] for match in matches])
    
    # Estimate transformation
    tform = transform.estimate_transform('projective', src_points, dst_points)
    
    # Get matrix
    matrix = tform.params
    print(f"Estimated transformation matrix:")
    print(matrix)
    
    return matrix

def phase_cross_correlation(reference, image):
    """
    Compute the phase cross-correlation between two images
    
    Parameters:
    -----------
    reference : 2D array
        Reference image
    image : 2D array
        Image to register
        
    Returns:
    --------
    shift : (y, x) tuple
        Estimated shift
    error : float
        Translation invariant normalized RMS error
    diffphase : float
        Global phase difference
    """
    # Ensure both images have the same shape
    min_shape = (min(reference.shape[0], image.shape[0]), 
                min(reference.shape[1], image.shape[1]))
    
    reference_crop = reference[:min_shape[0], :min_shape[1]]
    image_crop = image[:min_shape[0], :min_shape[1]]
    
    # Compute FFT-based cross-correlation
    shift_y, shift_x, error, diffphase = signal.phase_cross_correlation(
        reference_crop, image_crop, upsample_factor=100)
    
    print(f"FFT Registration: shift_x={shift_x:.2f}, shift_y={shift_y:.2f}, error={error:.4f}")
    return (shift_y, shift_x), error, diffphase

def validate_transformation(image1, image2, matrix, ref_stars=None, target_stars=None):
    """
    Validate a transformation matrix by comparing it with star detection-based matching
    and FFT registration
    
    Parameters:
    -----------
    image1 : 2D array
        Reference image
    image2 : 2D array
        Image to transform
    matrix : 3x3 array or None
        Transformation matrix to validate
    ref_stars : list of (x, y) tuples
        Precomputed reference star positions (optional)
    target_stars : list of (x, y) tuples
        Precomputed target star positions (optional)
        
    Returns:
    --------
    results : dict
        Validation results
    """
    results = {}
    
    # Detect stars if not provided
    if ref_stars is None:
        ref_stars = detect_stars(image1)
        
    if target_stars is None:
        target_stars = detect_stars(image2)
    
    # If star detection failed, skip
    if not ref_stars or not target_stars:
        print("Star detection failed")
        results['star_detection_success'] = False
        return results
    
    results['star_detection_success'] = True
    results['ref_stars'] = ref_stars
    results['target_stars'] = target_stars
    
    # Check if we have a valid matrix
    transformed_target_stars = []
    
    # Skip matrix transformation if matrix is None
    if matrix is not None and isinstance(matrix, np.ndarray) and matrix.shape == (3, 3):
        try:
            # Apply transformation to target stars
            matrix_inv = np.linalg.inv(matrix)
            
            for x, y in target_stars:
                # Convert to homogeneous coordinates
                point = np.array([x, y, 1.0])
                
                # Apply transformation
                transformed = np.dot(matrix_inv, point)
                
                # Convert back to Cartesian coordinates
                if transformed[2] != 0:
                    transformed_x = transformed[0] / transformed[2]
                    transformed_y = transformed[1] / transformed[2]
                    transformed_target_stars.append((transformed_x, transformed_y))
                else:
                    print(f"Warning: Invalid transformation for point ({x}, {y})")
        except Exception as e:
            print(f"Error applying transformation: {e}")
            # If matrix transformation fails, we'll continue using direct star matching
    else:
        print("No valid transformation matrix provided. Skipping matrix validation.")
        # Use original target stars positions since we can't transform
        transformed_target_stars = target_stars
    
    # Match stars
    matches = match_stars(ref_stars, transformed_target_stars)
    results['matches'] = matches
    
    # Calculate error
    if matches:
        errors = []
        for (ref_x, ref_y), (trans_x, trans_y) in matches:
            error = np.sqrt((ref_x - trans_x)**2 + (ref_y - trans_y)**2)
            errors.append(error)
        
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        results['avg_error'] = avg_error
        results['max_error'] = max_error
        
        print(f"Average error: {avg_error:.2f} pixels")
        print(f"Maximum error: {max_error:.2f} pixels")
    
    # Estimate transformation from matches
    if len(matches) >= 3:
        estimated_matrix = estimate_transformation(matches)
        results['estimated_matrix'] = estimated_matrix
        
        # Compare matrices
        if matrix is not None and estimated_matrix is not None:
            # Normalize matrices
            norm_matrix = matrix / np.linalg.norm(matrix)
            norm_estimated = estimated_matrix / np.linalg.norm(estimated_matrix)
            
            matrix_diff = np.abs(norm_matrix - norm_estimated)
            matrix_similarity = 1.0 - np.mean(matrix_diff)
            
            results['matrix_similarity'] = matrix_similarity
            print(f"Matrix similarity: {matrix_similarity:.4f} (1.0 = identical)")
    
    # Run FFT-based registration
    try:
        shift_result, fft_error, _ = phase_cross_correlation(image1, image2)
        results['fft_shift'] = shift_result
        results['fft_error'] = fft_error
    except Exception as e:
        print(f"FFT registration failed: {e}")
    
    return results

def visualize_validation(image1, image2, results, output_path):
    """
    Visualize validation results
    
    Parameters:
    -----------
    image1 : 2D array
        Reference image
    image2 : 2D array
        Image to transform
    results : dict
        Validation results
    output_path : str
        Output path for visualization
    """
    if not results.get('star_detection_success', False):
        print("Cannot visualize results: star detection failed")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Show images
    plt.subplot(2, 2, 1)
    plt.imshow(image1, cmap='gray', origin='lower')
    plt.title('Reference Image with Stars')
    plt.axis('on')
    
    # Plot reference stars
    for x, y in results['ref_stars']:
        plt.plot(x, y, 'ro', markersize=8, alpha=0.6)
    
    plt.subplot(2, 2, 2)
    plt.imshow(image2, cmap='gray', origin='lower')
    plt.title('Target Image with Stars')
    plt.axis('on')
    
    # Plot target stars
    for x, y in results['target_stars']:
        plt.plot(x, y, 'bo', markersize=8, alpha=0.6)
    
    # Plot matches
    if 'matches' in results:
        plt.subplot(2, 2, 3)
        plt.imshow(image1, cmap='gray', origin='lower')
        plt.title('Star Matches')
        plt.axis('on')
        
        for (ref_x, ref_y), (trans_x, trans_y) in results['matches']:
            plt.plot(ref_x, ref_y, 'ro', markersize=6, alpha=0.6)
            plt.plot(trans_x, trans_y, 'bx', markersize=6, alpha=0.6)
            plt.plot([ref_x, trans_x], [ref_y, trans_y], 'g-', alpha=0.3)
    
    # Plot difference image
    plt.subplot(2, 2, 4)
    
    # Ensure images have the same shape
    min_shape = (min(image1.shape[0], image2.shape[0]), 
                min(image1.shape[1], image2.shape[1]))
    
    image1_crop = image1[:min_shape[0], :min_shape[1]]
    image2_crop = image2[:min_shape[0], :min_shape[1]]
    
    # Normalize images for visualization
    image1_norm = image1_crop / np.max(image1_crop)
    image2_norm = image2_crop / np.max(image2_crop)
    
    # Create RGB difference image (red=image1, blue=image2)
    diff_image = np.zeros((min_shape[0], min_shape[1], 3))
    diff_image[:, :, 0] = image1_norm
    diff_image[:, :, 2] = image2_norm
    
    plt.imshow(diff_image, origin='lower')
    plt.title('Difference (Red=Ref, Blue=Target)')
    plt.axis('on')
    
    # Add summary
    plt.suptitle('Transformation Validation', fontsize=16)
    
    if 'avg_error' in results:
        plt.figtext(0.5, 0.01, 
                   f"Average error: {results['avg_error']:.2f} pixels, Max error: {results['max_error']:.2f} pixels", 
                   ha='center', fontsize=12)
    
    if 'matrix_similarity' in results:
        plt.figtext(0.5, 0.04,
                   f"Matrix similarity: {results['matrix_similarity']:.4f}", 
                   ha='center', fontsize=12)
    
    if 'fft_shift' in results:
        plt.figtext(0.5, 0.07,
                   f"FFT shift: x={results['fft_shift'][1]:.2f}, y={results['fft_shift'][0]:.2f}", 
                   ha='center', fontsize=12)
    
    # Save plot
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    try:
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    finally:
        plt.close()

def validate_fits_files(fits_files, matrices=None, output_dir='validation_results'):
    """
    Validate transformation matrices for a set of FITS files
    
    Parameters:
    -----------
    fits_files : list of str
        List of FITS file paths
    matrices : list of 3x3 arrays
        List of transformation matrices (optional)
    output_dir : str
        Output directory for validation results
    """
    if len(fits_files) < 2:
        print("Need at least 2 FITS files to validate transformations")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use first file as reference
    ref_file = fits_files[0]
    print(f"Using {ref_file} as reference")
    
    try:
        ref_image = fits.getdata(ref_file)
        
        # Ensure 2D image
        if len(ref_image.shape) > 2:
            print(f"Reference image has {len(ref_image.shape)} dimensions, using first channel")
            ref_image = ref_image[0]
            
        # Detect stars in reference
        ref_stars = detect_stars(ref_image)
        
        if not ref_stars:
            print("Failed to detect stars in reference image")
            return
        
        # Process each non-reference file
        for i, fits_file in enumerate(fits_files[1:], 1):
            print(f"\nValidating {fits_file}")
            
            try:
                # Load image
                target_image = fits.getdata(fits_file)
                
                # Ensure 2D image
                if len(target_image.shape) > 2:
                    print(f"Target image has {len(target_image.shape)} dimensions, using first channel")
                    target_image = target_image[0]
                
                # Get matrix if provided
                matrix = matrices[i] if matrices and i < len(matrices) else None
                
                # Detect stars
                target_stars = detect_stars(target_image)
                
                if not target_stars:
                    print("Failed to detect stars in target image")
                    continue
                
                # Run validation
                results = validate_transformation(ref_image, target_image, matrix, 
                                                ref_stars, target_stars)
                
                # Visualize results
                output_path = os.path.join(output_dir, f"validation_{i}.png")
                visualize_validation(ref_image, target_image, results, output_path)
                
            except Exception as e:
                print(f"Error processing {fits_file}: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"Error processing reference file {ref_file}: {e}")
        import traceback
        traceback.print_exc()

def extract_matrices_from_fits(fits_files):
    """Extract transformation matrices from FITS headers"""
    matrices = []
    
    for fits_file in fits_files:
        try:
            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                
                # Check if transformation matrix exists in header
                has_transform = all(f'TRNSFRM{i+1}{j+1}' in header 
                                  for i in range(3) for j in range(3))
                
                if has_transform:
                    # Extract the 3x3 transformation matrix
                    matrix = np.zeros((3, 3))
                    for i in range(3):
                        for j in range(3):
                            matrix[i, j] = header[f'TRNSFRM{i+1}{j+1}']
                    
                    # Print the first matrix details for debugging
                    if len(matrices) == 0:
                        print(f"\nFirst transformation matrix from {os.path.basename(fits_file)}:")
                        print(matrix)
                        print(f"Determinant: {np.linalg.det(matrix)}")
                        print(f"Shape: {matrix.shape}")
                        print()
                    
                    matrices.append(matrix)
                    print(f"Extracted transformation from {os.path.basename(fits_file)}")
                else:
                    print(f"No transformation matrix found in {os.path.basename(fits_file)}")
                    matrices.append(None)
                    
        except Exception as e:
            print(f"Error reading {fits_file}: {e}")
            matrices.append(None)
    
    return matrices

def main():
    parser = argparse.ArgumentParser(description='Validate transformation matrices using star detection')
    parser.add_argument('fits_directory', help='Directory containing FITS files')
    parser.add_argument('--output', '-o', default='validation_results', help='Output directory')
    parser.add_argument('--app_file', help='APP-processed file for comparison')
    parser.add_argument('--first_n', type=int, default=None, help='Process only first N files')
    parser.add_argument('--fwhm', type=float, default=3.0, help='FWHM for star detection (default: 3.0)')
    parser.add_argument('--threshold', type=float, default=5.0, help='Detection threshold in sigmas (default: 5.0)')
    parser.add_argument('--max_stars', type=int, default=100, help='Maximum stars to use (default: 100)')
    
    args = parser.parse_args()
    
    # Find all FITS files in the directory
    fits_files = sorted(glob.glob(os.path.join(args.fits_directory, "*.fits")))
    
    if not fits_files:
        print(f"No FITS files found in {args.fits_directory}")
        return
    
    print(f"Found {len(fits_files)} FITS files")
    
    # Add APP file if provided
    if args.app_file and os.path.exists(args.app_file):
        print(f"Including APP file: {args.app_file}")
        fits_files.insert(0, args.app_file)  # Use APP file as reference
    
    # Limit to first N files if specified
    if args.first_n and args.first_n < len(fits_files):
        print(f"Limiting to first {args.first_n} files")
        fits_files = fits_files[:args.first_n]
    
    # Extract matrices from FITS headers
    matrices = extract_matrices_from_fits(fits_files)
    
    # Set global parameters for star detection
    global detect_stars
    original_detect_stars = detect_stars
    
    # Create a wrapper with the user's parameters
    def custom_detect_stars(image):
        return original_detect_stars(image, fwhm=args.fwhm, threshold=args.threshold, max_stars=args.max_stars)
    
    # Replace the function
    detect_stars = custom_detect_stars
    
    try:
        # Validate transformations
        validate_fits_files(fits_files, matrices, args.output)
    finally:
        # Restore original function
        detect_stars = original_detect_stars

if __name__ == "__main__":
    main()
