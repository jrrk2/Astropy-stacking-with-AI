#!/usr/bin/env python3
"""
Script to compare transformation offsets from star detection versus FITS headers
"""

import os
import glob
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import matplotlib.pyplot as plt
from skimage import transform
from tqdm import tqdm
import argparse
import pandas as pd
from tabulate import tabulate

def detect_stars(image, fwhm=5.0, threshold=3.0, max_stars=100):
    """Detect stars in an image using photutils"""
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
                print("Cannot determine image statistics.")
                return []
        
        # Subtract background
        image_bg_sub = image - median
        
        # Find stars
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
        sources = daofind(image_bg_sub)
        
        if sources is None or len(sources) == 0:
            print("No stars detected. Trying with lower threshold...")
            daofind = DAOStarFinder(fwhm=fwhm, threshold=2.0*std)
            sources = daofind(image_bg_sub)
            
            if sources is None or len(sources) == 0:
                print("Still no stars detected.")
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
    """Match stars between two lists of positions"""
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
    """Estimate transformation matrix from matching stars"""
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

def extract_translation_from_matrix(matrix):
    """Extract translation components from a 3x3 transformation matrix"""
    if matrix is None or not isinstance(matrix, np.ndarray):
        return None, None

    # For a projective transformation, the translation is in the last column
    # We need to normalize by the perspective component
    if matrix[2, 2] != 0:
        tx = matrix[0, 2] / matrix[2, 2]
        ty = matrix[1, 2] / matrix[2, 2]
        return tx, ty
    else:
        return None, None

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

def extract_matrices_from_fits(fits_files):
    """Extract transformation matrices from FITS headers"""
    matrices = []
    
    for fits_file in fits_files:
        matrix = get_matrix_from_fits(fits_file)
        matrices.append(matrix)
        
        if matrix is not None:
            print(f"Extracted transformation from {os.path.basename(fits_file)}")
        else:
            print(f"No transformation matrix found in {os.path.basename(fits_file)}")
    
    return matrices

def compare_transformations(fits_files, output_file=None):
    """Compare transformations from star detection and FITS headers"""
    if len(fits_files) < 2:
        print("Need at least 2 FITS files for comparison")
        return
    
    # Use first file as reference
    ref_file = fits_files[0]
    print(f"Using {os.path.basename(ref_file)} as reference")
    
    # Extract matrices from FITS headers
    fits_matrices = extract_matrices_from_fits(fits_files)
    
    # Load reference image
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
        
        # Initialize results table
        results = []
        
        # Process each non-reference file
        for i, fits_file in enumerate(fits_files[1:], 1):
            file_basename = os.path.basename(fits_file)
            print(f"\nProcessing {file_basename}")
            
            try:
                # Load image
                target_image = fits.getdata(fits_file)
                
                # Ensure 2D image
                if len(target_image.shape) > 2:
                    target_image = target_image[0]
                
                # Get matrix from FITS header
                fits_matrix = fits_matrices[i]
                fits_tx, fits_ty = extract_translation_from_matrix(fits_matrix)
                
                # Detect stars
                target_stars = detect_stars(target_image)
                
                if not target_stars:
                    print("Failed to detect stars")
                    row = {
                        'File': file_basename,
                        'Stars Detected': 0,
                        'Stars Matched': 0,
                        'Star X Offset': None,
                        'Star Y Offset': None,
                        'FITS X Offset': fits_tx,
                        'FITS Y Offset': fits_ty,
                        'X Diff': None,
                        'Y Diff': None
                    }
                    results.append(row)
                    continue
                
                # Match stars
                matches = match_stars(ref_stars, target_stars)
                
                if len(matches) < 3:
                    print(f"Not enough matches: {len(matches)}")
                    row = {
                        'File': file_basename,
                        'Stars Detected': len(target_stars),
                        'Stars Matched': len(matches),
                        'Star X Offset': None,
                        'Star Y Offset': None,
                        'FITS X Offset': fits_tx,
                        'FITS Y Offset': fits_ty,
                        'X Diff': None,
                        'Y Diff': None
                    }
                    results.append(row)
                    continue
                
                # Calculate average offset directly from matched stars
                offsets = []
                for (ref_x, ref_y), (target_x, target_y) in matches:
                    dx = target_x - ref_x
                    dy = target_y - ref_y
                    offsets.append((dx, dy))
                
                avg_dx = np.mean([dx for dx, dy in offsets])
                avg_dy = np.mean([dy for dx, dy in offsets])
                
                # Estimate transformation matrix
                star_matrix = estimate_transformation(matches)
                star_tx, star_ty = extract_translation_from_matrix(star_matrix)
                
                # Calculate difference
                x_diff = None
                y_diff = None
                if fits_tx is not None and star_tx is not None:
                    x_diff = fits_tx - star_tx
                    y_diff = fits_ty - star_ty
                
                # Add to results
                row = {
                    'File': file_basename,
                    'Stars Detected': len(target_stars),
                    'Stars Matched': len(matches),
                    'Star X Offset': star_tx,
                    'Star Y Offset': star_ty,
                    'Star Avg X': avg_dx,
                    'Star Avg Y': avg_dy,
                    'FITS X Offset': fits_tx,
                    'FITS Y Offset': fits_ty,
                    'X Diff': x_diff,
                    'Y Diff': y_diff
                }
                results.append(row)
                
                # Create visualization
                output_dir = os.path.dirname(output_file) if output_file else '.'
                vis_dir = os.path.join(output_dir, 'visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                
                vis_file = os.path.join(vis_dir, f"comparison_{i}.png")
                create_comparison_visualization(ref_image, target_image, ref_stars, target_stars, 
                                             matches, fits_matrix, star_matrix, vis_file)
                
            except Exception as e:
                print(f"Error processing {fits_file}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create DataFrame from results
        df = pd.DataFrame(results)
        
        # Print table
        print("\nTransformation Comparison:\n")
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
        
        # Save to file if requested
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
            
            # Also save as formatted text
            with open(output_file.replace('.csv', '.txt'), 'w') as f:
                f.write("Transformation Comparison\n")
                f.write("=======================\n\n")
                f.write(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
                f.write("\n\n")
                f.write("Note: Star offsets are derived from star pattern matching.\n")
                f.write("FITS offsets are from the transformation matrices in FITS headers.\n")
                f.write("All offsets are in pixels.\n")
    
    except Exception as e:
        print(f"Error processing reference file {ref_file}: {e}")
        import traceback
        traceback.print_exc()

def create_comparison_visualization(ref_image, target_image, ref_stars, target_stars, 
                                 matches, fits_matrix, star_matrix, output_file):
    """Create visualization comparing star-based and FITS-based transformations"""
    plt.figure(figsize=(15, 12))
    
    # Plot reference image with stars
    plt.subplot(2, 2, 1)
    plt.imshow(ref_image, cmap='gray', origin='lower', norm=plt.Normalize(*sigma_clipped_stats(ref_image)[1:]))
    plt.title('Reference Image with Stars')
    for x, y in ref_stars:
        plt.plot(x, y, 'ro', markersize=5)
    
    # Plot target image with stars
    plt.subplot(2, 2, 2)
    plt.imshow(target_image, cmap='gray', origin='lower', norm=plt.Normalize(*sigma_clipped_stats(target_image)[1:]))
    plt.title('Target Image with Stars')
    for x, y in target_stars:
        plt.plot(x, y, 'bo', markersize=5)
    
    # Plot matches
    plt.subplot(2, 2, 3)
    plt.imshow(ref_image, cmap='gray', origin='lower', norm=plt.Normalize(*sigma_clipped_stats(ref_image)[1:]))
    plt.title('Star Matches')
    for (rx, ry), (tx, ty) in matches:
        plt.plot([rx, tx], [ry, ty], 'g-', alpha=0.3)
        plt.plot(rx, ry, 'ro', markersize=5)
        plt.plot(tx, ty, 'bo', markersize=5)
    
    # Plot transformation comparison
    plt.subplot(2, 2, 4)
    # Create a grid of points
    h, w = ref_image.shape
    grid_step = 200
    y, x = np.mgrid[0:h:grid_step, 0:w:grid_step]
    points = np.c_[x.flatten(), y.flatten()]
    
    # Transform points using both matrices
    if fits_matrix is not None and star_matrix is not None:
        # Homogeneous coordinates
        homog_points = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Transform using FITS matrix
        fits_transformed = np.dot(homog_points, fits_matrix.T)
        fits_transformed = fits_transformed[:, :2] / fits_transformed[:, 2, np.newaxis]
        
        # Transform using star-based matrix
        star_transformed = np.dot(homog_points, star_matrix.T)
        star_transformed = star_transformed[:, :2] / star_transformed[:, 2, np.newaxis]
        
        # Plot
        plt.imshow(ref_image, cmap='gray', origin='lower', norm=plt.Normalize(*sigma_clipped_stats(ref_image)[1:]))
        plt.title('Transformation Comparison')
        
        for i in range(len(points)):
            x, y = points[i]
            fx, fy = fits_transformed[i]
            sx, sy = star_transformed[i]
            
            plt.plot(x, y, 'ko', markersize=3)  # Original point
            plt.plot(fx, fy, 'ro', markersize=3)  # FITS transform
            plt.plot(sx, sy, 'bo', markersize=3)  # Star transform
            
            # Draw lines
            plt.plot([x, fx], [y, fy], 'r-', alpha=0.3)  # To FITS
            plt.plot([x, sx], [y, sy], 'b-', alpha=0.3)  # To star
        
        plt.legend(['Original', 'FITS Transform', 'Star Transform'])
    else:
        plt.text(0.5, 0.5, "Missing transformation matrices", 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    
    # Add title
    plt.suptitle('Star Detection vs. FITS Header Transformation Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save visualization
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved visualization to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Compare transformations from star detection and FITS headers')
    parser.add_argument('fits_directory', help='Directory containing FITS files')
    parser.add_argument('--output', '-o', default='transform_comparison.csv', help='Output file')
    parser.add_argument('--first_n', type=int, default=10, help='Process only first N files (default: 10)')
    parser.add_argument('--fwhm', type=float, default=5.0, help='FWHM for star detection (default: 5.0)')
    parser.add_argument('--threshold', type=float, default=3.0, help='Detection threshold in sigmas (default: 3.0)')
    parser.add_argument('--max_stars', type=int, default=100, help='Maximum stars to use (default: 100)')
    
    args = parser.parse_args()
    
    # Find all FITS files in the directory
    fits_files = sorted(glob.glob(os.path.join(args.fits_directory, "*.fits")))
    
    if not fits_files:
        print(f"No FITS files found in {args.fits_directory}")
        return
    
    print(f"Found {len(fits_files)} FITS files")
    
    # Limit to files with matrices
    print("Checking for files with matrices...")
    files_with_matrices = []
    for fits_file in tqdm(fits_files):
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
    
    # Override global parameters with command line arguments
    global detect_stars
    original_detect_stars = detect_stars
    
    def custom_detect_stars(image):
        return original_detect_stars(image, fwhm=args.fwhm, threshold=args.threshold, max_stars=args.max_stars)
    
    # Replace the function
    detect_stars = custom_detect_stars
    
    try:
        # Compare transformations
        compare_transformations(files_with_matrices, args.output)
    finally:
        # Restore original function
        detect_stars = original_detect_stars

if __name__ == "__main__":
    main()
