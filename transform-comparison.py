#!/usr/bin/env python3
"""
Script to compare transformation offsets from star detection versus FITS header matrices
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
                    
                    matrices.append(matrix)
                else:
                    matrices.append(None)
                    
        except Exception as e:
            print(f"Error reading {fits_file}: {e}")
            matrices.append(None)
    
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

def main():
    parser = argparse.ArgumentParser(description='Compare transformations from star detection and FITS headers')
    parser.add_argument('fits_directory', help='Directory containing FITS files')
    parser.add_argument('--output', '-o', default='transform_comparison.csv', help='Output file')
    parser.add_argument('--first_n', type=int, default=None, help='Process only first N files')
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
    
    # Limit to first N files if specified
    if args.first_n and args.first_n < len(fits_files):
        print(f"Limiting to first {args.first_n} files")
        fits_files = fits_files[:args.first_n]
    
    # Override global parameters with command line arguments
    global detect_stars
    original_detect_stars = detect_stars
    
    def custom_detect_stars(image):
        return original_detect_stars(image, fwhm=args.fwhm, threshold=args.threshold, max_stars=args.max_stars)
    
    detect_stars = custom_detect_stars
    
    try:
        # Compare transformations
        compare_transformations(fits_files, args.output)
    finally:
        # Restore original function
        detect_stars = original_detect_stars

if __name__ == "__main__":
    main()
