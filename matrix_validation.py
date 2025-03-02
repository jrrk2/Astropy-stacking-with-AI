#!/usr/bin/env python3
"""
Diagnostic script to validate transformation matrices in FITS headers and
visualize what they do to the images, without relying on star detection.
Now also correlates transformations with CRVAL1/2 changes.
"""

import os
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage import transform
from tqdm import tqdm
import argparse
from astropy import wcs
from astropy.coordinates import SkyCoord
import astropy.units as u

# Import functions from utils module
from matrix_validation_utils import (
    get_matrix_and_wcs_from_fits,
    apply_transformation,
    create_grid_image,
    extract_translation,
    extract_rotation,
    calculate_wcs_offset,
    generate_html_report
)

def validate_matrices(fits_files, output_dir, rotate_180=True, interpolation_order=1, margin_percent=10):
    """Validate transformation matrices by applying them to images and correlating with WCS
    
    Returns:
        list: A list of dictionaries containing validation results for each file
    """
    if len(fits_files) < 2:
        print("Need at least 2 FITS files")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store validations results
    results = []
    
    # Use first file as reference
    ref_file = fits_files[0]
    ref_matrix, ref_quality, ref_wcs_info = get_matrix_and_wcs_from_fits(ref_file)
    
    if ref_matrix is None:
        print(f"Reference file {ref_file} has no matrix. Trying another file...")
        for f in fits_files[1:]:
            ref_matrix, ref_quality, ref_wcs_info = get_matrix_and_wcs_from_fits(f)
            if ref_matrix is not None:
                ref_file = f
                break
                
        if ref_matrix is None:
            print("No files have matrices. Cannot proceed.")
            return
    
    print(f"Using {os.path.basename(ref_file)} as reference")
    
    # Load the reference image
    with fits.open(ref_file) as hdul:
        ref_data = hdul[0].data
        if len(ref_data.shape) > 2:
            print(f"Reference has {len(ref_data.shape)} dimensions, using first channel")
            ref_data = ref_data[0]
    
    # Calculate padded dimensions
    h, w = ref_data.shape
    margin_h = int(h * margin_percent / 100)
    margin_w = int(w * margin_percent / 100)
    padded_h = h + 2 * margin_h
    padded_w = w + 2 * margin_w
    output_shape = (padded_h, padded_w)
    
    print(f"Original dimensions: {w}x{h}")
    print(f"Padded dimensions: {padded_w}x{padded_h} (with {margin_percent}% margin)")
    
    # Create grid image for transformation visualization
    grid_image = create_grid_image((h, w), grid_spacing=100)
    
    # Create identity matrix for the reference
    identity = np.eye(3)
    
    # Apply to reference (just centers it in the padded output)
    ref_centered = apply_transformation(
        ref_data, identity, output_shape=output_shape, 
        rotate_180=rotate_180, order=interpolation_order
    )
    
    # Also transform the grid
    grid_centered = apply_transformation(
        grid_image, identity, output_shape=output_shape,
        rotate_180=rotate_180, order=interpolation_order
    )
    
    # Save reference images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(ref_centered, cmap='gray', origin='lower', 
              norm=plt.Normalize(*np.percentile(ref_centered[ref_centered > 0], [1, 99])))
    plt.title('Reference Image')
    plt.subplot(1, 2, 2)
    plt.imshow(grid_centered, cmap='gray', origin='lower')
    plt.title('Reference Grid')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reference.png'))
    plt.close()
    
    # Display reference WCS info if available
    if ref_wcs_info:
        print(f"Reference CRVAL: ({ref_wcs_info['CRVAL1']}, {ref_wcs_info['CRVAL2']})")
    
    # Process each non-reference file
    for i, fits_file in enumerate(fits_files[1:], 1):
        file_basename = os.path.basename(fits_file)
        print(f"Processing {file_basename}")
        
        # Get matrix and WCS info
        matrix, quality, wcs_info = get_matrix_and_wcs_from_fits(fits_file)
        
        if matrix is None:
            print(f"No matrix found in {file_basename}")
            continue
        
        # Calculate WCS offset if available
        wcs_dx, wcs_dy, wcs_sep = None, None, None
        if ref_wcs_info and wcs_info:
            wcs_dx, wcs_dy, wcs_sep = calculate_wcs_offset(ref_wcs_info, wcs_info)
            print(f"WCS offset: ({wcs_dx:.2f}, {wcs_dy:.2f}) pixels, {wcs_sep:.2f} arcsec")
        
        # Load the image
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data
            if len(image_data.shape) > 2:
                image_data = image_data[0]
        
        # Now we need to calculate the transformation relative to the reference
        # Since each matrix transforms from the frame to the world frame,
        # we need: frame2 -> world -> frame1
        # Which is: inv(ref_matrix) @ frame_matrix
        try:
            ref_matrix_inv = np.linalg.inv(ref_matrix)
            combined_matrix = np.dot(ref_matrix_inv, matrix)
            
            # Extract translation and rotation
            tx, ty = extract_translation(combined_matrix)
            rotation = extract_rotation(combined_matrix)
            
            print(f"Translation: ({tx:.2f}, {ty:.2f}) pixels, Rotation: {rotation:.2f} degrees")
            
            # Apply the transformation
            transformed_image = apply_transformation(
                image_data, combined_matrix, output_shape=output_shape,
                rotate_180=rotate_180, order=interpolation_order
            )
            
            # Also transform the grid
            transformed_grid = apply_transformation(
                grid_image, combined_matrix, output_shape=output_shape,
                rotate_180=rotate_180, order=interpolation_order
            )
            
            # Create visualizations
            # 1. Difference image (red/blue overlay)
            diff_image = np.zeros((padded_h, padded_w, 3))
            # Normalize images for visualization
            ref_norm = ref_centered / np.max(ref_centered) if np.max(ref_centered) > 0 else ref_centered
            img_norm = transformed_image / np.max(transformed_image) if np.max(transformed_image) > 0 else transformed_image
            diff_image[:, :, 0] = ref_norm  # Red channel = reference
            diff_image[:, :, 2] = img_norm  # Blue channel = transformed
            
            # Calculate metrics
            mse = np.mean((ref_norm - img_norm) ** 2)
            max_diff = np.max(np.abs(ref_norm - img_norm))
            
            # 2. Image comparison
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)
            plt.imshow(ref_centered, cmap='gray', origin='lower', 
                      norm=plt.Normalize(*np.percentile(ref_centered[ref_centered > 0], [1, 99])))
            plt.title('Reference Image')
            
            plt.subplot(2, 3, 2)
            plt.imshow(transformed_image, cmap='gray', origin='lower',
                      norm=plt.Normalize(*np.percentile(transformed_image[transformed_image > 0], [1, 99])))
            plt.title(f'Transformed Image {i}')
            
            plt.subplot(2, 3, 3)
            plt.imshow(diff_image, origin='lower')
            plt.title('Difference (Red=Ref, Blue=Trans)')
            
            plt.subplot(2, 3, 4)
            plt.imshow(grid_centered, cmap='gray', origin='lower')
            plt.title('Reference Grid')
            
            plt.subplot(2, 3, 5)
            plt.imshow(transformed_grid, cmap='gray', origin='lower')
            plt.title(f'Transformed Grid {i}')
            
            # Plot metrics
            ax = plt.subplot(2, 3, 6)
            ax.text(0.5, 0.9, f"Filename: {file_basename}", ha='center', va='center', fontsize=10)
            ax.text(0.5, 0.8, f"Translation: ({tx:.2f}, {ty:.2f}) pixels", ha='center', va='center', fontsize=10)
            ax.text(0.5, 0.7, f"Rotation: {rotation:.2f} degrees", ha='center', va='center', fontsize=10)
            
            # Add WCS info if available
            if wcs_info:
                ax.text(0.5, 0.6, f"CRVAL: ({wcs_info['CRVAL1']:.6f}, {wcs_info['CRVAL2']:.6f})", ha='center', va='center', fontsize=10)
                if wcs_dx is not None and wcs_dy is not None:
                    ax.text(0.5, 0.5, f"WCS offset: ({wcs_dx:.2f}, {wcs_dy:.2f}) px", ha='center', va='center', fontsize=10)
                    ax.text(0.5, 0.4, f"WCS separation: {wcs_sep:.2f} arcsec", ha='center', va='center', fontsize=10)
                    
                    # Calculate correlation between transformation and WCS
                    tx_corr = np.sign(tx) == np.sign(wcs_dx)
                    ty_corr = np.sign(ty) == np.sign(wcs_dy)
                    tx_ratio = abs(tx / wcs_dx) if wcs_dx != 0 else float('inf')
                    ty_ratio = abs(ty / wcs_dy) if wcs_dy != 0 else float('inf')
                    
                    ax.text(0.5, 0.3, f"TX/WCS-X ratio: {tx_ratio:.2f}", ha='center', va='center', fontsize=10)
                    ax.text(0.5, 0.2, f"TY/WCS-Y ratio: {ty_ratio:.2f}", ha='center', va='center', fontsize=10)
            
            ax.text(0.5, 0.1, f"MSE: {mse:.6f}", ha='center', va='center', fontsize=10)
            ax.axis('off')
            
            plt.suptitle(f'Transformation Validation: {file_basename}', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(output_dir, f'validation_{i:03d}.png'))
            plt.close()
            
            # Store results
            result = {
                'file': file_basename,
                'tx': tx,
                'ty': ty,
                'rotation': rotation,
                'stars': quality['stars'],
                'roundness': quality['roundness'],
                'index': quality['index'],
                'mse': mse,
                'max_diff': max_diff
            }
            
            # Add WCS info if available
            if wcs_info:
                result.update({
                    'crval1': wcs_info['CRVAL1'],
                    'crval2': wcs_info['CRVAL2'],
                    'wcs_dx': wcs_dx,
                    'wcs_dy': wcs_dy,
                    'wcs_sep': wcs_sep
                })
                
                if wcs_dx is not None and wcs_dy is not None:
                    result.update({
                        'tx_wcs_ratio': abs(tx / wcs_dx) if wcs_dx != 0 else float('inf'),
                        'ty_wcs_ratio': abs(ty / wcs_dy) if wcs_dy != 0 else float('inf'),
                    })
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {file_basename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if results:
        with open(os.path.join(output_dir, 'validation_results.txt'), 'w') as f:
            f.write("Transformation Validation Results\n")
            f.write("===============================\n\n")
            f.write(f"Reference file: {os.path.basename(ref_file)}\n")
            if ref_wcs_info:
                f.write(f"Reference CRVAL: ({ref_wcs_info['CRVAL1']}, {ref_wcs_info['CRVAL2']})\n\n")
            else:
                f.write("\n")
            
            # Check if we have WCS info
            has_wcs = any('crval1' in r for r in results)
            
            if has_wcs:
                f.write("File                    | TX       | TY       | Rotation | WCS-X    | WCS-Y    | TX/WCS-X | TY/WCS-Y | WCS Sep   | MSE        \n")
                f.write("------------------------|----------|----------|----------|----------|----------|----------|----------|-----------|------------\n")
                
                for r in results:
                    wcs_dx_str = f"{r.get('wcs_dx', 'N/A'):8.2f}" if r.get('wcs_dx') is not None else "    N/A  "
                    wcs_dy_str = f"{r.get('wcs_dy', 'N/A'):8.2f}" if r.get('wcs_dy') is not None else "    N/A  "
                    tx_ratio_str = f"{r.get('tx_wcs_ratio', 'N/A'):8.2f}" if r.get('tx_wcs_ratio') is not None else "    N/A  "
                    ty_ratio_str = f"{r.get('ty_wcs_ratio', 'N/A'):8.2f}" if r.get('ty_wcs_ratio') is not None else "    N/A  "
                    wcs_sep_str = f"{r.get('wcs_sep', 'N/A'):9.2f}" if r.get('wcs_sep') is not None else "    N/A    "
                    
                    f.write(f"{r['file']:<24} | {r['tx']:8.2f} | {r['ty']:8.2f} | {r['rotation']:8.2f} | {wcs_dx_str} | {wcs_dy_str} | {tx_ratio_str} | {ty_ratio_str} | {wcs_sep_str} | {r['mse']:10.6f}\n")
            else:
                f.write("File                    | TX       | TY       | Rotation | Stars | Roundness | MSE        | Max Diff\n")
                f.write("------------------------|----------|----------|----------|-------|-----------|------------|----------\n")
                
                for r in results:
                    f.write(f"{r['file']:<24} | {r['tx']:8.2f} | {r['ty']:8.2f} | {r['rotation']:8.2f} | {r['stars']:5d} | {r['roundness']:9.3f} | {r['mse']:10.6f} | {r['max_diff']:8.6f}\n")
        
        print(f"Saved validation results to {os.path.join(output_dir, 'validation_results.txt')}")
        
        # Create summary plots
        plt.figure(figsize=(10, 8))
        tx_values = [r['tx'] for r in results]
        ty_values = [r['ty'] for r in results]
        plt.scatter(tx_values, ty_values, c=range(len(results)), cmap='viridis')
        for i, r in enumerate(results):
            plt.annotate(str(i+1), (r['tx'], r['ty']), fontsize=8)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Translation Components from Transformation Matrices')
        plt.xlabel('X Translation (pixels)')
        plt.ylabel('Y Translation (pixels)')
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='Frame Number')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'translation_summary.png'))
        plt.close()
        
        # Create WCS comparison plot if WCS data is available
        if has_wcs and any(r.get('wcs_dx') is not None for r in results):
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Matrix vs WCS X offset
            plt.subplot(2, 2, 1)
            valid_results = [r for r in results if r.get('wcs_dx') is not None]
            x = [r['wcs_dx'] for r in valid_results]
            y = [r['tx'] for r in valid_results]
            plt.scatter(x, y)
            for i, r in enumerate(valid_results):
                plt.annotate(str(i+1), (r['wcs_dx'], r['tx']), fontsize=8)
            
            # Add best fit line
            if len(x) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.array([min(x), max(x)])
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, 'r-', alpha=0.7)
                plt.text(0.05, 0.95, f"y = {slope:.4f}x + {intercept:.4f}", transform=plt.gca().transAxes)
            
            plt.xlabel('WCS X Offset (pixels)')
            plt.ylabel('Matrix TX (pixels)')
            plt.title('Matrix TX vs WCS X Offset')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Matrix vs WCS Y offset
            plt.subplot(2, 2, 2)
            x = [r['wcs_dy'] for r in valid_results]
            y = [r['ty'] for r in valid_results]
            plt.scatter(x, y)
            for i, r in enumerate(valid_results):
                plt.annotate(str(i+1), (r['wcs_dy'], r['ty']), fontsize=8)
            
            # Add best fit line
            if len(x) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.array([min(x), max(x)])
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, 'r-', alpha=0.7)
                plt.text(0.05, 0.95, f"y = {slope:.4f}x + {intercept:.4f}", transform=plt.gca().transAxes)
            
            plt.xlabel('WCS Y Offset (pixels)')
            plt.ylabel('Matrix TY (pixels)')
            plt.title('Matrix TY vs WCS Y Offset')
            plt.grid(True, alpha=0.3)
            
            # Plot 3: WCS Offset X vs Y
            plt.subplot(2, 2, 3)
            x = [r['wcs_dx'] for r in valid_results]
            y = [r['wcs_dy'] for r in valid_results]
            plt.scatter(x, y)
            for i, r in enumerate(valid_results):
                plt.annotate(str(i+1), (r['wcs_dx'], r['wcs_dy']), fontsize=8)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
            plt.xlabel('WCS X Offset (pixels)')
            plt.ylabel('WCS Y Offset (pixels)')
            plt.title('WCS X vs Y Offset')
            plt.grid(True, alpha=0.3)
            
            # Plot 4: WCS separation vs Matrix distance
            plt.subplot(2, 2, 4)
            x = [r['wcs_sep'] for r in valid_results]
            y = [np.sqrt(r['tx']**2 + r['ty']**2) for r in valid_results]
            plt.scatter(x, y)
            for i, r in enumerate(valid_results):
                plt.annotate(str(i+1), (r['wcs_sep'], np.sqrt(r['tx']**2 + r['ty']**2)), fontsize=8)
            
            # Add best fit line
            if len(x) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.array([min(x), max(x)])
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, 'r-', alpha=0.7)
                plt.text(0.05, 0.95, f"y = {slope:.4f}x + {intercept:.4f}", transform=plt.gca().transAxes)
            
            plt.xlabel('WCS Separation (arcsec)')
            plt.ylabel('Matrix Distance (pixels)')
            plt.title('WCS Separation vs Matrix Distance')
            plt.grid(True, alpha=0.3)
            
            plt.suptitle('Correlation between Transformation Matrices and WCS Changes', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(os.path.join(output_dir, 'wcs_correlation.png'))
            plt.close()
            
            # Create a plot of CRVAL changes
            if has_wcs and any(r.get('crval1') is not None for r in results):
                plt.figure(figsize=(12, 8))
                
                # Extract CRVAL values
                crval1_values = [r.get('crval1') for r in results if r.get('crval1') is not None]
                crval2_values = [r.get('crval2') for r in results if r.get('crval2') is not None]
                
                if crval1_values and crval2_values:
                    plt.scatter(crval1_values, crval2_values, c=range(len(crval1_values)), cmap='viridis')
                    
                    # Connect points with lines to show progression
                    plt.plot(crval1_values, crval2_values, 'k-', alpha=0.3)
                    
                    # Add labels
                    for i, (x, y) in enumerate(zip(crval1_values, crval2_values)):
                        plt.annotate(str(i+1), (x, y), fontsize=8)
                    
                    # Add reference CRVAL if available
                    if ref_wcs_info:
                        plt.scatter([ref_wcs_info['CRVAL1']], [ref_wcs_info['CRVAL2']], 
                                   c='red', marker='*', s=100, label='Reference')
                    
                    plt.xlabel('CRVAL1 (degrees)')
                    plt.ylabel('CRVAL2 (degrees)')
                    plt.title('CRVAL Position Changes Across Files')
                    plt.grid(True, alpha=0.3)
                    if ref_wcs_info:
                        plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'crval_changes.png'))
                    plt.close()
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Validate transformation matrices in FITS headers and correlate with CRVAL changes')
    parser.add_argument('fits_directory', help='Directory containing FITS files')
    parser.add_argument('--output', '-o', default='matrix_validation', help='Output directory')
    parser.add_argument('--first_n', type=int, default=10, help='Process only first N files (default: 10)')
    parser.add_argument('--no-rotation', action='store_true', help='Disable 180-degree rotation')
    parser.add_argument('--interpolation', '-i', type=int, choices=[0, 1, 3, 5], default=1,
                      help='Interpolation order: 0=nearest, 1=linear, 3=cubic (default: 1)')
    parser.add_argument('--margin', type=int, default=10, help='Margin percentage (default: 10)')
    parser.add_argument('--ref-file', help='Specific file to use as reference (optional)')
    parser.add_argument('--wcs-only', action='store_true', help='Only check files with valid WCS information')
    
    args = parser.parse_args()
    
    # Find all FITS files in the directory
    fits_files = sorted(glob.glob(os.path.join(args.fits_directory, "*.fits")))
    
    if not fits_files:
        print(f"No FITS files found in {args.fits_directory}")
        return
    
    print(f"Found {len(fits_files)} FITS files")
    
    # Filter for files with matrices and optionally WCS info
    files_with_matrices = []
    wcs_files = []
    for fits_file in tqdm(fits_files):
        matrix, _, wcs_info = get_matrix_and_wcs_from_fits(fits_file)
        if matrix is not None:
            files_with_matrices.append(fits_file)
            if wcs_info is not None:
                wcs_files.append(fits_file)
    
    if not files_with_matrices:
        print("No files with matrices found. Cannot proceed.")
        return
    
    print(f"Found {len(files_with_matrices)} files with transformation matrices")
    print(f"Found {len(wcs_files)} files with WCS information")
    
    # Use the files with WCS if requested
    if args.wcs_only and wcs_files:
        print("Using only files with WCS information")
        filtered_files = wcs_files
    else:
        filtered_files = files_with_matrices
    
    # Use specified reference file if provided
    if args.ref_file:
        if os.path.exists(args.ref_file):
            ref_file = args.ref_file
            matrix, _, wcs_info = get_matrix_and_wcs_from_fits(ref_file)
            if matrix is None:
                print(f"Specified reference file {ref_file} has no transformation matrix. Using default.")
                ref_file = None
            elif args.wcs_only and wcs_info is None:
                print(f"Specified reference file {ref_file} has no WCS information. Using default.")
                ref_file = None
            else:
                print(f"Using specified reference file: {ref_file}")
                # Remove reference file from the list if it's there
                if ref_file in filtered_files:
                    filtered_files.remove(ref_file)
                # Add it at the beginning
                filtered_files.insert(0, ref_file)
        else:
            print(f"Specified reference file {args.ref_file} not found. Using default.")
    
    # Limit to first N files
    if args.first_n and args.first_n < len(filtered_files):
        print(f"Limiting to first {args.first_n} files")
        filtered_files = filtered_files[:args.first_n]
    
    # Validate matrices
    results = validate_matrices(
        filtered_files,
        args.output,
        rotate_180=not args.no_rotation,
        interpolation_order=args.interpolation,
        margin_percent=args.margin
    )
    
    # Generate HTML report if results were obtained
    if results:
        generate_html_report(args.output, results, filtered_files[0])

if __name__ == "__main__":
    main()
