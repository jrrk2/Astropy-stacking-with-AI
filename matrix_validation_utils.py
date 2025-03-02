

def generate_html_report(output_dir, results, ref_file):
    """Generate an HTML report summarizing the validation results"""
    html_file = os.path.join(output_dir, 'validation_report.html')
    
    # Check if we have WCS info
    has_wcs = any('crval1' in r for r in results)
    
    with open(html_file, 'w') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FITS Transformation Matrix Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
        h1, h2, h3 { color: #2c3e50; }
        .container { max-width: 1200px; margin: 0 auto; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .gallery { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
        .gallery img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        .summary-images { display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin-top: 20px; }
        .summary-images img { max-width: 45%; height: auto; border: 1px solid #ddd; }
        .image-container { margin-bottom: 30px; }
        .warning { color: #e74c3c; }
        .success { color: #2ecc71; }
    </style>
</head>
<body>
    <div class="container">
        <h1>FITS Transformation Matrix Validation Report</h1>
        <p>This report shows the validation results of transformation matrices in FITS headers.</p>
        
        <h2>Summary</h2>
        <p>Reference file: <strong>""")
        
        f.write(os.path.basename(ref_file))
        
        f.write("""</strong></p>
        <p>Total files processed: <strong>""")
        
        f.write(str(len(results) + 1))  # +1 for reference
        
        f.write("""</strong></p>
        
        <h2>Results Table</h2>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>TX (px)</th>
                    <th>TY (px)</th>
                    <th>Rotation (°)</th>""")
        
        if has_wcs:
            f.write("""
                    <th>CRVAL1</th>
                    <th>CRVAL2</th>
                    <th>WCS-X (px)</th>
                    <th>WCS-Y (px)</th>
                    <th>WCS Sep (arcsec)</th>""")
            
        f.write("""
                    <th>Stars</th>
                    <th>MSE</th>
                </tr>
            </thead>
            <tbody>""")
        
        for r in results:
            f.write(f"""
                <tr>
                    <td>{r['file']}</td>
                    <td>{r['tx']:.2f}</td>
                    <td>{r['ty']:.2f}</td>
                    <td>{r['rotation']:.2f}</td>""")
            
            if has_wcs:
                crval1 = r.get('crval1', 'N/A')
                crval2 = r.get('crval2', 'N/A')
                wcs_dx = r.get('wcs_dx', 'N/A')
                wcs_dy = r.get('wcs_dy', 'N/A')
                wcs_sep = r.get('wcs_sep', 'N/A')
                
                if crval1 != 'N/A':
                    crval1 = f"{crval1:.6f}"
                if crval2 != 'N/A':
                    crval2 = f"{crval2:.6f}"
                if wcs_dx != 'N/A':
                    wcs_dx = f"{wcs_dx:.2f}"
                if wcs_dy != 'N/A':
                    wcs_dy = f"{wcs_dy:.2f}"
                if wcs_sep != 'N/A':
                    wcs_sep = f"{wcs_sep:.2f}"
                
                f.write(f"""
                    <td>{crval1}</td>
                    <td>{crval2}</td>
                    <td>{wcs_dx}</td>
                    <td>{wcs_dy}</td>
                    <td>{wcs_sep}</td>""")
            
            f.write(f"""
                    <td>{r['stars']}</td>
                    <td>{r['mse']:.6f}</td>
                </tr>""")
        
        f.write("""
            </tbody>
        </table>
        
        <h2>Summary Visualizations</h2>
        <div class="summary-images">
            <img src="translation_summary.png" alt="Translation Summary">""")
        
        if has_wcs:
            f.write("""
            <img src="wcs_correlation.png" alt="WCS Correlation">
            <img src="crval_changes.png" alt="CRVAL Changes">""")
        
        f.write("""
        </div>
        
        <h2>Detailed Results</h2>
        <p>Click on an image to see a larger version.</p>
        
        <h3>Reference Image</h3>
        <div class="image-container">
            <img src="reference.png" alt="Reference Image">
        </div>
        
        <h3>Individual Validation Results</h3>
        <div class="gallery">""")
        
        # Add image tags for each validation result
        for i in range(1, len(results) + 1):
            f.write(f"""
            <img src="validation_{i:03d}.png" alt="Validation {i}">""")
        
        f.write("""
        </div>
    </div>
</body>
</html>""")
    
    print(f"Generated HTML report at {html_file}")
    return html_file

def apply_transformation(image, matrix, output_shape=None, rotate_180=False, order=1):
    """Apply a transformation matrix to an image"""
    h, w = image.shape
    
    # If no output shape is provided, use the input shape
    if output_shape is None:
        output_shape = (h, w)
    
    # Calculate translation to center the image in the padded canvas
    if output_shape[0] > h or output_shape[1] > w:
        # Calculate the center shift
        y_shift = (output_shape[0] - h) / 2
        x_shift = (output_shape[1] - w) / 2
        
        # Create a translation matrix to center the image
        center_matrix = np.array([
            [1, 0, x_shift],
            [0, 1, y_shift],
            [0, 0, 1]
        ])
        
        # Apply centering first
        matrix = np.dot(center_matrix, matrix)
    
    # Apply 180 degree rotation if needed
    if rotate_180:
        # Create rotation matrix for 180 degrees (around the center of the output)
        rot_matrix = np.array([
            [-1, 0, output_shape[1]],  # Flip X and translate
            [0, -1, output_shape[0]],  # Flip Y and translate
            [0, 0, 1]
        ])
        
        # Combine with the existing transformation
        matrix = np.dot(rot_matrix, matrix)
    
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

def create_grid_image(shape, grid_spacing=100):
    """Create a grid image for visualizing transformations"""
    h, w = shape
    grid = np.zeros(shape, dtype=np.uint8)
    
    # Create grid lines
    for i in range(0, h, grid_spacing):
        grid[i, :] = 255
    for j in range(0, w, grid_spacing):
        grid[:, j] = 255
    
    # Add cross in the center
    center_h, center_w = h // 2, w // 2
    thickness = 3
    grid[center_h-thickness:center_h+thickness, :] = 255
    grid[:, center_w-thickness:center_w+thickness] = 255
    
    return grid

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

def extract_rotation(matrix):
    """Extract rotation angle from a transformation matrix"""
    if matrix is None:
        return None
    
    try:
        # For a similarity/affine transformation with rotation, we can derive the angle from the top-left 2x2 block
        # Assume the matrix is in the form:
        # [r11 r12 tx]
        # [r21 r22 ty]
        # [0   0   1 ]
        # where r11 = cos(theta), r12 = -sin(theta), r21 = sin(theta), r22 = cos(theta)
        
        theta = np.arctan2(matrix[1, 0], matrix[0, 0])
        return theta * 180 / np.pi  # Convert to degrees
    except:
        return None

def calculate_wcs_offset(ref_wcs_info, wcs_info):
    """Calculate the offset between two WCS coordinates in pixels"""
    if ref_wcs_info is None or wcs_info is None:
        return None, None, None
    
    try:
        # Create WCS objects
        ref_w = wcs.WCS(ref_wcs_info)
        curr_w = wcs.WCS(wcs_info)
        
        # Get image centers
        ref_crval = SkyCoord(ref_wcs_info['CRVAL1'], ref_wcs_info['CRVAL2'], unit='deg')
        curr_crval = SkyCoord(wcs_info['CRVAL1'], wcs_info['CRVAL2'], unit='deg')
        
        # Calculate separation in arcseconds
        sep = ref_crval.separation(curr_crval).arcsecond
        
        # Convert reference point to pixels in current frame
        ref_in_curr_pixels = curr_w.world_to_pixel(ref_crval)
        
        # Get the current frame's center in pixels
        curr_center_pixels = (wcs_info['CRPIX1'], wcs_info['CRPIX2'])
        
        # Calculate offset in pixels
        dx = ref_in_curr_pixels[0] - curr_center_pixels[0]
        dy = ref_in_curr_pixels[1] - curr_center_pixels[1]
        
        return dx, dy, sep
    except Exception as e:
        print(f"Error calculating WCS offset: {e}")
        return None, None, None#!/usr/bin/env python3
"""
Utility functions for validating transformation matrices in FITS headers
and correlating them with CRVAL1/2 changes.
"""

import os
import numpy as np
from astropy.io import fits
from skimage import transform
from astropy import wcs
from astropy.coordinates import SkyCoord
import astropy.units as u

def get_matrix_and_wcs_from_fits(fits_file):
    """Extract transformation matrix and WCS info from FITS header"""
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
                return None, None, None
            
            # Extract the matrix
            matrix = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    matrix[i, j] = header[f'{key_prefix}{i+1}{j+1}']
            
            # Get registration quality if available
            reg_quality = {
                'roundness': header.get('REGRNDS', 0),
                'stars': header.get('REGSTARS', 0),
                'index': header.get('REGINDX', -1)
            }
            
            # Get WCS information if available
            wcs_info = None
            if all(key in header for key in ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']):
                wcs_info = {
                    'CRVAL1': header.get('CRVAL1'),
                    'CRVAL2': header.get('CRVAL2'),
                    'CRPIX1': header.get('CRPIX1'),
                    'CRPIX2': header.get('CRPIX2'),
                    'CD1_1': header.get('CD1_1', 0),
                    'CD1_2': header.get('CD1_2', 0),
                    'CD2_1': header.get('CD2_1', 0),
                    'CD2_2': header.get('CD2_2', 0),
                    'CTYPE1': header.get('CTYPE1', ''),
                    'CTYPE2': header.get('CTYPE2', '')
                }
            
            return matrix, reg_quality, wcs_info
            
    except Exception as e:
        print(f"Error reading matrix from {fits_file}: {e}")
        return None, None, None