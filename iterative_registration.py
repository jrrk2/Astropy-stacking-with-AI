#!/usr/bin/env python3
"""
Iterative Image Registration Tool for Faint Astronomical Images

This script implements a robust registration approach for faint astronomical
images where traditional plate solving may fail. It uses a combination of
ALT/AZ data with systematic corrections and multi-scale FFT-based registration.

Usage:
    python iterative_registration.py --input 'lights/temp_290/*.fits' --output registered_output

Author: Claude
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage
from skimage.transform import resize, SimilarityTransform, warp
from datetime import datetime
import json
import logging
import ephem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("registration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Iterative image registration for faint astronomical images')
    parser.add_argument('--input', required=True, help='Input pattern for FITS files (e.g. "lights/*.fits")')
    parser.add_argument('--json_dir', help='Directory containing matching JSON files (optional)')
    parser.add_argument('--output', default='registered', help='Output directory')
    parser.add_argument('--reference', help='Index of reference frame (default: highest quality frame)')
    parser.add_argument('--lat', type=float, default=52.2, help='Observer latitude')
    parser.add_argument('--lon', type=float, default=0.12, help='Observer longitude')
    parser.add_argument('--scales', type=str, default='1.0,0.5,0.25', help='Scale factors for multi-scale approach')
    parser.add_argument('--iterations', type=int, default=3, help='Max iterations per scale')
    parser.add_argument('--diagnostic', action='store_true', help='Generate diagnostic visualizations')
    return parser.parse_args()

def load_fits_data(file_path):
    """Load FITS data and header"""
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    return data, header

def find_json_for_fits(fits_path, json_dir=None):
    """Find matching JSON file for a FITS file"""
    if not json_dir:
        # Try same directory as FITS
        json_dir = os.path.dirname(fits_path)
    
    fits_basename = os.path.basename(fits_path)
    # Extract index pattern (e.g., 0001 from img-0001.fits)
    import re
    match = re.search(r'[^0-9](\d+)[^0-9]', fits_basename)
    if not match:
        return None
    
    index = match.group(1)
    # Try to find matching JSON with same index
    json_pattern = os.path.join(json_dir, f"*{index}*.json")
    json_files = glob.glob(json_pattern)
    
    if json_files:
        return json_files[0]
    
    return None

def load_json_data(json_path):
    """Load and parse JSON file"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def extract_alt_az(json_data):
    """Extract altitude and azimuth from JSON data"""
    try:
        # Try common locations in JSON structure
        if 'motors' in json_data:
            return json_data['motors'].get('ALT'), json_data['motors'].get('AZ')
        elif 'data' in json_data and 'acqResult' in json_data['data']:
            return json_data['data']['acqResult']['motors'].get('ALT'), json_data['data']['acqResult']['motors'].get('AZ')
        elif 'result' in json_data and 'acqResult' in json_data['result']:
            return json_data['result']['acqResult']['motors'].get('ALT'), json_data['result']['acqResult']['motors'].get('AZ')
    except Exception:
        pass
    
    return None, None

def alt_az_to_radec(alt, az, date_obs, lat, lon):
    """Convert altitude and azimuth to RA/Dec with observer location"""
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.elevation = 20  # meters, typical
    
    # Convert ISO format to ephem date format
    try:
        dt = datetime.strptime(date_obs, '%Y-%m-%dT%H:%M:%S')
        observer.date = ephem.Date(dt)
        
        ra, dec = observer.radec_of(az * np.pi/180, alt * np.pi/180)
        return float(ra) * 180/np.pi, float(dec) * 180/np.pi
    except Exception as e:
        logger.error(f"Error converting coordinates: {str(e)}")
        return None, None

def apply_altaz_correction(ra, dec, alt):
    """Apply systematic correction based on altitude"""
    # Apply correction based on altitude
    if alt < 25:
        ra_correction = -36.0 / 3600.0  # -36 arcsec in degrees
        dec_correction = -74.0 / 3600.0  # -74 arcsec in degrees
    elif alt < 30:
        ra_correction = -34.0 / 3600.0
        dec_correction = -68.0 / 3600.0
    else:
        ra_correction = -32.0 / 3600.0
        dec_correction = -64.0 / 3600.0
    
    # Apply corrections
    ra_corrected = ra + ra_correction
    dec_corrected = dec + dec_correction
    
    return ra_corrected, dec_corrected

def preprocess_image(image):
    """Preprocess image to enhance features for registration"""
    # Remove background
    bkg = ndimage.median_filter(image, size=50)
    img_nobkg = image - bkg
    
    # Handle negative values and normalize
    img_nobkg[img_nobkg < 0] = 0
    
    # Enhance contrast with gamma correction
    gamma = 0.7
    img_enhanced = np.power(img_nobkg / (np.percentile(img_nobkg, 99) + 1e-10), gamma)
    
    # Apply mild gaussian smoothing to reduce noise
    img_smooth = ndimage.gaussian_filter(img_enhanced, sigma=1.0)
    
    return img_smooth

def align_fft(image, reference):
    """Align image to reference using FFT phase correlation"""
    from scipy.signal import fftconvolve
    
    # Apply window function to reduce edge effects
    y, x = np.indices(reference.shape)
    window = np.outer(np.hanning(reference.shape[0]), np.hanning(reference.shape[1]))
    
    # Apply window to both images
    ref_windowed = reference * window
    img_windowed = image * window
    
    # Compute FFT of both images
    ref_fft = np.fft.fft2(ref_windowed)
    img_fft = np.fft.fft2(img_windowed)
    
    # Compute cross-power spectrum
    cross_power = np.fft.ifft2(ref_fft * np.conj(img_fft))
    abs_cross_power = np.abs(cross_power)
    
    # Find the peak in the cross-power
    max_y, max_x = np.unravel_index(np.argmax(abs_cross_power), abs_cross_power.shape)
    
    # Convert to centered shifts
    shift_y = max_y if max_y < reference.shape[0]//2 else max_y - reference.shape[0]
    shift_x = max_x if max_x < reference.shape[1]//2 else max_x - reference.shape[1]
    
    # Calculate confidence metrics
    peak_val = abs_cross_power[max_y, max_x]
    mean_val = np.mean(abs_cross_power)
    std_val = np.std(abs_cross_power)
    
    peak_significance = (peak_val - mean_val) / std_val
    confidence = min(100.0, max(0.0, (peak_significance - 3.0) * 20.0))
    
    # Sub-pixel refinement
    try:
        # Extract 3x3 region around peak
        if 0 < max_y < abs_cross_power.shape[0]-1 and 0 < max_x < abs_cross_power.shape[1]-1:
            y_range = slice(max(0, max_y-1), min(abs_cross_power.shape[0], max_y+2))
            x_range = slice(max(0, max_x-1), min(abs_cross_power.shape[1], max_x+2))
            region = abs_cross_power[y_range, x_range]
            
            # Calculate centroid in this region
            y_indices, x_indices = np.indices(region.shape)
            y_centroid = np.sum(y_indices * region) / np.sum(region)
            x_centroid = np.sum(x_indices * region) / np.sum(region)
            
            # Adjust shift by centroid offset from center of extracted region
            shift_y += (y_centroid - 1)  # -1 because the center of a 3x3 is at index 1
            shift_x += (x_centroid - 1)
    except Exception:
        # Fall back to integer shift if refinement fails
        pass
    
    # Metadata for diagnostics
    metadata = {
        'peak_val': float(peak_val),
        'mean_val': float(mean_val),
        'std_val': float(std_val),
        'significance': float(peak_significance),
        'confidence': float(confidence)
    }
    
    return -shift_y, -shift_x, metadata

def evaluate_registration(reference, registered):
    """Evaluate registration quality"""
    # Calculate MSE
    mse = np.mean((reference - registered)**2)
    
    # Calculate NCC
    from scipy.signal import correlate2d
    corr = correlate2d(reference, registered, mode='same')
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = corr[peak_y, peak_x]
    mean_val = np.mean(corr)
    std_val = np.std(corr)
    
    peak_significance = (peak_val - mean_val) / std_val
    ncc_score = min(100.0, max(0.0, (peak_significance - 3.0) * 20.0))
    
    return {
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'ncc_score': float(ncc_score),
        'peak_offset': (float(peak_y - reference.shape[0]//2), 
                        float(peak_x - reference.shape[1]//2))
    }

def iterative_register_image(image, reference, scale_factors, max_iterations):
    """Register image to reference using multi-scale iterative approach"""
    # Initialize transform (identity)
    transform = SimilarityTransform()
    
    # Start with preprocessed images
    ref_processed = preprocess_image(reference)
    img_processed = preprocess_image(image)
    
    # Multi-scale registration
    for i, scale in enumerate(scale_factors):
        logger.info(f"Processing scale factor: {scale}")
        
        # Resize images for current scale
        if scale < 1.0:
            ref_scaled = resize(ref_processed, 
                              (int(ref_processed.shape[0] * scale), 
                               int(ref_processed.shape[1] * scale)))
            img_scaled = resize(img_processed, 
                              (int(img_processed.shape[0] * scale), 
                               int(img_processed.shape[1] * scale)))
        else:
            ref_scaled = ref_processed
            img_scaled = img_processed
        
        # Iterative refinement at current scale
        for iteration in range(max_iterations):
            # Apply current transform to get intermediate result
            warped = warp(img_scaled, transform.inverse)
            
            # Calculate correction using FFT alignment
            dy, dx, metadata = align_fft(warped, ref_scaled)
            
            # If shift is very small, consider converged
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                logger.info(f"  Iteration {iteration+1}: Converged (shifts < 0.5 pixel)")
                break
            
            # Scale shifts back to original image size
            dx_full = dx / scale
            dy_full = dy / scale
            
            # Update transform with this iteration's correction
            correction = SimilarityTransform(translation=(-dx, -dy))
            transform = transform + correction
            
            logger.info(f"  Iteration {iteration+1}: "
                       f"Shift ({dx:.2f}, {dy:.2f}), "
                       f"Confidence {metadata['confidence']:.1f}%")
            
            # Check confidence - if too low, registration might be failing
            if metadata['confidence'] < 30 and i > 0:
                logger.warning(f"  Low confidence ({metadata['confidence']:.1f}%). "
                              f"Registration may be failing.")
    
    # Apply final transform to original image
    registered_image = warp(image, transform.inverse)
    
    # Evaluate final registration quality
    quality = evaluate_registration(reference, registered_image)
    logger.info(f"Final registration quality: "
               f"NCC: {quality['ncc_score']:.1f}%, "
               f"RMSE: {quality['rmse']:.6f}")
    
    return registered_image, transform, quality

def create_diagnostic_plot(reference, registered, original, title="Registration Diagnostic"):
    """Create diagnostic visualization"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create custom diverging colormap for difference image
    diff_cmap = LinearSegmentedColormap.from_list('diff_cmap', 
                                                 ['blue', 'black', 'red'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Normalize images for display
    def normalize_for_display(img):
        vmin, vmax = np.percentile(img, (1, 99))
        img_norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        return img_norm
    
    ref_norm = normalize_for_display(reference)
    reg_norm = normalize_for_display(registered)
    orig_norm = normalize_for_display(original)
    
    # Display reference image
    axes[0, 0].imshow(ref_norm, cmap='viridis')
    axes[0, 0].set_title("Reference Image")
    
    # Display registered image
    axes[0, 1].imshow(reg_norm, cmap='viridis')
    axes[0, 1].set_title("Registered Image")
    
    # Calculate and display difference image
    diff = reference - registered
    diff_vmax = np.percentile(np.abs(diff), 99)
    axes[1, 0].imshow(diff, cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)
    axes[1, 0].set_title("Difference (Reference - Registered)")
    
    # Show original image
    axes[1, 1].imshow(orig_norm, cmap='viridis')
    axes[1, 1].set_title("Original (Before Registration)")
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Find input files
    input_files = sorted(glob.glob(args.input))
    if not input_files:
        logger.error(f"No files found matching pattern: {args.input}")
        return
    
    logger.info(f"Found {len(input_files)} files")
    
    # Load all files
    frames = []
    headers = []
    for file_path in input_files:
        try:
            data, header = load_fits_data(file_path)
            frames.append(data)
            headers.append(header)
            logger.info(f"Loaded: {file_path}, shape: {data.shape}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    if not frames:
        logger.error("No valid frames loaded.")
        return
    
    # Determine reference frame
    if args.reference is not None:
        ref_idx = int(args.reference)
    else:
        # Default to first frame
        ref_idx = 0
    
    reference_frame = frames[ref_idx]
    reference_header = headers[ref_idx]
    
    logger.info(f"Using frame {ref_idx} as reference: {input_files[ref_idx]}")
    
    # Parse scale factors
    scale_factors = [float(s) for s in args.scales.split(',')]
    scale_factors = sorted(scale_factors, reverse=True)  # Largest first
    
    # Process each frame
    registered_frames = []
    transforms = []
    qualities = []
    
    for i, (frame, header, file_path) in enumerate(zip(frames, headers, input_files)):
        logger.info(f"Processing frame {i+1}/{len(frames)}: {file_path}")
        
        if i == ref_idx:
            # Reference frame needs no registration
            registered_frames.append(frame)
            transforms.append(SimilarityTransform())
            qualities.append({'mse': 0, 'rmse': 0, 'ncc_score': 100})
            logger.info("This is the reference frame (no registration needed)")
            continue
        
        # Try to use ALT/AZ data if available
        json_path = find_json_for_fits(file_path, args.json_dir)
        if json_path:
            logger.info(f"Found matching JSON: {json_path}")
            json_data = load_json_data(json_path)
            
            if json_data:
                alt, az = extract_alt_az(json_data)
                date_obs = header.get('DATE-OBS')
                
                if alt is not None and az is not None and date_obs:
                    logger.info(f"Using ALT/AZ data: ALT={alt}, AZ={az}")
                    
                    # Calculate RA/Dec with correction
                    ra, dec = alt_az_to_radec(alt, az, date_obs, args.lat, args.lon)
                    
                    if ra is not None and dec is not None:
                        ra_corr, dec_corr = apply_altaz_correction(ra, dec, alt)
                        logger.info(f"Calculated coordinates: "
                                   f"RA={ra:.6f}→{ra_corr:.6f}, "
                                   f"Dec={dec:.6f}→{dec_corr:.6f}")
        
        # Perform iterative registration
        registered, transform, quality = iterative_register_image(
            frame, reference_frame, scale_factors, args.iterations
        )
        
        registered_frames.append(registered)
        transforms.append(transform)
        qualities.append(quality)
        
        # Create and save diagnostic visualization
        if args.diagnostic:
            fig = create_diagnostic_plot(
                reference_frame, registered, frame,
                title=f"Registration Results - Frame {i+1}"
            )
            diag_path = os.path.join(args.output, f"diagnostic_{i+1:04d}.png")
            fig.savefig(diag_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved diagnostic image: {diag_path}")
        
        # Save registered frame
        output_fits = os.path.join(args.output, f"registered_{i+1:04d}.fits")
        hdu = fits.PrimaryHDU(data=registered, header=header)
        
        # Add registration metadata to header
        hdu.header['REG_MSE'] = (quality['rmse'], 'Registration RMSE')
        hdu.header['REG_NCC'] = (quality['ncc_score'], 'Registration NCC score (0-100)')
        hdu.header['REG_FILE'] = (os.path.basename(file_path), 'Original file')
        
        # Add transform parameters
        params = transform.params.flatten()
        for j, param in enumerate(params):
            hdu.header[f'XFORM_P{j}'] = (param, f'Transform parameter {j}')
        
        hdu.writeto(output_fits, overwrite=True)
        logger.info(f"Saved registered frame: {output_fits}")
    
    # Create and save quality summary
    quality_summary = {
        'filenames': [os.path.basename(f) for f in input_files],
        'rmse': [q['rmse'] for q in qualities],
        'ncc_score': [q['ncc_score'] for q in qualities]
    }
    
    import pandas as pd
    summary_df = pd.DataFrame(quality_summary)
    csv_path = os.path.join(args.output, 'registration_quality.csv')
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved quality summary: {csv_path}")
    
    logger.info("Registration complete!")

if __name__ == "__main__":
    main()