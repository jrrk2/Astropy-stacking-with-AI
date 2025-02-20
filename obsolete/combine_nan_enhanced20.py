from astropy.io import fits
import numpy as np
import cv2
import argparse
from scipy import interpolate
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io.votable import parse_single_table
from astroquery.vizier import Vizier
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Processing with Background Gradient Elimination and Photometric Calibration')
    parser.add_argument('--method', type=str, default='none', 
                        choices=['none', 'median', 'polynomial', 'local', 'wavelet'],
                        help='Background gradient elimination method')
    parser.add_argument('--catalog', type=str, default='apass',
                        choices=['apass', 'sdss', 'panstarrs'],
                        help='Star catalog to use for photometric calibration')
    parser.add_argument('--mag-limit', type=float, default=16.0,
                        help='Magnitude limit for reference stars')
    return parser.parse_args()

def get_catalog_stars(wcs, catalog='apass', mag_limit=16.0):
    """
    Query online star catalog for reference stars in the image field
    """
    # Get image corners
    naxis1, naxis2 = wcs.pixel_shape
    corners = np.array([[0, 0], [0, naxis2], [naxis1, 0], [naxis1, naxis2]])
    ra_dec_corners = wcs.all_pix2world(corners, 0)
    
    # Calculate field center and radius
    center_ra = np.mean(ra_dec_corners[:, 0])
    center_dec = np.mean(ra_dec_corners[:, 1])
    radius = np.max(np.sqrt(
        (ra_dec_corners[:, 0] - center_ra)**2 +
        (ra_dec_corners[:, 1] - center_dec)**2
    )) * u.deg
    
    center = SkyCoord(center_ra, center_dec, unit=(u.deg, u.deg))
    
    # Configure Vizier
    vizier = Vizier(columns=['*', '+_r'])  # Get all columns plus distance
    vizier.ROW_LIMIT = -1  # No row limit
    
    catalog_params = {
        'apass': {
            'catalog': "II/336/apass9",
            'magnitude_column': 'r_mag',
            'mag_limit_column': 'r_mag',
            'ra_column': 'RAJ2000',
            'dec_column': 'DEJ2000'
        },
        'sdss': {
            'catalog': "V/147/sdss12",
            'magnitude_column': 'rmag',
            'mag_limit_column': 'rmag',
            'ra_column': 'RA_ICRS',
            'dec_column': 'DE_ICRS'
        },
        'panstarrs': {
            'catalog': "II/349/ps1",
            'magnitude_column': 'rmag',
            'mag_limit_column': 'rmag',
            'ra_column': 'RAJ2000',
            'dec_column': 'DEJ2000'
        }
    }
    
    if catalog not in catalog_params:
        print(f"Unknown catalog {catalog}")
        return None
        
    params = catalog_params[catalog]
    vizier.column_filters[params['mag_limit_column']] = f"<{mag_limit}"
    
    try:
        catalog_query = vizier.query_region(
            center, 
            radius=radius,
            catalog=params['catalog']
        )
        
        if len(catalog_query) > 0 and len(catalog_query[0]) > 0:
            stars = []
            for row in catalog_query[0]:
                stars.append({
                    'ra': float(row[params['ra_column']]),
                    'dec': float(row[params['dec_column']]),
                    'magnitude_r': float(row[params['magnitude_column']])
                })
            print(f"Found {len(stars)} stars in {catalog} catalog")
            return stars
    except Exception as e:
        print(f"Error querying {catalog} catalog: {str(e)}")
        
    return None

def perform_photometric_calibration(image, wcs, reference_stars):
    """
    Perform photometric calibration using reference stars
    """
    # Extract star positions and fluxes
    star_positions = []
    star_mags = []
    measured_fluxes = []
    
    for star in reference_stars:
        # Convert sky coordinates to pixel coordinates
        coord = SkyCoord(star['ra'], star['dec'], unit=(u.deg, u.deg))
        x, y = wcs.world_to_pixel(coord)
        
        # Extract flux in aperture
        aperture_radius = 5  # pixels
        y_int, x_int = int(round(y)), int(round(x))
        y_grid, x_grid = np.ogrid[-aperture_radius:aperture_radius+1, 
                                 -aperture_radius:aperture_radius+1]
        aperture_mask = x_grid*x_grid + y_grid*y_grid <= aperture_radius*aperture_radius
        
        # Measure flux
        try:
            flux = np.sum(image[y_int-aperture_radius:y_int+aperture_radius+1,
                               x_int-aperture_radius:x_int+aperture_radius+1][aperture_mask])
            if flux > 0:  # Only use valid measurements
                measured_fluxes.append(flux)
                star_mags.append(star['magnitude_r'])  # Use r-band magnitude for calibration
                star_positions.append((x, y))
        except IndexError:
            continue
    
    # Calculate zero point
    if len(measured_fluxes) > 0:
        zero_point = np.median(star_mags + 2.5 * np.log10(measured_fluxes))
        
        # Apply calibration
        calibrated_image = image.copy()
        valid_pixels = calibrated_image > 0
        calibrated_image[valid_pixels] = zero_point - 2.5 * np.log10(calibrated_image[valid_pixels])
        
        return calibrated_image, zero_point
    else:
        return image, None

def validate_and_fix_wcs(wcs_header):
    """
    Validate and fix WCS header information
    """
    try:
        print("Original WCS header values:")
        for key in ['CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
            if key in wcs_header:
                print(f"{key} = {wcs_header[key]}")
                
        # Create WCS object from header
        wcs = WCS(wcs_header)
        
        # Get center coordinates
        center_x = wcs_header['NAXIS1'] / 2
        center_y = wcs_header['NAXIS2'] / 2
        print(f"Image center pixels: x={center_x}, y={center_y}")
        
        center_ra, center_dec = wcs.all_pix2world([[center_x, center_y]], 0)[0]
        print(f"Image center coordinates: RA = {center_ra:.6f}, Dec = {center_dec:.6f}")
        
        modified = False
        # Check if coordinates are valid
        if not (-90 <= center_dec <= 90):
            print(f"Warning: Invalid declination value {center_dec}, attempting to fix...")
            # Try to fix by wrapping declination
            center_dec = ((center_dec + 90) % 180) - 90
            print(f"Corrected declination: {center_dec}")
            
            # Update WCS header
            if 'CRVAL2' in wcs_header:
                wcs_header['CRVAL2'] = center_dec
                modified = True
        
        if center_ra < 0 or center_ra > 360:
            print(f"Warning: Invalid RA value {center_ra}, attempting to fix...")
            # Wrap RA to 0-360 range
            center_ra = center_ra % 360
            print(f"Corrected RA: {center_ra}")
            
            # Update WCS header
            if 'CRVAL1' in wcs_header:
                wcs_header['CRVAL1'] = center_ra
                modified = True
        
        if modified:
            print("\nUpdated WCS header values:")
            for key in ['CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
                if key in wcs_header:
                    print(f"{key} = {wcs_header[key]}")
            
            # Verify the fix worked
            wcs = WCS(wcs_header)
            center_ra, center_dec = wcs.all_pix2world([[center_x, center_y]], 0)[0]
            print(f"Verified center coordinates: RA = {center_ra:.6f}, Dec = {center_dec:.6f}")
        
        return WCS(wcs_header)
    except Exception as e:
        print(f"Error validating WCS: {str(e)}")
        print("WCS header contents:")
        for key in wcs_header:
            print(f"{key} = {wcs_header[key]}")
        return None

def find_valid_region(images):
    """
    Find the largest rectangular region without NaN values across multiple images
    """
    # Print initial diagnostics
    for i, img in enumerate(images):
        nan_count = np.sum(np.isnan(img))
        total_pixels = img.size
        print(f"Channel {i}: {nan_count}/{total_pixels} NaN pixels ({nan_count/total_pixels*100:.1f}%)")
        
    # Create combined mask of valid (non-NaN) pixels across all images
    combined_valid = np.ones_like(images[0], dtype=bool)
    for img in images:
        valid_mask = ~np.isnan(img)
        combined_valid &= valid_mask
        
    # Count valid pixels
    total_valid = np.sum(combined_valid)
    print(f"Total valid pixels across all channels: {total_valid}/{combined_valid.size} ({total_valid/combined_valid.size*100:.1f}%)")
    
    # Find rows and columns that contain enough valid pixels
    min_valid_fraction = 0.5  # At least 50% of pixels in row/column must be valid
    valid_rows = []
    valid_cols = []
    
    # Check rows
    for i in range(combined_valid.shape[0]):
        valid_fraction = np.mean(combined_valid[i, :])
        if valid_fraction >= min_valid_fraction:
            valid_rows.append(i)
            
    # Check columns
    for j in range(combined_valid.shape[1]):
        valid_fraction = np.mean(combined_valid[:, j])
        if valid_fraction >= min_valid_fraction:
            valid_cols.append(j)
    
    print(f"Found {len(valid_rows)} valid rows and {len(valid_cols)} valid columns")
    
    if len(valid_rows) == 0 or len(valid_cols) == 0:
        # Try more aggressive thresholding
        print("Retrying with lower valid fraction threshold...")
        min_valid_fraction = 0.3  # Lower threshold to 30%
        
        valid_rows = []
        valid_cols = []
        
        for i in range(combined_valid.shape[0]):
            valid_fraction = np.mean(combined_valid[i, :])
            if valid_fraction >= min_valid_fraction:
                valid_rows.append(i)
                
        for j in range(combined_valid.shape[1]):
            valid_fraction = np.mean(combined_valid[:, j])
            if valid_fraction >= min_valid_fraction:
                valid_cols.append(j)
        
        print(f"After retry: Found {len(valid_rows)} valid rows and {len(valid_cols)} valid columns")
        
        if len(valid_rows) == 0 or len(valid_cols) == 0:
            raise ValueError("No valid region found in images even with relaxed constraints")
    
    # Convert to contiguous regions
    valid_rows = np.array(valid_rows)
    valid_cols = np.array(valid_cols)
    
    # Find largest contiguous region
    row_groups = np.split(valid_rows, np.where(np.diff(valid_rows) != 1)[0] + 1)
    col_groups = np.split(valid_cols, np.where(np.diff(valid_cols) != 1)[0] + 1)
    
    # Select largest contiguous regions
    largest_row_group = max(row_groups, key=len)
    largest_col_group = max(col_groups, key=len)
    
    y_start, y_end = largest_row_group[0], largest_row_group[-1] + 1
    x_start, x_end = largest_col_group[0], largest_col_group[-1] + 1
    
    region_height = y_end - y_start
    region_width = x_end - x_start
    original_height, original_width = combined_valid.shape
    
    print(f"Original dimensions: {original_height}x{original_width}")
    print(f"Valid region dimensions: {region_height}x{region_width}")
    print(f"Valid region: rows {y_start}:{y_end}, columns {x_start}:{x_end}")
    print(f"Keeping {region_height*region_width/(original_height*original_width)*100:.1f}% of original image")
    
    return y_start, y_end, x_start, x_end
    """
    Find the largest rectangular region without NaN values across multiple images
    """
    # Print initial diagnostics
    for i, img in enumerate(images):
        nan_count = np.sum(np.isnan(img))
        total_pixels = img.size
        print(f"Channel {i}: {nan_count}/{total_pixels} NaN pixels ({nan_count/total_pixels*100:.1f}%)")
        
    # Create combined mask of valid (non-NaN) pixels across all images
    combined_valid = np.ones_like(images[0], dtype=bool)
    for img in images:
        valid_mask = ~np.isnan(img)
        combined_valid &= valid_mask
        
    # Count valid pixels
    total_valid = np.sum(combined_valid)
    print(f"Total valid pixels across all channels: {total_valid}/{combined_valid.size} ({total_valid/combined_valid.size*100:.1f}%)")
    
    # Find rows and columns that contain enough valid pixels
    min_valid_fraction = 0.5  # At least 50% of pixels in row/column must be valid
    valid_rows = []
    valid_cols = []
    
    # Check rows
    for i in range(combined_valid.shape[0]):
        valid_fraction = np.mean(combined_valid[i, :])
        if valid_fraction >= min_valid_fraction:
            valid_rows.append(i)
            
    # Check columns
    for j in range(combined_valid.shape[1]):
        valid_fraction = np.mean(combined_valid[:, j])
        if valid_fraction >= min_valid_fraction:
            valid_cols.append(j)
    
    print(f"Found {len(valid_rows)} valid rows and {len(valid_cols)} valid columns")
    
    if len(valid_rows) == 0 or len(valid_cols) == 0:
        # Try more aggressive thresholding
        print("Retrying with lower valid fraction threshold...")
        min_valid_fraction = 0.3  # Lower threshold to 30%
        
        valid_rows = []
        valid_cols = []
        
        for i in range(combined_valid.shape[0]):
            valid_fraction = np.mean(combined_valid[i, :])
            if valid_fraction >= min_valid_fraction:
                valid_rows.append(i)
                
        for j in range(combined_valid.shape[1]):
            valid_fraction = np.mean(combined_valid[:, j])
            if valid_fraction >= min_valid_fraction:
                valid_cols.append(j)
        
        print(f"After retry: Found {len(valid_rows)} valid rows and {len(valid_cols)} valid columns")
        
        if len(valid_rows) == 0 or len(valid_cols) == 0:
            raise ValueError("No valid region found in images even with relaxed constraints")
    
    # Convert to contiguous regions
    valid_rows = np.array(valid_rows)
    valid_cols = np.array(valid_cols)
    
    # Find largest contiguous region
    row_groups = np.split(valid_rows, np.where(np.diff(valid_rows) != 1)[0] + 1)
    col_groups = np.split(valid_cols, np.where(np.diff(valid_cols) != 1)[0] + 1)
    
    # Select largest contiguous regions
    largest_row_group = max(row_groups, key=len)
    largest_col_group = max(col_groups, key=len)
    
    y_start, y_end = largest_row_group[0], largest_row_group[-1] + 1
    x_start, x_end = largest_col_group[0], largest_col_group[-1] + 1
    
    region_height = y_end - y_start
    region_width = x_end - x_start
    original_height, original_width = combined_valid.shape
    
    print(f"Original dimensions: {original_height}x{original_width}")
    print(f"Valid region dimensions: {region_height}x{region_width}")
    print(f"Valid region: rows {y_start}:{y_end}, columns {x_start}:{x_end}")
    print(f"Keeping {region_height*region_width/(original_height*original_width)*100:.1f}% of original image")
    
    return y_start, y_end, x_start, x_end

def custom_stretch(image, black_point=0.1, white_point=0.999, gamma=0.5):
    """
    Apply a custom stretch with histogram equalization
    """
    valid_data = image[~np.isnan(image)]
    black = np.percentile(valid_data, black_point * 100)
    white = np.percentile(valid_data, white_point * 100)
    stretched = np.clip((image - black) / (white - black), 0, 1)
    stretched = np.power(stretched, gamma)
    return stretched

def raised_cosine_taper_2d(shape, taper_width_percent=10):
    """
    Create a 2D raised cosine (Hann) window taper
    """
    height, width = shape
    h_taper = int(height * taper_width_percent / 100)
    w_taper = int(width * taper_width_percent / 100)
    
    def raised_cosine_window(length):
        return 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, length)))
    
    h_window = np.ones(width)
    v_window = np.ones(height)
    
    if w_taper > 0:
        h_window[:w_taper] = raised_cosine_window(w_taper)
        h_window[-w_taper:] = raised_cosine_window(w_taper)[::-1]
    if h_taper > 0:
        v_window[:h_taper] = raised_cosine_window(h_taper)
        v_window[-h_taper:] = raised_cosine_window(h_taper)[::-1]
    
    return v_window[:, np.newaxis] * h_window[np.newaxis, :]

def eliminate_background_wavelet(image, levels=3, taper_width_percent=10, debug_plot=False):
    """
    Remove background using wavelet transform with raised cosine tapering
    """
    try:
        import pywt
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("PyWavelets or Matplotlib not installed. Falling back to local background elimination.")
        return eliminate_background_local(image)
    
    # Create raised cosine taper mask
    taper_mask = raised_cosine_taper_2d(image.shape, taper_width_percent)
    
    # Perform wavelet decomposition
    coeffs = list(pywt.wavedec2(image, 'db4', level=levels))  # Convert to list
    
    # Store original approximation coefficients
    original_approx = coeffs[0].copy()
    
    # Modify detailed coefficients to reduce noise while preserving structure
    for i in range(1, len(coeffs)):
        coeffs[i] = list(coeffs[i])  # Convert tuple to list
        for j in range(len(coeffs[i])):
            coeffs[i][j] *= 0.5
        coeffs[i] = tuple(coeffs[i])  # Convert back to tuple
    
    # Reconstruct image with modified coefficients
    background = pywt.waverec2(coeffs, 'db4')
    
    # Ensure background has same shape as original
    background = background[:image.shape[0], :image.shape[1]]
    
    # Apply raised cosine tapering to background subtraction
    tapered_background = background * taper_mask
    
    return image - tapered_background

def save_16bit_png(data, filename, max_size_mb=20, max_trim_percent=0.10):
    """
    Save a normalized 16-bit PNG, automatically reducing size if needed
    """
    def try_save_with_trim(data, trim_percent):
        data_transposed = data.transpose(1, 2, 0)
        height, width, _ = data_transposed.shape
        
        trim_height = int(height * trim_percent)
        trim_width = int(width * trim_percent)
        
        trimmed_data = data_transposed[
            trim_height:height-trim_height, 
            trim_width:width-trim_width
        ]
        
        # Handle invalid values before scaling
        valid_mask = ~np.isnan(trimmed_data) & ~np.isinf(trimmed_data)
        cleaned_data = np.zeros_like(trimmed_data)
        cleaned_data[valid_mask] = np.clip(trimmed_data[valid_mask], 0, 1)
        
        # Scale to 16-bit range
        scaled_data = (cleaned_data * 65535).astype(np.uint16)
        temp_filename = f"{filename}.temp.png"
        
        cv2.imwrite(temp_filename, scaled_data, [
            cv2.IMWRITE_PNG_COMPRESSION, 9,
            cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT
        ])
        
        file_size_mb = os.path.getsize(temp_filename) / (1024 * 1024)
        os.remove(temp_filename)
        return scaled_data, file_size_mb
    
    trim_steps = np.linspace(0, max_trim_percent, num=5)
    for trim_percent in trim_steps:
        scaled_data, file_size_mb = try_save_with_trim(data, trim_percent)
        print(f"Trying {trim_percent*100:.1f}% trim: {file_size_mb:.1f}MB")
        
        if file_size_mb <= max_size_mb:
            print(f"Found acceptable trim: {trim_percent*100:.1f}%")
            cv2.imwrite(filename, scaled_data, [
                cv2.IMWRITE_PNG_COMPRESSION, 9,
                cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT
            ])
            print(f"Final PNG size: {os.path.getsize(filename) / (1024 * 1024):.2f}MB")
            return
    
    print("Warning: Could not reduce file size below target even with maximum trimming")

def main():
    args = parse_arguments()
    
    # Read the individual stacks with WCS information
    r_hdu = fits.open('stacked_r.fits')[0]
    g_hdu = fits.open('stacked_g.fits')[0]
    b_hdu = fits.open('stacked_b.fits')[0]
    
    r = r_hdu.data
    g = g_hdu.data
    b = b_hdu.data
    
    # Get and validate WCS information
    print("\nValidating WCS information...")
    wcs = validate_and_fix_wcs(r_hdu.header)
    if wcs is None:
        print("Warning: Invalid WCS information, photometric calibration may fail")
        wcs = WCS(r_hdu.header)  # Use original WCS as fallback
    
    print("\nAnalyzing input images...")
    try:
        y_start, y_end, x_start, x_end = find_valid_region([r, g, b])
        print("\nCropping images to valid region...")
        
        # Crop all channels to valid region
        r = r[y_start:y_end, x_start:x_end]
        g = g[y_start:y_end, x_start:x_end]
        b = b[y_start:y_end, x_start:x_end]
        
        # Update WCS for cropped region
        wcs = wcs.slice((slice(y_start, y_end), slice(x_start, x_end)))
    except ValueError as e:
        print(f"\nError: {str(e)}")
        print("Attempting to proceed with uncropped images...")
        # Fill NaN values with zeros instead
        r[np.isnan(r)] = 0
        g[np.isnan(g)] = 0
        b[np.isnan(b)] = 0
    except ValueError as e:
        print(f"Error: {str(e)}")
        exit(1)
    
    # Stretch each channel
    r_stretched = custom_stretch(r)
    g_stretched = custom_stretch(g)
    b_stretched = custom_stretch(b)
    
    # Apply background gradient elimination if specified
    if args.method == 'wavelet':
        r_stretched = eliminate_background_wavelet(r_stretched)
        g_stretched = eliminate_background_wavelet(g_stretched)
        b_stretched = eliminate_background_wavelet(b_stretched)
    
    # Perform photometric calibration using catalog stars
    try:
        print(f"Querying {args.catalog} catalog for reference stars...")
        reference_stars = get_catalog_stars(wcs, catalog=args.catalog, mag_limit=args.mag_limit)
        
        if reference_stars is None or len(reference_stars) == 0:
            print(f"No suitable reference stars found in {args.catalog} catalog")
        else:
            print(f"Found {len(reference_stars)} reference stars")
            
            # Calibrate each channel
            r_calibrated, r_zp = perform_photometric_calibration(r_stretched, wcs, reference_stars)
            g_calibrated, g_zp = perform_photometric_calibration(g_stretched, wcs, reference_stars)
            b_calibrated, b_zp = perform_photometric_calibration(b_stretched, wcs, reference_stars)
            
            if all(zp is not None for zp in [r_zp, g_zp, b_zp]):
                print(f"Zero points - R: {r_zp:.2f}, G: {g_zp:.2f}, B: {b_zp:.2f}")
                r_stretched, g_stretched, b_stretched = r_calibrated, g_calibrated, b_calibrated
            else:
                print("Warning: Photometric calibration failed for one or more channels")
    except Exception as e:
        print(f"Error during photometric calibration: {str(e)}")
    
    # Stack stretched channels
    rgb_stretched = np.stack((r_stretched, g_stretched, b_stretched), axis=0)
    
    # Save outputs with updated WCS
    new_header = r_hdu.header.copy()
    new_header.update(wcs.to_header())
    
    # Update NAXIS1/NAXIS2 to match cropped size
    new_header['NAXIS1'] = rgb_stretched.shape[2]
    new_header['NAXIS2'] = rgb_stretched.shape[1]
    
    # Save FITS with updated header
    hdu = fits.PrimaryHDU(rgb_stretched, header=new_header)
    hdu.writeto('merged_rgb_stretched.fits', overwrite=True)
    
    # Save PNG
    save_16bit_png(rgb_stretched, "merged_rgb_stretched16.png")
    
    print("Image processing complete.")

if __name__ == "__main__":
    main()