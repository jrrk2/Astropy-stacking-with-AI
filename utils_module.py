import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

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
        y_int, x_int = int(y), int(x)
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
