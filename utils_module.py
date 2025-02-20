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

def perform_photometric_calibration(images, wcs, reference_stars):
    """
    Analyze potential photometric calibration scaling
    
    Parameters:
    - images: Single image or list of images
    - wcs: World Coordinate System object
    - reference_stars: List of reference stars
    
    Returns:
    - Scaling information 
    """
    # Ensure images is a list
    if not isinstance(images, list):
        images = [images]
    
    # Validate input
    print(f"Number of images: {len(images)}")
    for i, img in enumerate(images):
        print(f"Image {i} shape: {img.shape}")
    
    channel_scalings = []
    
    for channel_index, channel_image in enumerate(images):
        # Extract star positions and fluxes for this channel
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
            
            # Robust bounds checking
            if (y_int-aperture_radius < 0 or y_int+aperture_radius+1 > channel_image.shape[0] or
                x_int-aperture_radius < 0 or x_int+aperture_radius+1 > channel_image.shape[1]):
                continue
            
            y_grid, x_grid = np.ogrid[-aperture_radius:aperture_radius+1, 
                                     -aperture_radius:aperture_radius+1]
            aperture_mask = x_grid*x_grid + y_grid*y_grid <= aperture_radius*aperture_radius
            
            # Measure flux
            try:
                flux = np.sum(channel_image[y_int-aperture_radius:y_int+aperture_radius+1,
                                   x_int-aperture_radius:x_int+aperture_radius+1][aperture_mask])
                if flux > 0:  # Only use valid measurements
                    measured_fluxes.append(flux)
                    star_mags.append(star['magnitude_r'])  # Use r-band magnitude for calibration
                    star_positions.append((x, y))
            except IndexError:
                continue
        
        # Analyze scaling for this channel
        if len(measured_fluxes) > 0:
            # Find the maximum pixel value (brightest point)
            max_pixel_value = np.max(channel_image)
            
            # Calculate zero point using star magnitudes and fluxes
            log_fluxes = 2.5 * np.log10(measured_fluxes)
            zero_point = np.median(star_mags + log_fluxes)
            
            # Compute scaling factor based on flux measurements
            flux_scaling = np.percentile(measured_fluxes, 99) / np.max(measured_fluxes)
            
            # Prepare scaling information
            channel_name = ['R', 'G', 'B'][channel_index] if len(images) > 1 else 'Monochrome'
            scaling_info = {
                'channel': channel_name,
                'max_pixel_value': max_pixel_value,
                'zero_point': zero_point,
                'flux_scaling': flux_scaling,
                'num_reference_stars': len(measured_fluxes)
            }
            
            channel_scalings.append(scaling_info)
            
            # Print out detailed scaling information
            print(f"{scaling_info['channel']} Channel Scaling Analysis:")
            print(f"  Number of reference stars: {scaling_info['num_reference_stars']}")
            print(f"  Max pixel value: {scaling_info['max_pixel_value']:.2f}")
            print(f"  Zero point: {scaling_info['zero_point']:.2f}")
            print(f"  Flux scaling factor: {scaling_info['flux_scaling']:.4f}")
            print()
        else:
            channel_name = ['R', 'G', 'B'][channel_index] if len(images) > 1 else 'Monochrome'
            print(f"{channel_name} Channel: No valid reference stars found")
    
    if len(channel_scalings) == len(images):
        zero_points = [scaling['zero_point'] for scaling in channel_scalings]
        return images, zero_points
    return images, None
