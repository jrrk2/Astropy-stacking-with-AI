def solve_astrometry(image_path):
    """
    Run astrometry.net's solve-field on an image
    """
    import subprocess
    import os
    
    print(f"\nRunning astrometry.net on {image_path}")
    try:
        # Run solve-field with typical options
        cmd = [
            'solve-field',
            '--overwrite',
            '--no-plots',
            '--no-verify',
            '--scale-units', 'arcsecperpix',
            '--scale-low', '0.5',
            '--scale-high', '2.0',
            image_path
        ]
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Astrometric solution succeeded")
            
            # Get the new file path (.new)
            base = os.path.splitext(image_path)[0]
            new_path = base + '.new'
            
            # Check if solved file exists
            if os.path.exists(new_path):
                os.replace(new_path, image_path)
                print(f"Updated {image_path} with new astrometric solution")
                
                # Clean up temporary files
                for ext in ['.axy', '.corr', '.match', '.rdls', '.solved', '.wcs']:
                    temp_file = base + ext
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                return True
            else:
                print(f"Warning: Solved file not found at {new_path}")
                return False
        else:
            print("Astrometric solution failed")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running solve-field: {str(e)}")
        return False

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