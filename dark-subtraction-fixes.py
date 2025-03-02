def get_dark_frame(fits_file, dark_frame_path=None, dark_manager=None, rotate_dark_180=True):
    """
    Get appropriately rotated dark frame for calibration
    
    Parameters:
    -----------
    fits_file : str
        Path to the FITS file to be calibrated
    dark_frame_path : str
        Path to a fixed dark frame file
    dark_manager : DarkFrameManager
        Initialized dark frame manager for temperature-matched calibration
    rotate_dark_180 : bool
        Whether to rotate the dark frame 180 degrees before use
    
    Returns:
    --------
    dark_data : ndarray or None
        Dark frame data if available, or None
    message : str
        Information about the dark frame processing
    """
    # If no dark frame sources, return None
    if dark_frame_path is None and dark_manager is None:
        return None, "No dark frame source provided"
    
    # Try to read the FITS file to get metadata
    try:
        with fits.open(fits_file) as hdul:
            light_data = hdul[0].data
            header = hdul[0].header
            
            # Get exposure time for possible scaling
            light_exptime = header.get('EXPTIME', header.get('EXP_TIME', 0))
            
            # Check for multiple possible temperature keys
            temp_keys = ['CCD-TEMP', 'CCDTEMP', 'CCD_TEMP', 'TEMP', 'TEMPERATURE']
            temp_c = 20  # Default temperature
            for key in temp_keys:
                if key in header:
                    temp_c = float(header[key])
                    break
            
            # Determine Bayer pattern
            pattern = header.get('BAYERPAT', header.get('BAYER', 'RGGB')).strip()
            
    except Exception as e:
        return None, f"Error reading {fits_file}: {e}"
    
    dark_data = None
    dark_header = None
    message = ""
    
    # Option 1: Use dark manager with temperature matching
    if dark_manager is not None:
        try:
            dark_data, message = dark_manager.get_dark_frame(temp_c, pattern, light_data)
            # Continue with dark_data processing if it was found
        except Exception as e:
            print(f"Error getting dark frame from manager: {e}")
            dark_data = None
            message = f"Dark manager error: {str(e)}"
            # Fall back to fixed dark frame if available
    
    # Option 2: Use fixed dark frame if dark_manager didn't work or isn't available
    if dark_data is None and dark_frame_path:
        try:
            with fits.open(dark_frame_path) as hdul:
                dark_data = hdul[0].data.astype(np.float32)
                dark_header = hdul[0].header
                message = f"Using fixed dark frame: {dark_frame_path}"
        except Exception as e:
            return None, f"Error loading dark frame {dark_frame_path}: {e}"
    
    # If we still have no dark data, return None
    if dark_data is None:
        return None, "Failed to get appropriate dark frame"
    
    # Get dark exposure time for scaling
    dark_exptime = 0
    if dark_header is not None:
        dark_exptime = dark_header.get('EXPTIME', dark_header.get('EXP_TIME', 0))
    
    # Scale the dark frame if exposure times differ
    if dark_exptime > 0 and light_exptime > 0 and abs(dark_exptime - light_exptime) > 0.001:
        scaling_factor = light_exptime / dark_exptime
        dark_data = dark_data * scaling_factor
        message += f" (scaled by {scaling_factor:.2f} for exposure time match)"
    
    # Shape handling - for minor differences, try to reshape/crop
    if dark_data.shape != light_data.shape:
        # Only handle simple cases - when dimensions are the same but in different order
        if dark_data.size == light_data.size and len(dark_data.shape) == len(light_data.shape):
            try:
                dark_data = dark_data.reshape(light_data.shape)
                message += " (reshaped to match light frame)"
            except ValueError:
                # Reshaping failed, continue and let the caller handle it
                pass
    
    # Apply 180-degree rotation if requested
    if rotate_dark_180 and dark_data is not None:
        dark_data = np.rot90(dark_data, 2)  # Rotate 180 degrees
        message += " (rotated 180°)"
    
    return dark_data, message


def process_fits_files(fits_directory, transformations, output_file, 
                      dark_frame_path=None, dark_directory=None,
                      stacking_method='mean', quality_threshold=0.4, max_files=None,
                      interpolation_order=3, rotate_180=True, margin_percent=10,
                      normalize_output=False, save_calibrated=False, rotate_dark_180=True):
    """Process FITS files using the extracted transformations"""
    start_time = time.time()
    print(f"Processing FITS files from: {fits_directory}")
    
    # Find all FITS files in the directory
    fits_files = sorted(glob.glob(os.path.join(fits_directory, "*.fits")))
    
    if not fits_files:
        print(f"No FITS files found in {fits_directory}")
        return
    
    num_files = len(fits_files)
    print(f"Found {num_files} FITS files")
    
    # Initialize dark manager if directory provided
    dark_manager = None
    if dark_directory and DARK_MANAGER_AVAILABLE:
        try:
            dark_manager = DarkFrameManager(dark_directory)
            print(f"Initialized dark manager from {dark_directory}")
        except Exception as e:
            print(f"Error initializing dark manager: {e}")
            print("Falling back to fixed dark frame if provided")
    
    # ... [rest of the function remains unchanged until the dark frame usage] ...
    
    # Apply dark frame subtraction
    dark_data, message = get_dark_frame(file_to_use, dark_frame_path, dark_manager, rotate_dark_180)
    if dark_data is not None:
        if dark_data.shape == image_data.shape:
            # Apply dark subtraction with clamping to zero
            image_data = np.maximum(image_data - dark_data, 0)
            dark_messages.append(f"{os.path.basename(file_to_use)}: {message}")
            
            # Add metadata about dark subtraction to the output
            if save_calibrated:
                cal_header = fits.Header()
                cal_header['HISTORY'] = f"Dark subtracted: {message}"
                cal_header['DARKSUB'] = 'True'
        else:
            print(f"Warning: Dark frame shape {dark_data.shape} doesn't match image shape {image_data.shape}")
            dark_messages.append(f"{os.path.basename(file_to_use)}: Dark frame shape mismatch ({dark_data.shape} vs {image_data.shape})")
    
    # ... [Continue with the rest of the function] ...