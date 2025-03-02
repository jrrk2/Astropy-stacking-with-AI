def progressive_stack_fft(self, filtered_frames, dark_frame=None, dark_scale=1.0):
    """
    Progressively stack frames using enhanced FFT alignment with dark frame subtraction
    for faint subframes where star detection might be unreliable
    
    Parameters:
    -----------
    filtered_frames : list
        List of frames to stack
    dark_frame : ndarray or str, optional
        Dark frame for calibration. Can be a numpy array or path to FITS file
    dark_scale : float, optional
        Scaling factor for dark frame if exposure times differ
    
    Returns:
    --------
    stacked_image : ndarray
        Final stacked image
    """
    # Load dark frame if provided as a path
    if dark_frame is not None and isinstance(dark_frame, str):
        try:
            dark_path = dark_frame
            dark_frame = fits.getdata(dark_path)
            print(f"Loaded dark frame from: {dark_path}")
        except Exception as e:
            print(f"Error loading dark frame: {e}")
            dark_frame = None
    
    # Make sure frames are sorted by quality score
    quality_df = self.metrics_df[self.metrics_df['filename'].isin(filtered_frames)]
    sorted_frames = quality_df.sort_values('quality_score', ascending=False)['filename'].tolist()
    
    # Select highest quality frame as reference
    reference_filename = sorted_frames[0]
    reference_path = os.path.join(self.fits_directory, reference_filename)
    raw_reference_frame = fits.getdata(reference_path)
    
    # Apply dark subtraction to reference frame if available
    if dark_frame is not None:
        # Ensure shapes match
        if raw_reference_frame.shape != dark_frame.shape:
            print(f"Warning: Dark frame shape {dark_frame.shape} doesn't match light frame shape {raw_reference_frame.shape}")
            print("Skipping dark subtraction")
            self.reference_frame = raw_reference_frame.astype(np.float64)
        else:
            print(f"Applying dark subtraction with scale factor: {dark_scale}")
            self.reference_frame = raw_reference_frame.astype(np.float64) - (dark_frame.astype(np.float64) * dark_scale)
            # Clip negative values to zero
            self.reference_frame = np.clip(self.reference_frame, 0, None)
    else:
        self.reference_frame = raw_reference_frame.astype(np.float64)
    
    # Initialize stacked image with reference frame
    stacked_image = self.reference_frame.copy()
    frames_stacked = 1
    
    # Track quality metrics of reference frame
    current_snr = self.metrics_df.loc[self.metrics_df['filename'] == reference_filename, 'mean_snr'].values[0]
    base_snr = current_snr  # Store initial SNR for relative comparison
    
    print(f"Started FFT-based stacking with reference frame: {reference_filename} (SNR: {current_snr:.2f})")
    
    # Store alignment info for logging
    alignment_info = []
    alignment_info.append({
        'filename': reference_filename,
        'shift': (0, 0),
        'confidence': 100.0,
        'snr': current_snr
    })
    
    # Keep track of rejected frames
    rejected_frames = []
    low_confidence_threshold = 60.0  # Minimum confidence for inclusion
    
    # Iterate through remaining frames
    for frame_name in sorted_frames[1:]:
        # Read frame
        frame_path = os.path.join(self.fits_directory, frame_name)
        raw_frame = fits.getdata(frame_path)
        
        # Apply dark subtraction if available
        if dark_frame is not None and raw_frame.shape == dark_frame.shape:
            current_frame = raw_frame.astype(np.float64) - (dark_frame.astype(np.float64) * dark_scale)
            current_frame = np.clip(current_frame, 0, None)  # Clip negative values
        else:
            current_frame = raw_frame.astype(np.float64)
        
        # Get frame's SNR
        new_snr = self.metrics_df.loc[self.metrics_df['filename'] == frame_name, 'mean_snr'].values[0]
        
        # Align frame using FFT
        aligned_frame, shift, metadata = self.align_frame_fft_enhanced(current_frame, self.reference_frame)
        
        # Store alignment info
        frame_info = {
            'filename': frame_name,
            'shift': shift,
            'confidence': metadata['confidence'],
            'snr': new_snr
        }
        
        # Determine if the frame should be included
        # Criteria: 
        # 1. Good alignment confidence
        # 2. SNR within reasonable limits of reference
        if (metadata['confidence'] >= low_confidence_threshold and 
            new_snr >= base_snr * 0.5):  # Accept frames with at least 50% of reference SNR
            
            # Add to stack
            stacked_image += aligned_frame
            frames_stacked += 1
            
            # Update running average SNR
            current_snr = (current_snr * (frames_stacked - 1) + new_snr) / frames_stacked
            
            # Store success info
            frame_info['status'] = 'stacked'
            alignment_info.append(frame_info)
            
            print(f"Stacked frame: {frame_name} (Frames: {frames_stacked}, Confidence: {metadata['confidence']:.1f}%, SNR: {new_snr:.2f})")
            
            # Option to update reference every N frames for very long sequences
            if frames_stacked % 20 == 0:  # Every 20 frames
                # Consider refreshing reference if doing very long sequences
                pass
        else:
            # Reject frame and log reason
            if metadata['confidence'] < low_confidence_threshold:
                reason = f"Low alignment confidence ({metadata['confidence']:.1f}%)"
            else:
                reason = f"Low SNR ({new_snr:.2f} < {base_snr * 0.5:.2f})"
                
            frame_info['status'] = 'rejected'
            frame_info['reason'] = reason
            rejected_frames.append(frame_info)
            
            print(f"Rejected frame: {frame_name} - {reason}")
    
    # Normalize the stacked image
    if frames_stacked > 0:
        self.stacked_image = stacked_image / frames_stacked
        print(f"Final stack complete with {frames_stacked} frames ({len(rejected_frames)} rejected)")
        
        # Optional: Generate report of alignment results
        successful_frames = [info for info in alignment_info if info.get('status', '') == 'stacked']
        if successful_frames:
            shifts = np.array([info['shift'] for info in successful_frames[1:]])  # Skip reference
            if len(shifts) > 0:
                mean_shift = np.mean(shifts, axis=0)
                std_shift = np.std(shifts, axis=0)
                max_shift = np.max(np.abs(shifts), axis=0)
                
                print("\nAlignment Statistics:")
                print(f"Mean shift (y,x): ({mean_shift[0]:.2f}, {mean_shift[1]:.2f}) pixels")
                print(f"Std deviation: ({std_shift[0]:.2f}, {std_shift[1]:.2f}) pixels")
                print(f"Max absolute shift: ({max_shift[0]:.2f}, {max_shift[1]:.2f}) pixels")
                
        return self.stacked_image, alignment_info, rejected_frames
    else:
        print("No frames were stacked!")
        return None, alignment_info, rejected_frames

def process_with_fft(self, output_path='stacked_fft_image.fits', dark_frame=None, dark_scale=1.0):
    """
    Process pipeline using FFT alignment with dark subtraction - ideal for faint targets
    
    Parameters:
    -----------
    output_path : str, optional
        Path to save the stacked image
    dark_frame : ndarray or str, optional
        Dark frame for calibration. Can be a numpy array or path to FITS file
    dark_scale : float, optional
        Scaling factor for dark frame if exposure times differ
    
    Returns:
    --------
    tuple
        (stacked_image, alignment_info, rejected_frames)
    """
    # Filter frames based on quality
    filtered_frames = self.filter_frames()
    print(f"Frames to be stacked: {len(filtered_frames)}")
    
    # Perform FFT-based progressive stacking with dark subtraction
    stacked_image, alignment_info, rejected_frames = self.progressive_stack_fft(
        filtered_frames, dark_frame, dark_scale
    )
    
    # Save stacked image
    if stacked_image is not None:
        # Create a new FITS HDU with the stacked image
        hdu = fits.PrimaryHDU(stacked_image)
        
        # Add stacking metadata
        hdu.header['NFRAMES'] = (len(alignment_info) - 1, 'Number of frames stacked')
        hdu.header['NREJECT'] = (len(rejected_frames), 'Number of frames rejected')
        hdu.header['STACKMTD'] = ('FFT', 'Stacking method')
        
        # Add reference frame info
        ref_info = alignment_info[0]
        hdu.header['REFFRAME'] = (ref_info['filename'], 'Reference frame')
        
        # Add dark calibration info
        if dark_frame is not None:
            hdu.header['DARKSUB'] = (True, 'Dark subtraction applied')
            hdu.header['DARKSCL'] = (dark_scale, 'Dark frame scaling factor')
            if isinstance(dark_frame, str):
                hdu.header['DARKFILE'] = (os.path.basename(dark_frame), 'Dark frame filename')
        else:
            hdu.header['DARKSUB'] = (False, 'No dark subtraction applied')
        
        # Add creation date
        from datetime import datetime
        hdu.header['DATE'] = (datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'Creation date')
        
        # Save the file
        hdu.writeto(output_path, overwrite=True)
        print(f"Stacked image saved to {output_path}")
        
        return stacked_image, alignment_info, rejected_frames
    else:
        print("No stacked image to save")
        return None, alignment_info, rejected_frames