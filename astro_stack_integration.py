import os
import numpy as np
import astropy.io.fits as fits
from enhanced_registration import enhanced_registration, normalize_frame, visualize_alignment

class EnhancedAstroStackProcessor:
    def __init__(self, original_processor):
        """
        Enhance an existing AstroStackProcessor with improved registration
        
        Parameters:
        -----------
        original_processor : AstroStackProcessor
            Original processor to enhance
        """
        # Save reference to original processor
        self.processor = original_processor
        
        # Copy key attributes
        self.metrics_df = original_processor.metrics_df
        self.fits_directory = original_processor.fits_directory
        self.quality_thresholds = original_processor.quality_thresholds
        self.stacking_config = original_processor.stacking_config
        
        # Additional configuration for enhanced registration
        self.registration_config = {
            'detect_rotation': True,  # Enable rotation detection
            'max_rotation': 1.0,      # Maximum rotation to detect (degrees)
            'debug_output': True,     # Generate debug visualizations
        }
        
        # Create debug directory
        os.makedirs("debug", exist_ok=True)
        
    def process_with_enhanced_registration(self, filtered_frames=None, dark_frame=None, 
                                          dark_scale=1.0, debayer=True, bayer_pattern='RGGB',
                                          output_path='stacked_enhanced.fits'):
        """
        Process pipeline using enhanced registration with rotation correction
        
        Parameters:
        -----------
        filtered_frames : list or None
            List of frames to stack. If None, will be selected automatically
        dark_frame : ndarray or str, optional
            Dark frame for calibration
        dark_scale : float, optional
            Scaling factor for dark frame
        debayer : bool, optional
            Whether to perform debayering on raw frames
        bayer_pattern : str, optional
            Bayer pattern for debayering
        output_path : str, optional
            Path to save stacked image
            
        Returns:
        --------
        tuple
            (stacked_image, alignment_info, rejected_frames)
        """
        # Get frames to stack if not provided
        if filtered_frames is None:
            filtered_frames = self.processor.filter_frames()
            
        print(f"Enhanced registration with {len(filtered_frames)} frames")
        
        # Sort frames by quality
        quality_df = self.metrics_df[self.metrics_df['filename'].isin(filtered_frames)]
        sorted_frames = quality_df.sort_values('quality_score', ascending=False)['filename'].tolist()
        
        # Select highest quality frame as reference
        reference_filename = sorted_frames[0]
        reference_path = os.path.join(self.fits_directory, reference_filename)
        raw_reference_frame = fits.getdata(reference_path)
        
        # Apply preprocessing (debayering, dark subtraction)
        if debayer:
            try:
                # Use the debayer method from original processor if available
                if hasattr(self.processor, 'debayer_frame'):
                    reference_frame = self.processor.debayer_frame(raw_reference_frame)
                else:
                    # Basic debayering fallback using OpenCV if available
                    import cv2
                    bayer_map = {
                        'RGGB': cv2.COLOR_BAYER_BG2RGB,
                        'BGGR': cv2.COLOR_BAYER_RG2RGB,
                        'GRBG': cv2.COLOR_BAYER_GB2RGB,
                        'GBRG': cv2.COLOR_BAYER_GR2RGB
                    }
                    
                    # Ensure frame is of proper type for debayering
                    frame_uint8 = raw_reference_frame.astype(np.uint8) if raw_reference_frame.max() <= 255 else (raw_reference_frame / 256).astype(np.uint8)
                    color_frame = cv2.cvtColor(frame_uint8, bayer_map.get(bayer_pattern, cv2.COLOR_BAYER_BG2RGB))
                    reference_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2GRAY).astype(np.float64)
            except Exception as e:
                print(f"Debayering failed: {e}. Using raw frame.")
                reference_frame = raw_reference_frame.astype(np.float64)
        else:
            reference_frame = raw_reference_frame.astype(np.float64)
        
        # Apply dark subtraction if provided
        if dark_frame is not None:
            # Load dark frame if it's a path
            if isinstance(dark_frame, str):
                dark_frame = fits.getdata(dark_frame)
                if debayer:
                    if hasattr(self.processor, 'debayer_frame'):
                        dark_frame = self.processor.debayer_frame(dark_frame)
                    else:
                        try:
                            import cv2
                            dark_uint8 = dark_frame.astype(np.uint8) if dark_frame.max() <= 255 else (dark_frame / 256).astype(np.uint8)
                            dark_color = cv2.cvtColor(dark_uint8, bayer_map.get(bayer_pattern, cv2.COLOR_BAYER_BG2RGB))
                            dark_frame = cv2.cvtColor(dark_color, cv2.COLOR_RGB2GRAY).astype(np.float64)
                        except:
                            pass
            
            # Apply dark subtraction
            if reference_frame.shape == dark_frame.shape:
                reference_frame = np.clip(reference_frame - dark_frame * dark_scale, 0, None)
                print("Applied dark subtraction to reference frame")
        
        # Initialize stacked image and tracking variables
        stacked_image = reference_frame.copy()
        frames_stacked = 1
        
        # Track alignment info and rejected frames
        alignment_info = []
        rejected_frames = []
        
        # Record reference frame info
        alignment_info.append({
            'filename': reference_filename,
            'method': 'reference',
            'shift_x': 0.0,
            'shift_y': 0.0,
            'rotation': 0.0
        })
        
        print(f"Starting enhanced registration with reference: {reference_filename}")
        
        # Process each frame
        for frame_idx, frame_filename in enumerate(sorted_frames[1:]):
            frame_path = os.path.join(self.fits_directory, frame_filename)
            print(f"Processing frame {frame_idx+1}/{len(sorted_frames)-1}: {frame_filename}")
            
            # Load frame
            raw_frame = fits.getdata(frame_path)
            
            # Apply same preprocessing as reference
            if debayer:
                try:
                    if hasattr(self.processor, 'debayer_frame'):
                        current_frame = self.processor.debayer_frame(raw_frame)
                    else:
                        # Basic debayering fallback
                        import cv2
                        frame_uint8 = raw_frame.astype(np.uint8) if raw_frame.max() <= 255 else (raw_frame / 256).astype(np.uint8)
                        color_frame = cv2.cvtColor(frame_uint8, bayer_map.get(bayer_pattern, cv2.COLOR_BAYER_BG2RGB))
                        current_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2GRAY).astype(np.float64)
                except Exception as e:
                    print(f"Debayering failed: {e}. Using raw frame.")
                    current_frame = raw_frame.astype(np.float64)
            else:
                current_frame = raw_frame.astype(np.float64)
            
            # Apply dark subtraction if provided
            if dark_frame is not None and current_frame.shape == dark_frame.shape:
                current_frame = np.clip(current_frame - dark_frame * dark_scale, 0, None)
            
            # Apply enhanced registration
            try:
                aligned_frame, transform = enhanced_registration(
                    current_frame, 
                    reference_frame,
                    detect_rotation=self.registration_config['detect_rotation'],
                    max_rotation=self.registration_config['max_rotation']
                )
                
                # Save debug visualization for the first few frames
                if frame_idx < 5 and self.registration_config['debug_output']:
                    visualize_alignment(
                        normalize_frame(reference_frame), 
                        normalize_frame(current_frame),
                        normalize_frame(aligned_frame)
                    )
                    os.rename('alignment_visualization.png', f'debug/alignment_{frame_filename}.png')
                
                # Decide whether to include the frame based on alignment quality
                if transform['success'] and transform['quality']['ncc'] > 0.6:  # Adjust threshold as needed
                    # Add to stack
                    stacked_image += aligned_frame
                    frames_stacked += 1
                    
                    # Record alignment info
                    alignment_info.append({
                        'filename': frame_filename,
                        'method': transform['method'],
                        'shift_x': transform['shift_x'],
                        'shift_y': transform['shift_y'],
                        'rotation': transform['rotation'],
                        'ncc': transform['quality']['ncc'],
                        'ssim': transform['quality'].get('ssim', 0.0)
                    })
                    
                    print(f"  Stacked using {transform['method']} method. "
                          f"Shift: ({transform['shift_x']:.2f}, {transform['shift_y']:.2f}), "
                          f"Rotation: {transform['rotation']:.3f}°, Quality: {transform['quality']['ncc']:.3f}")
                    
                    # Save intermediate stacks periodically
                    if frames_stacked % 10 == 0:
                        intermediate = stacked_image / frames_stacked
                        fits.writeto(f"debug/stack_{frames_stacked}.fits", intermediate, overwrite=True)
                else:
                    # Reject frame
                    reason = f"Poor alignment quality: {transform['quality']['ncc']:.3f}" if transform['success'] else "Registration failed"
                    rejected_frames.append({
                        'filename': frame_filename,
                        'reason': reason
                    })
                    print(f"  Rejected: {reason}")
            except Exception as e:
                print(f"  Error processing frame: {e}")
                rejected_frames.append({
                    'filename': frame_filename,
                    'reason': f"Error: {str(e)}"
                })
        
        # Normalize stacked image
        if frames_stacked > 0:
            final_stack = stacked_image / frames_stacked
            print(f"Final stack complete with {frames_stacked} frames ({len(rejected_frames)} rejected)")
            
            # Save stacked image
            hdu = fits.PrimaryHDU(final_stack)
            
            # Add stacking metadata
            hdu.header['NFRAMES'] = (frames_stacked, 'Number of frames stacked')
            hdu.header['NREJECT'] = (len(rejected_frames), 'Number of frames rejected')
            hdu.header['STACKMTD'] = ('ENHANCED', 'Enhanced registration')
            hdu.header['REFFRAME'] = (reference_filename, 'Reference frame')
            
            # Add dark calibration info
            if dark_frame is not None:
                hdu.header['DARKSUB'] = (True, 'Dark subtraction applied')
                hdu.header['DARKSCL'] = (dark_scale, 'Dark frame scaling factor')
                if isinstance(dark_frame, str):
                    hdu.header['DARKFILE'] = (os.path.basename(dark_frame), 'Dark frame filename')
            
            # Add creation date
            from datetime import datetime
            hdu.header['DATE'] = (datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'Creation date')
            
            # Save the file
            hdu.writeto(output_path, overwrite=True)
            print(f"Stacked image saved to {output_path}")
            
            # Save alignment report
            try:
                import pandas as pd
                align_df = pd.DataFrame(alignment_info)
                align_df.to_csv("debug/alignment_report.csv", index=False)
                reject_df = pd.DataFrame(rejected_frames)
                if len(reject_df) > 0:
                    reject_df.to_csv("debug/rejected_frames.csv", index=False)
            except Exception as e:
                print(f"Error saving alignment report: {e}")
            
            return final_stack, alignment_info, rejected_frames
        else:
            print("No frames were stacked!")
            return None, alignment_info, rejected_frames
    
    def analyze_rotation_statistics(self, filtered_frames=None):
        """
        Analyze rotation angles across frames to check field derotator performance
        
        Parameters:
        -----------
        filtered_frames : list or None
            List of frames to analyze
            
        Returns:
        --------
        dict
            Dictionary with rotation statistics
        """
        if filtered_frames is None:
            filtered_frames = self.processor.filter_frames()
            
        # Sort frames by quality
        quality_df = self.metrics_df[self.metrics_df['filename'].isin(filtered_frames)]
        sorted_frames = quality_df.sort_values('quality_score', ascending=False)['filename'].tolist()
        
        # Load reference frame
        reference_filename = sorted_frames[0]
        reference_path = os.path.join(self.fits_directory, reference_filename)
        reference_frame = fits.getdata(reference_path).astype(np.float64)
        
        # Normalize reference frame
        ref_norm = normalize_frame(reference_frame)
        
        # Track rotation angles
        rotation_data = []
        
        print("\nAnalyzing Rotation Statistics:")
        print(f"{'Frame':<30} {'Rotation (deg)':<15} {'NCC Quality':<15}")
        print("-" * 60)
        
        # Process each frame
        for frame_idx, frame_filename in enumerate(sorted_frames[1:10]):  # Analyze first 10 frames
            frame_path = os.path.join(self.fits_directory, frame_filename)
            
            # Load frame
            frame = fits.getdata(frame_path).astype(np.float64)
            frame_norm = normalize_frame(frame)
            
            # Detect rotation using enhanced registration
            rotation_angle = detect_rotation_angle(frame_norm, ref_norm, max_angle=2.0, angle_steps=40)
            
            # Apply rotation to check quality
            from skimage.transform import rotate
            frame_rotated = rotate(frame_norm, rotation_angle, preserve_range=True)
            
            # Measure quality
            ncc = calculate_ncc(ref_norm, frame_rotated)
            
            # Store data
            rotation_data.append({
                'filename': frame_filename,
                'rotation_angle': rotation_angle,
                'ncc': ncc
            })
            
            print(f"{frame_filename:<30} {rotation_angle:>9.3f}°      {ncc:>9.3f}")
        
        # Calculate statistics
        angles = [data['rotation_angle'] for data in rotation_data]
        
        stats = {
            'mean_rotation': np.mean(angles),
            'median_rotation': np.median(angles),
            'std_deviation': np.std(angles),
            'min_rotation': np.min(angles),
            'max_rotation': np.max(angles),
            'rotation_range': np.max(angles) - np.min(angles)
        }
        
        print("\nRotation Statistics:")
        print(f"Mean rotation: {stats['mean_rotation']:.3f}°")
        print(f"Median rotation: {stats['median_rotation']:.3f}°")
        print(f"Standard deviation: {stats['std_deviation']:.3f}°")
        print(f"Range: {stats['min_rotation']:.3f}° to {stats['max_rotation']:.3f}° (span: {stats['rotation_range']:.3f}°)")
        
        # Create rotation plot
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(angles)), angles, marker='o')
            plt.axhline(stats['mean_rotation'], color='r', linestyle='--', label=f'Mean: {stats["mean_rotation"]:.3f}°')
            plt.fill_between(
                range(len(angles)),
                [stats['mean_rotation'] - stats['std_deviation']] * len(angles),
                [stats['mean_rotation'] + stats['std_deviation']] * len(angles),
                alpha=0.2, color='r', label=f'±1σ: {stats["std_deviation"]:.3f}°'
            )
            plt.xlabel('Frame Number')
            plt.ylabel('Rotation Angle (degrees)')
            plt.title('Field Rotation Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('debug/rotation_analysis.png')
            plt.close()
            print(f"Rotation analysis plot saved to debug/rotation_analysis.png")
        except:
            print("Could not create rotation plot")
        
        return stats

    def compute_drizzle_params(self, filtered_frames=None):
        """
        Compute optimal drizzle parameters based on the alignment data
        
        Parameters:
        -----------
        filtered_frames : list or None
            List of frames to analyze
            
        Returns:
        --------
        dict
            Dictionary with recommended drizzle parameters
        """
        # Process frames with enhanced registration to get alignment data
        if filtered_frames is None:
            filtered_frames = self.processor.filter_frames()
        
        # Run enhanced registration if not already run
        _, alignment_info, _ = self.process_with_enhanced_registration(
            filtered_frames=filtered_frames,
            output_path="debug/pre_drizzle_stack.fits"
        )
        
        # Extract shifts from alignment info
        shifts_x = [info.get('shift_x', 0) for info in alignment_info if 'shift_x' in info]
        shifts_y = [info.get('shift_y', 0) for info in alignment_info if 'shift_y' in info]
        
        # Calculate statistics
        if len(shifts_x) > 1 and len(shifts_y) > 1:
            # Calculate median absolute deviation as robust measure of jitter
            med_x = np.median(shifts_x)
            med_y = np.median(shifts_y)
            mad_x = np.median(np.abs(np.array(shifts_x) - med_x))
            mad_y = np.median(np.abs(np.array(shifts_y) - med_y))
            
            # Converted MAD to standard deviation equivalent (multiply by 1.4826)
            std_x = 1.4826 * mad_x
            std_y = 1.4826 * mad_y
            
            # Calculate jitter as average of x and y standard deviations
            jitter = (std_x + std_y) / 2
            
            # Calculate recommended drizzle parameters
            pixfrac = max(0.5, min(1.0, 0.7 * jitter))  # Scale pixel fraction with jitter
            scale = min(2.0, max(1.0, 0.5 + jitter))    # Scale output resolution with jitter
            
            if jitter < 0.5:
                # Low jitter - conservative drizzle
                pixfrac = 0.8
                scale = 1.5
            elif jitter >= 0.5 and jitter < 1.0:
                # Medium jitter - moderate drizzle
                pixfrac = 0.7
                scale = 1.5
            else:
                # High jitter - aggressive drizzle
                pixfrac = 0.6
                scale = 2.0
            
            drizzle_params = {
                'pixfrac': pixfrac,
                'scale': scale,
                'jitter_stats': {
                    'median_shift_x': med_x,
                    'median_shift_y': med_y,
                    'std_shift_x': std_x,
                    'std_shift_y': std_y,
                    'jitter': jitter
                }
            }
            
            print("\nDrizzle Parameter Analysis:")
            print(f"Average frame jitter: {jitter:.3f} pixels")
            print(f"Recommended drizzle parameters:")
            print(f"  pixfrac: {pixfrac:.2f}")
            print(f"  scale: {scale:.2f}x")
            
            return drizzle_params
        else:
            print("Not enough alignment data to compute drizzle parameters")
            return {
                'pixfrac': 0.7,
                'scale': 1.5
            }

# Import required functions
from enhanced_registration import detect_rotation_angle, calculate_ncc

# Example usage
if __name__ == "__main__":
    # Import original processor class
    from astro_stacking_script import AstroStackProcessor
    
    # Initialize original processor
    csv_path = './quality_results/quality_metrics.csv'
    fits_directory = './lights/temp_290/'
    
    processor = AstroStackProcessor(csv_path, fits_directory)
    
    # Enhance with new registration methods
    enhanced_processor = EnhancedAstroStackProcessor(processor)
    
    # Process with enhanced registration
    enhanced_processor.process_with_enhanced_registration(
        output_path='enhanced_stacked_image.fits',
        dark_frame='./lights/temp_290/MD-IG_200.0-E10.0s-stellina-5df04c-3072x2080-RGB-session_1.fits',
        dark_scale=1.0,
        debayer=True,
        bayer_pattern='RGGB'
    )
    
    # Analyze rotation statistics
    enhanced_processor.analyze_rotation_statistics()
    
    # Get drizzle parameters
    drizzle_params = enhanced_processor.compute_drizzle_params()
