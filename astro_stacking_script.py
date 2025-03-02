import numpy as np
import pandas as pd
import os
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats
from scipy import ndimage
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from photutils.detection import DAOStarFinder
from photutils.centroids import centroid_2dg

class AstroStackProcessor:
    def __init__(self, csv_path, fits_directory):
        """
        Initialize the stacking processor
        
        Parameters:
        -----------
        csv_path : str
            Path to the quality metrics CSV file
        fits_directory : str
            Directory containing FITS files
        """
        # Load quality metrics
        self.metrics_df = pd.read_csv(csv_path)
        
        # Compute dataset characteristics
        fwhm_max = self.metrics_df['mean_fwhm'].max()
        
        # Set quality thresholds based on dataset characteristics
        self.quality_thresholds = {
            'fwhm': min(10.73, fwhm_max),  # Use best FWHM or dataset max
            'ellipticity': 1.50,
            'snr': max(11.96, self.metrics_df['mean_snr'].quantile(0.25)),
            'quality_score': max(30.60, self.metrics_df['quality_score'].quantile(0.25))
        }
        
        # Additional configuration for flexible stacking
        self.stacking_config = {
            'max_frames_to_stack': 50,      # Limit number of frames to prevent memory issues
            'star_detection_threshold': 5,  # Sigma threshold for star detection
            'alignment_min_stars': 3,       # Minimum stars required for alignment
        }
        
        # Sort frames by quality score in descending order
        self.metrics_df = self.metrics_df.sort_values('quality_score', ascending=False)
        
        # FITS directory
        self.fits_directory = fits_directory
        
        # Stacking parameters
        self.reference_frame = None
        self.stacked_image = None
        
        # Print initialization details
        print("Stacking Processor Initialized:")
        print(f"Total Frames: {len(self.metrics_df)}")
        print("Quality Thresholds:")
        for key, value in self.quality_thresholds.items():
            print(f"  {key}: {value}")
    def progressive_stack_fft(self, filtered_frames, dark_frame=None, dark_scale=1.0, debayer=True, bayer_pattern='RGGB'):
        """
        Progressively stack frames using enhanced FFT alignment with dark subtraction and debayering
        for faint subframes where star detection might be unreliable

        Parameters:
        -----------
        filtered_frames : list
            List of frames to stack
        dark_frame : ndarray or str, optional
            Dark frame for calibration. Can be a numpy array or path to FITS file
        dark_scale : float, optional
            Scaling factor for dark frame if exposure times differ
        debayer : bool, optional
            Whether to perform debayering (demosaicing) on raw frames
        bayer_pattern : str, optional
            Bayer pattern of the camera sensor ('RGGB', 'BGGR', 'GRBG', 'GBRG')

        Returns:
        --------
        stacked_image : ndarray
            Final stacked image
        """
        # Check if OpenCV is available for debayering
        try:
            import cv2
            cv2_available = True
        except ImportError:
            print("Warning: OpenCV not available, falling back to scikit-image for debayering")
            cv2_available = False
            try:
                from skimage.color import demosaicing_bayer
            except ImportError:
                print("Error: scikit-image not available either. Disabling debayering.")
                debayer = False

        # Bayer pattern to OpenCV constant mapping
        bayer_map = {
            'RGGB': cv2.COLOR_BAYER_BG2RGB if cv2_available else 'RGGB',
            'BGGR': cv2.COLOR_BAYER_RG2RGB if cv2_available else 'BGGR',
            'GRBG': cv2.COLOR_BAYER_GB2RGB if cv2_available else 'GRBG',
            'GBRG': cv2.COLOR_BAYER_GR2RGB if cv2_available else 'GBRG'
        }

        # Check if pattern is supported
        if debayer and bayer_pattern not in bayer_map:
            print(f"Warning: Unsupported Bayer pattern '{bayer_pattern}'. Using RGGB.")
            bayer_pattern = 'RGGB'

        # Helper function for debayering
        def debayer_frame(frame):
            if not debayer:
                return frame

            # Ensure frame is of proper type for debayering
            frame_uint8 = frame.astype(np.uint8) if frame.max() <= 255 else (frame / 256).astype(np.uint8)

            if cv2_available:
                # Use OpenCV for debayering (faster and better quality)
                color_frame = cv2.cvtColor(frame_uint8, bayer_map[bayer_pattern])
                # Convert to grayscale for alignment and stacking
                gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2GRAY)
                return gray_frame.astype(np.float64)
            else:
                # Use scikit-image for debayering
                color_frame = demosaicing_bayer(frame_uint8, bayer_map[bayer_pattern])
                # Convert to grayscale (average of RGB channels)
                gray_frame = np.mean(color_frame, axis=2)
                return gray_frame.astype(np.float64)

        # Load dark frame if provided as a path
        if dark_frame is not None and isinstance(dark_frame, str):
            try:
                dark_path = dark_frame
                dark_frame = fits.getdata(dark_path)
                print(f"Loaded dark frame from: {dark_path}")

                # Debayer the dark frame if needed
                if debayer:
                    dark_frame = debayer_frame(dark_frame)
                    print("Debayered dark frame")
            except Exception as e:
                print(f"Error loading dark frame: {e}")
                dark_frame = None
        elif dark_frame is not None and debayer:
            # Debayer the already-loaded dark frame
            dark_frame = debayer_frame(dark_frame)

        # Make sure frames are sorted by quality score
        quality_df = self.metrics_df[self.metrics_df['filename'].isin(filtered_frames)]
        sorted_frames = quality_df.sort_values('quality_score', ascending=False)['filename'].tolist()

        # Select highest quality frame as reference
        reference_filename = sorted_frames[0]
        reference_path = os.path.join(self.fits_directory, reference_filename)
        raw_reference_frame = fits.getdata(reference_path)

        # Apply debayering to reference frame if needed
        if debayer:
            print(f"Debayering images using pattern: {bayer_pattern}")
            raw_reference_frame = debayer_frame(raw_reference_frame)

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

        # Save a copy of the processed reference frame for debugging
        try:
            debug_dir = "debug"
            os.makedirs(debug_dir, exist_ok=True)
            fits.writeto(os.path.join(debug_dir, "reference_processed.fits"), 
                         self.reference_frame, overwrite=True)
            print(f"Saved processed reference frame to {debug_dir}/reference_processed.fits")
        except Exception as e:
            print(f"Warning: Could not save debug frame: {e}")

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

            # Apply debayering if needed
            if debayer:
                raw_frame = debayer_frame(raw_frame)

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

            # Save a copy of the aligned frame for debugging (first few frames only)
            if frames_stacked < 5:
                try:
                    debug_dir = "debug"
                    os.makedirs(debug_dir, exist_ok=True)
                    fits.writeto(os.path.join(debug_dir, f"aligned_{frames_stacked}.fits"), 
                                 aligned_frame, overwrite=True)
                    print(f"Saved aligned frame to {debug_dir}/aligned_{frames_stacked}.fits")
                except Exception as e:
                    print(f"Warning: Could not save debug frame: {e}")

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

                # Save intermediate stacks periodically
                if frames_stacked % 10 == 0:
                    try:
                        debug_dir = "debug"
                        os.makedirs(debug_dir, exist_ok=True)
                        intermediate_stack = stacked_image / frames_stacked
                        fits.writeto(os.path.join(debug_dir, f"stack_{frames_stacked}.fits"), 
                                     intermediate_stack, overwrite=True)
                        print(f"Saved intermediate stack to {debug_dir}/stack_{frames_stacked}.fits")
                    except Exception as e:
                        print(f"Warning: Could not save intermediate stack: {e}")
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

    def process_with_fft(self, output_path='stacked_fft_image.fits', dark_frame=None, dark_scale=1.0, 
                        debayer=True, bayer_pattern='RGGB'):
        """
        Process pipeline using FFT alignment with dark subtraction and debayering

        Parameters:
        -----------
        output_path : str, optional
            Path to save the stacked image
        dark_frame : ndarray or str, optional
            Dark frame for calibration. Can be a numpy array or path to FITS file
        dark_scale : float, optional
            Scaling factor for dark frame if exposure times differ
        debayer : bool, optional
            Whether to perform debayering (demosaicing) on raw frames
        bayer_pattern : str, optional
            Bayer pattern of the camera sensor ('RGGB', 'BGGR', 'GRBG', 'GBRG')

        Returns:
        --------
        tuple
            (stacked_image, alignment_info, rejected_frames)
        """
        # Filter frames based on quality
        filtered_frames = self.filter_frames()
        print(f"Frames to be stacked: {len(filtered_frames)}")

        # Create debug directory
        debug_dir = "debug"
        try:
            os.makedirs(debug_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create debug directory: {e}")

        # Perform FFT-based progressive stacking with dark subtraction
        stacked_image, alignment_info, rejected_frames = self.progressive_stack_fft(
            filtered_frames, dark_frame, dark_scale, debayer, bayer_pattern
        )

        # Save stacked image
        if stacked_image is not None:
            # Create a new FITS HDU with the stacked image
            hdu = fits.PrimaryHDU(stacked_image)

            # Add stacking metadata
            hdu.header['NFRAMES'] = (len(alignment_info) - 1, 'Number of frames stacked')
            hdu.header['NREJECT'] = (len(rejected_frames), 'Number of frames rejected')
            hdu.header['STACKMTD'] = ('FFT', 'Stacking method')

            # Add debayering info
            if debayer:
                hdu.header['DEBAYER'] = (True, 'Debayering applied')
                hdu.header['BAYERPAT'] = (bayer_pattern, 'Bayer pattern used for debayering')
            else:
                hdu.header['DEBAYER'] = (False, 'No debayering applied')

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

            # Generate alignment report CSV
            try:
                import pandas as pd
                report_df = pd.DataFrame(alignment_info + rejected_frames)
                report_df.to_csv(os.path.join(debug_dir, "alignment_report.csv"), index=False)
                print(f"Alignment report saved to {debug_dir}/alignment_report.csv")
            except Exception as e:
                print(f"Warning: Could not save alignment report: {e}")

            return stacked_image, alignment_info, rejected_frames
        else:
            print("No stacked image to save")
            return None, alignment_info, rejected_frames

    def detect_bayer_pattern(self, fits_path):
        """
        Try to detect the Bayer pattern from FITS header

        Parameters:
        -----------
        fits_path : str
            Path to a FITS file

        Returns:
        --------
        str
            Detected Bayer pattern or 'RGGB' as default
        """
        try:
            header = fits.getheader(fits_path)

            # Check common Bayer pattern keywords
            for keyword in ['BAYERPAT', 'COLORTYP', 'BAYERALG', 'FILTER']:
                if keyword in header:
                    pattern = header[keyword]
                    # Some cameras store it as "BAYER_RGGB" or similar
                    if isinstance(pattern, str) and "RGGB" in pattern:
                        return "RGGB"
                    elif isinstance(pattern, str) and "BGGR" in pattern:
                        return "BGGR"
                    elif isinstance(pattern, str) and "GRBG" in pattern:
                        return "GRBG"
                    elif isinstance(pattern, str) and "GBRG" in pattern:
                        return "GBRG"

            # Try telescope or camera model to make an educated guess
            telescope = header.get('TELESCOP', '').lower()
            instrume = header.get('INSTRUME', '').lower()

            # Add known camera models and their Bayer patterns
            # This is just an example - would need to be expanded with real data
            if 'zwo' in instrume or 'asi' in instrume:
                return 'RGGB'  # Common for ZWO cameras
            elif 'qhy' in instrume:
                return 'RGGB'  # Common for QHY cameras

            # Default to RGGB which is most common
            return 'RGGB'
        except Exception as e:
            print(f"Error detecting Bayer pattern: {e}")
            return 'RGGB'  # Default to most common pattern
        
    def analyze_pointing_accuracy(self, filtered_frames, output_csv='pointing_accuracy.csv'):
        """
        Analyze pointing accuracy by comparing frames using enhanced FFT alignment
        and WCS header information

        Parameters:
        -----------
        filtered_frames : list
            List of FITS filenames to analyze
        output_csv : str, optional
            Path to save CSV results

        Returns:
        --------
        analysis_results : DataFrame
            DataFrame containing analysis results
        """
        # Sort frames by quality
        quality_df = self.metrics_df[self.metrics_df['filename'].isin(filtered_frames)]
        sorted_frames = quality_df.sort_values('quality_score', ascending=False)['filename'].tolist()

        print("\nPointing Accuracy Analysis")
        print("=" * 80)
        print(f"{'Frame':<25} {'FFT Offset (px)':<20} {'Offset (arcsec)':<15} {'Confidence':<10} {'RA/DEC from WCS'}")
        print("-" * 80)

        # Select reference frame
        reference_filename = sorted_frames[0]
        reference_path = os.path.join(self.fits_directory, reference_filename)
        reference_frame = fits.getdata(reference_path)
        reference_header = fits.getheader(reference_path)

        # Extract reference RA/DEC
        ref_ra = reference_header.get('CRVAL1', reference_header.get('RA', None))
        ref_dec = reference_header.get('CRVAL2', reference_header.get('DEC', None))

        # Initialize results storage
        results = []

        # Record reference frame info
        results.append({
            'filename': reference_filename,
            'dx_pixels': 0.0,
            'dy_pixels': 0.0,
            'offset_pixels': 0.0,
            'ra_offset_arcsec': 0.0,
            'dec_offset_arcsec': 0.0,
            'total_offset_arcsec': 0.0,
            'confidence': 100.0,
            'ra': ref_ra,
            'dec': ref_dec,
            'ra_dec_diff_arcsec': 0.0,
            'is_reference': True
        })

        print(f"{reference_filename:<25} {'0.00, 0.00':<20} {'0.00':<15} {'REF':<10} {ref_ra:.6f}, {ref_dec:.6f}")

        # Analyze each frame
        for frame_name in sorted_frames[1:]:
            frame_path = os.path.join(self.fits_directory, frame_name)
            frame = fits.getdata(frame_path)
            frame_header = fits.getheader(frame_path)

            # Get RA/DEC from header
            frame_ra = frame_header.get('CRVAL1', frame_header.get('RA', None))
            frame_dec = frame_header.get('CRVAL2', frame_header.get('DEC', None))

            # Calculate RA/DEC difference in arcseconds if available
            ra_dec_diff_arcsec = None
            if ref_ra is not None and ref_dec is not None and frame_ra is not None and frame_dec is not None:
                # Convert RA difference to arcseconds (accounting for cos(dec) factor)
                ra_diff_deg = (frame_ra - ref_ra) * np.cos(np.radians(ref_dec))
                dec_diff_deg = frame_dec - ref_dec

                # Convert to arcseconds
                ra_diff_arcsec = ra_diff_deg * 3600
                dec_diff_arcsec = dec_diff_deg * 3600

                # Total angular separation
                ra_dec_diff_arcsec = np.sqrt(ra_diff_arcsec**2 + dec_diff_arcsec**2)

            # Perform enhanced FFT alignment
            _, (dy, dx), metadata = self.align_frame_fft_enhanced(frame, reference_frame)

            # Calculate pixel offset
            pixel_offset = np.sqrt(dx**2 + dy**2)

            # Get RA/DEC offsets from metadata
            ra_offset_arcsec = metadata['wcs']['ra_offset_arcsec']
            dec_offset_arcsec = metadata['wcs']['dec_offset_arcsec']
            total_offset_arcsec = metadata['wcs']['total_offset_arcsec']

            # Store results
            results.append({
                'filename': frame_name,
                'dx_pixels': dx,
                'dy_pixels': dy,
                'offset_pixels': pixel_offset,
                'ra_offset_arcsec': ra_offset_arcsec,
                'dec_offset_arcsec': dec_offset_arcsec,
                'total_offset_arcsec': total_offset_arcsec,
                'confidence': metadata['confidence'],
                'ra': frame_ra,
                'dec': frame_dec,
                'ra_dec_diff_arcsec': ra_dec_diff_arcsec,
                'is_reference': False
            })

            # Format for printing
            confidence_str = f"{metadata['confidence']:.1f}%"
            print(f"{frame_name:<25} {dx:.2f}, {dy:.2f}{' ':<10} {total_offset_arcsec:.2f}{' ':<8} "
                  f"{confidence_str:<10} {frame_ra:.6f}, {frame_dec:.6f}")

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Calculate statistics
        non_ref_df = results_df[~results_df['is_reference']]

        median_offset_px = non_ref_df['offset_pixels'].median()
        mean_offset_px = non_ref_df['offset_pixels'].mean()
        std_offset_px = non_ref_df['offset_pixels'].std()
        max_offset_px = non_ref_df['offset_pixels'].max()

        median_offset_arcsec = non_ref_df['total_offset_arcsec'].median()
        mean_offset_arcsec = non_ref_df['total_offset_arcsec'].mean()
        std_offset_arcsec = non_ref_df['total_offset_arcsec'].std()
        max_offset_arcsec = non_ref_df['total_offset_arcsec'].max()

        # Calculate RMS error
        rms_error_px = np.sqrt(np.mean(non_ref_df['offset_pixels']**2))
        rms_error_arcsec = np.sqrt(np.mean(non_ref_df['total_offset_arcsec']**2))

        # Print summary statistics
        print("\nPointing Accuracy Statistics")
        print("=" * 50)
        print(f"Number of frames analyzed: {len(non_ref_df)}")
        print(f"Median offset: {median_offset_px:.2f} px ({median_offset_arcsec:.2f} arcsec)")
        print(f"Mean offset: {mean_offset_px:.2f} px ({mean_offset_arcsec:.2f} arcsec)")
        print(f"Standard deviation: {std_offset_px:.2f} px ({std_offset_arcsec:.2f} arcsec)")
        print(f"Maximum offset: {max_offset_px:.2f} px ({max_offset_arcsec:.2f} arcsec)")
        print(f"RMS pointing error: {rms_error_px:.2f} px ({rms_error_arcsec:.2f} arcsec)")

        # If WCS RA/DEC available, compare with FFT-derived offsets
        if all(~non_ref_df['ra_dec_diff_arcsec'].isna()):
            wcs_median = non_ref_df['ra_dec_diff_arcsec'].median()
            wcs_mean = non_ref_df['ra_dec_diff_arcsec'].mean()
            wcs_std = non_ref_df['ra_dec_diff_arcsec'].std()
            wcs_max = non_ref_df['ra_dec_diff_arcsec'].max()

            print("\nWCS-derived Pointing Statistics")
            print(f"Median WCS offset: {wcs_median:.2f} arcsec")
            print(f"Mean WCS offset: {wcs_mean:.2f} arcsec")
            print(f"Standard deviation: {wcs_std:.2f} arcsec")
            print(f"Maximum offset: {wcs_max:.2f} arcsec")

            # Compare FFT with WCS
            print("\nFFT vs WCS Comparison")
            print(f"Mean difference: {abs(mean_offset_arcsec - wcs_mean):.2f} arcsec")

        # Save results to CSV
        if output_csv:
            results_df.to_csv(output_csv, index=False)
            print(f"\nResults saved to {output_csv}")

        return results_df            
    def align_frame_fft_enhanced(self, frame, reference_frame):
        """
        Align a frame to a reference frame using enhanced FFT-based phase correlation
        with sub-pixel precision

        Parameters:
        -----------
        frame : ndarray
            Input frame to align
        reference_frame : ndarray
            Reference frame to align to

        Returns:
        --------
        aligned_frame : ndarray
            Aligned frame
        shift : tuple
            (y_shift, x_shift) alignment parameters
        metadata : dict
            Additional alignment metadata including pixel scale and RA/DEC offsets
        """
        try:
            # Ensure both frames are float64
            frame = frame.astype(np.float64)
            reference = reference_frame.astype(np.float64)

            # Image preprocessing to enhance features
            def preprocess_image(img):
                # Remove background
                bg = np.percentile(img, 10)  # Use a low percentile as background estimate
                img_bg = img - bg

                # Clip negative values
                img_bg = np.clip(img_bg, 0, None)

                # Apply a mild stretch to enhance contrast
                img_norm = np.power(img_bg / (np.percentile(img_bg, 99) + 1e-10), 0.5)

                # Apply Gaussian blur to reduce noise (optional)
                from scipy.ndimage import gaussian_filter
                img_smooth = gaussian_filter(img_norm, sigma=1.0)

                return img_smooth

            # Preprocess both images
            frame_proc = preprocess_image(frame)
            ref_proc = preprocess_image(reference)

            # Apply window function to reduce edge effects
            height, width = frame_proc.shape
            y = np.hanning(height).reshape(-1, 1)
            x = np.hanning(width).reshape(1, -1)
            window = y * x

            frame_windowed = frame_proc * window
            ref_windowed = ref_proc * window

            # Compute FFT of both images
            f1 = np.fft.fft2(frame_windowed)
            f2 = np.fft.fft2(ref_windowed)

            # Compute cross-power spectrum
            cross_power = f1 * np.conj(f2)
            abs_cross_power = np.abs(cross_power) + 1e-10
            normalized_cross_power = cross_power / abs_cross_power

            # Compute inverse FFT to get correlation
            correlation = np.fft.ifft2(normalized_cross_power)
            correlation_abs = np.abs(correlation)

            # Find maximum correlation
            y_max, x_max = np.unravel_index(np.argmax(correlation_abs), correlation.shape)

            # Adjust for FFT's circular nature
            y_shift = y_max if y_max < height // 2 else y_max - height
            x_shift = x_max if x_max < width // 2 else x_max - width

            # Sub-pixel refinement using 2D Gaussian fit around peak
            # Extract a small region around the peak
            region_size = 5
            y_min = max(0, y_max - region_size)
            y_max = min(height - 1, y_max + region_size)
            x_min = max(0, x_max - region_size)
            x_max = min(width - 1, x_max + region_size)

            # Extract region and create coordinate grid
            region = correlation_abs[y_min:y_max+1, x_min:x_max+1]
            y_grid, x_grid = np.mgrid[y_min:y_max+1, x_min:x_max+1]

            # Fit 2D Gaussian using weighted centroid method
            total = np.sum(region)
            if total > 0:
                y_centroid = np.sum(y_grid * region) / total
                x_centroid = np.sum(x_grid * region) / total

                # Refine integer shifts to sub-pixel precision
                if y_centroid < height // 2:
                    refined_y_shift = y_centroid
                else:
                    refined_y_shift = y_centroid - height

                if x_centroid < width // 2:
                    refined_x_shift = x_centroid
                else:
                    refined_x_shift = x_centroid - width
            else:
                refined_y_shift = y_shift
                refined_x_shift = x_shift

            # Calculate confidence based on peak sharpness
            max_val = np.max(correlation_abs)
            mean_val = np.mean(correlation_abs)
            std_val = np.std(correlation_abs)
            peak_significance = (max_val - mean_val) / std_val
            confidence = min(1.0, peak_significance / 10.0) * 100  # Scale to percentage

            # Apply shift using scipy's shift function for subpixel precision
            from scipy.ndimage import shift as ndimage_shift
            aligned_frame = ndimage_shift(frame, (refined_y_shift, refined_x_shift), 
                                          order=3, mode='constant', cval=0)

            # Get WCS information if available
            wcs_data = {}
            try:
                # Try to get WCS headers from both frames
                ref_header = fits.getheader(os.path.join(self.fits_directory, 
                                            self.metrics_df.sort_values('quality_score', 
                                                                       ascending=False)['filename'].iloc[0]))

                # Extract pixel scale from CDELT keywords if available
                if 'CDELT1' in ref_header and 'CDELT2' in ref_header:
                    # CDELT values are in degrees per pixel
                    cdelt1 = abs(ref_header.get('CDELT1', 0)) * 3600  # Convert to arcsec/pixel
                    cdelt2 = abs(ref_header.get('CDELT2', 0)) * 3600  # Convert to arcsec/pixel
                    pixel_scale = (cdelt1 + cdelt2) / 2  # Average of both axes
                elif 'CD1_1' in ref_header and 'CD2_2' in ref_header:
                    # CD matrix elements are in degrees per pixel
                    cd1_1 = abs(ref_header.get('CD1_1', 0)) * 3600
                    cd2_2 = abs(ref_header.get('CD2_2', 0)) * 3600
                    pixel_scale = (cd1_1 + cd2_2) / 2
                else:
                    # Default assumption for pixel scale if not found
                    pixel_scale = 1.0  # Generic 1 arcsec/pixel

                # Calculate RA/DEC offset in arcseconds
                ra_offset_arcsec = refined_x_shift * pixel_scale
                dec_offset_arcsec = refined_y_shift * pixel_scale

                wcs_data = {
                    'pixel_scale': pixel_scale,
                    'ra_offset_arcsec': ra_offset_arcsec,
                    'dec_offset_arcsec': dec_offset_arcsec,
                    'total_offset_arcsec': np.sqrt(ra_offset_arcsec**2 + dec_offset_arcsec**2)
                }
            except Exception as e:
                print(f"WCS data extraction error: {e}")
                wcs_data = {
                    'pixel_scale': 1.0,  # Assumed default
                    'ra_offset_arcsec': refined_x_shift,  # Raw pixel value
                    'dec_offset_arcsec': refined_y_shift,  # Raw pixel value
                    'total_offset_arcsec': np.sqrt(refined_x_shift**2 + refined_y_shift**2)
                }

            # Prepare detailed metadata
            metadata = {
                'integer_shift': (y_shift, x_shift),
                'refined_shift': (refined_y_shift, refined_x_shift),
                'confidence': confidence,
                'correlation_peak': max_val,
                'correlation_significance': peak_significance,
                'wcs': wcs_data
            }

            print(f"Enhanced FFT alignment: dx={refined_x_shift:.2f}, dy={refined_y_shift:.2f} " 
                  f"(confidence: {confidence:.1f}%)")
            if 'total_offset_arcsec' in wcs_data:
                print(f"Estimated pointing offset: {wcs_data['total_offset_arcsec']:.2f} arcsec")

            return aligned_frame, (refined_y_shift, refined_x_shift), metadata

        except Exception as e:
            print(f"Enhanced FFT alignment error: {e}")
            import traceback
            traceback.print_exc()
            return frame, (0, 0), {'error': str(e)}
    
    def validate_wcs_headers(self, filtered_frames):
        """
        Read and compare CRVAL1/CRVAL2 (RA/DEC) values from FITS headers as a sanity check

        Parameters:
        -----------
        filtered_frames : list
            List of frames to check
        """
        # Sort frames by quality
        quality_df = self.metrics_df[self.metrics_df['filename'].isin(filtered_frames)]
        sorted_frames = quality_df.sort_values('quality_score', ascending=False)['filename'].tolist()

        print("\nFITS Header WCS Validation")
        print("=" * 60)
        print(f"{'Frame':<25} {'CRVAL1 (RA)':<15} {'CRVAL2 (DEC)':<15} {'Offset from Ref (arcmin)':<20}")
        print("-" * 60)

        # Track reference values
        ref_ra, ref_dec = None, None

        # Check headers for each frame
        for i, frame_name in enumerate(sorted_frames[:10]):  # Check first 10 frames
            frame_path = os.path.join(self.fits_directory, frame_name)

            try:
                # Read header only
                header = fits.getheader(frame_path)

                # Extract WCS coordinates if they exist
                ra = header.get('CRVAL1', None)
                dec = header.get('CRVAL2', None)

                # Alternative keywords to check if primary ones aren't found
                if ra is None:
                    ra = header.get('RA', header.get('OBJCTRA', None))
                if dec is None:
                    dec = header.get('DEC', header.get('OBJCTDEC', None))

                # Convert string values if needed
                if isinstance(ra, str) and ':' in ra:
                    # Convert from HH:MM:SS format to degrees
                    h, m, s = map(float, ra.split(':'))
                    ra = 15 * (h + m/60 + s/3600)  # 15 deg per hour of RA

                if isinstance(dec, str) and ':' in dec:
                    # Convert from DD:MM:SS format to degrees
                    sign = -1 if dec.startswith('-') else 1
                    dec = dec.lstrip('-+')
                    d, m, s = map(float, dec.split(':'))
                    dec = sign * (d + m/60 + s/3600)

                # Calculate offset from reference frame
                offset_str = "N/A"
                if i == 0:
                    # Set reference values from first frame
                    ref_ra, ref_dec = ra, dec
                    offset_str = "REFERENCE"
                elif ra is not None and dec is not None and ref_ra is not None and ref_dec is not None:
                    # Simple angular separation in arcminutes
                    # Note: This is approximate and doesn't account for cos(dec) factor for RA
                    ra_diff = (ra - ref_ra) * 60  # Convert to arcmin
                    dec_diff = (dec - ref_dec) * 60  # Convert to arcmin
                    offset = np.sqrt(ra_diff**2 + dec_diff**2)
                    offset_str = f"{offset:.2f}"

                # Format values for printing
                ra_str = f"{ra:.6f}" if ra is not None else "Not found"
                dec_str = f"{dec:.6f}" if dec is not None else "Not found"

                print(f"{frame_name:<25} {ra_str:<15} {dec_str:<15} {offset_str:<20}")

            except Exception as e:
                print(f"{frame_name:<25} Error reading header: {str(e)}")

        print("=" * 60)
        print("Note: Offsets are approximate and given in arcminutes.")
    
    def align_frame_fft(self, frame, reference_frame):
        """
        Align a frame to a reference frame using improved FFT-based phase correlation

        Parameters:
        -----------
        frame : ndarray
            Input frame to align
        reference_frame : ndarray
            Reference frame to align to

        Returns:
        --------
        aligned_frame : ndarray
            Aligned frame
        shift : tuple
            (y_shift, x_shift) alignment parameters
        """
        try:
            # Ensure both frames are float64
            frame = frame.astype(np.float64)
            reference = reference_frame.astype(np.float64)

            # Basic preprocessing steps
            # 1. Background subtraction
            frame_bg = np.median(frame)
            ref_bg = np.median(reference)
            frame_proc = frame - frame_bg
            ref_proc = reference - ref_bg

            # 2. Simple intensity scaling and clip negative values
            frame_proc = np.clip(frame_proc, 0, None)
            ref_proc = np.clip(ref_proc, 0, None)

            # 3. Log transform to enhance features
            frame_proc = np.log1p(frame_proc)
            ref_proc = np.log1p(ref_proc)

            # 4. Normalize values to 0-1 range
            frame_proc = (frame_proc - np.min(frame_proc)) / (np.max(frame_proc) - np.min(frame_proc) + 1e-10)
            ref_proc = (ref_proc - np.min(ref_proc)) / (np.max(ref_proc) - np.min(ref_proc) + 1e-10)

            # Apply window function to reduce edge effects
            height, width = frame_proc.shape
            window = np.outer(
                np.hanning(height),
                np.hanning(width)
            )
            frame_windowed = frame_proc * window
            ref_windowed = ref_proc * window

            # Compute FFT of both images
            f1 = np.fft.fft2(frame_windowed)
            f2 = np.fft.fft2(ref_windowed)

            # Compute cross-power spectrum (phase correlation)
            cross_power = f1 * np.conj(f2)
            cross_power = cross_power / (np.abs(cross_power) + 1e-10)

            # Compute inverse FFT to get correlation
            correlation = np.abs(np.fft.ifft2(cross_power))

            # Find maximum correlation
            y_max, x_max = np.unravel_index(np.argmax(correlation), correlation.shape)

            # Calculate shifts (handle FFT's circular nature)
            y_shift = y_max if y_max < height // 2 else y_max - height
            x_shift = x_max if x_max < width // 2 else x_max - width

            print(f"FFT alignment shift: dx={x_shift}, dy={y_shift}")

            # Apply shift using scipy's shift function
            from scipy.ndimage import shift as ndimage_shift
            aligned_frame = ndimage_shift(frame, (y_shift, x_shift), order=1, mode='constant', cval=0)

            return aligned_frame, (y_shift, x_shift)

        except Exception as e:
            print(f"FFT alignment error: {e}")
            import traceback
            traceback.print_exc()
            return frame, (0, 0)            

    def validate_alignment(self, filtered_frames):
        """
        Compare star-based and FFT-based alignment for validation

        Parameters:
        -----------
        filtered_frames : list
            List of frames to compare alignment methods
        """
        # Sort frames by quality
        quality_df = self.metrics_df[self.metrics_df['filename'].isin(filtered_frames)]
        sorted_frames = quality_df.sort_values('quality_score', ascending=False)['filename'].tolist()

        # Select reference frame (highest quality)
        reference_filename = sorted_frames[0]
        reference_path = os.path.join(self.fits_directory, reference_filename)
        reference_frame = fits.getdata(reference_path)

        # Detect stars in reference frame
        ref_stars = self.detect_alignment_stars(reference_frame)

        print(f"\nAlignment Validation with Reference: {reference_filename}")
        print("=" * 60)
        print(f"{'Frame':<25} {'Star-based dx,dy':<20} {'FFT-based dx,dy':<20} {'Difference':<15}")
        print("-" * 60)

        # Test alignment with a few frames
        for frame_name in sorted_frames[1:6]:  # First 5 frames after reference
            # Load frame
            frame_path = os.path.join(self.fits_directory, frame_name)
            current_frame = fits.getdata(frame_path)

            # Star-based alignment
            target_stars = self.detect_alignment_stars(current_frame)
            if target_stars is not None and ref_stars is not None:
                # Extract star coordinates
                ref_coords = np.array([(float(star['xcentroid']), float(star['ycentroid'])) for star in ref_stars])
                target_coords = np.array([(float(star['xcentroid']), float(star['ycentroid'])) for star in target_stars])

                # Calculate mean shift
                mean_ref = np.mean(ref_coords, axis=0)
                mean_target = np.mean(target_coords, axis=0)
                star_shift = mean_ref - mean_target
                star_dx, star_dy = star_shift[0], star_shift[1]
            else:
                star_dx, star_dy = np.nan, np.nan

            # FFT-based alignment
            _, (fft_dy, fft_dx) = self.align_frame_fft(current_frame, reference_frame)

            # Calculate difference if both methods worked
            if not np.isnan(star_dx) and not np.isnan(fft_dx):
                diff = np.sqrt((star_dx - fft_dx)**2 + (star_dy - fft_dy)**2)
                diff_str = f"{diff:.2f} px"
            else:
                diff_str = "N/A"

            # Print comparison
            print(f"{frame_name:<25} {star_dx:.2f},{star_dy:.2f}{' ':>10} {fft_dx:.2f},{fft_dy:.2f}{' ':>10} {diff_str:<15}")

        print("=" * 60)
        
    def filter_frames(self):
        """
        Adaptive frame filtering with progressive quality relaxation
        
        Returns:
        --------
        filtered_frames : list
            List of filenames meeting quality criteria
        """
        # Print initial dataset statistics
        print("Initial Dataset Quality Check:")
        print(f"Total Frames: {len(self.metrics_df)}")
        print(f"FWHM Range: {self.metrics_df['mean_fwhm'].min():.2f} - {self.metrics_df['mean_fwhm'].max():.2f}")
        print(f"SNR Range: {self.metrics_df['mean_snr'].min():.2f} - {self.metrics_df['mean_snr'].max():.2f}")
        print(f"Quality Score Range: {self.metrics_df['quality_score'].min():.2f} - {self.metrics_df['quality_score'].max():.2f}")
        
        # Progressive filtering stages
        filtering_stages = [
            {
                'name': 'Relaxed FWHM Filtering',
                'filters': {
                    'mean_fwhm': ('<=', self.quality_thresholds['fwhm'] * 1.5),
                    'mean_ellipticity': ('<=', self.quality_thresholds['ellipticity'] * 1.2),
                    'mean_snr': ('>=', self.quality_thresholds['snr'] * 0.9),
                    'quality_score': ('>=', self.quality_thresholds['quality_score'] * 0.9)
                }
            },
            {
                'name': 'Minimal Quality Filtering',
                'filters': {
                    'mean_snr': ('>=', self.quality_thresholds['snr'] * 0.7),
                    'quality_score': ('>=', self.quality_thresholds['quality_score'] * 0.7)
                }
            }
        ]
        
        # Try each filtering stage
        for stage in filtering_stages:
            # Construct the filter conditions
            filter_conditions = []
            for column, (comparison, threshold) in stage['filters'].items():
                if comparison == '<=':
                    condition = self.metrics_df[column] <= threshold
                elif comparison == '>=':
                    condition = self.metrics_df[column] >= threshold
                else:
                    raise ValueError(f"Unsupported comparison: {comparison}")
                filter_conditions.append(condition)
            
            # Combine conditions
            combined_condition = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_condition &= condition
            
            # Apply filtering
            filtered_df = self.metrics_df[combined_condition]
            
            # Print stage results
            print(f"\n{stage['name']}:")
            print(f"Frames meeting criteria: {len(filtered_df)}")
            
            # If frames found, return their filenames
            if len(filtered_df) > 0:
                # Sort by quality score and limit to max frames
                filtered_df = filtered_df.nlargest(
                    min(len(filtered_df), self.stacking_config['max_frames_to_stack']), 
                    'quality_score'
                )
                
                print("Selected frames:")
                for _, row in filtered_df.head().iterrows():
                    print(f"  {row['filename']} (FWHM: {row['mean_fwhm']:.2f}, SNR: {row['mean_snr']:.2f}, Quality: {row['quality_score']:.2f})")
                
                return filtered_df['filename'].tolist()
        
        # If no frames found in any stage
        print("\nWARNING: No frames meet even minimal quality criteria!")
        print("Selecting top frames by quality score")
        
        # Fallback: select top frames by quality score
        fallback_frames = self.metrics_df.nlargest(
            min(20, len(self.metrics_df)), 
            'quality_score'
        )
        
        print("Fallback frames:")
        for _, row in fallback_frames.iterrows():
            print(f"  {row['filename']} (FWHM: {row['mean_fwhm']:.2f}, SNR: {row['mean_snr']:.2f}, Quality: {row['quality_score']:.2f})")
        
        return fallback_frames['filename'].tolist()
    
    def detect_alignment_stars(self, frame, n_stars=25):
        """
        Detect stars for alignment using DAOStarFinder with improved centroid refinement

        Parameters:
        -----------
        frame : ndarray
            Input image frame
        n_stars : int, optional
            Target number of stars to detect

        Returns:
        --------
        stars : astropy.table.Table or None
            Detected star coordinates
        """
        try:
            # Ensure frame is a numpy array of float type
            frame = frame.astype(np.float64)

            # Calculate background statistics with more robust clipping
            mean, median, std = sigma_clipped_stats(frame, sigma=3.0, maxiters=5)

            # Estimate FWHM from dataset metrics
            est_fwhm = self.quality_thresholds.get('fwhm', 3.0)

            # Create star finder with adjusted parameters
            daofind = DAOStarFinder(
                fwhm=est_fwhm, 
                threshold=self.stacking_config['star_detection_threshold'] * std,
                sharplo=0.2,  # Lower limit for sharpness
                sharphi=1.0,  # Upper limit for sharpness
                roundlo=0.0,  # Lower limit for roundness
                roundhi=0.7   # Upper limit for roundness
            )

            # Detect stars
            sources = daofind(frame - median)

            if sources is None or len(sources) == 0:
                print("Warning: No stars detected in frame.")
                return None

            # Sort by peak value and take strongest stars
            sources.sort('peak', reverse=True)
            if len(sources) > 50:  # Take top 50 peaks
                sources = sources[:50]

            # Refine star centroids
            valid_sources = []

            for source in sources:
                try:
                    # Extract x and y for clarity (integer positions)
                    x_pos = int(source['xcentroid'])
                    y_pos = int(source['ycentroid'])

                    # Ensure position is within bounds with good margin
                    if (x_pos < 10 or y_pos < 10 or 
                        x_pos >= frame.shape[1] - 10 or y_pos >= frame.shape[0] - 10):
                        continue

                    # Use a fixed size cutout for all stars (15x15 pixels)
                    size = 15
                    half_size = size // 2

                    # Extract region ensuring it's within bounds
                    y_min = max(0, y_pos - half_size)
                    y_max = min(frame.shape[0], y_pos + half_size + 1)
                    x_min = max(0, x_pos - half_size)
                    x_max = min(frame.shape[1], x_pos + half_size + 1)

                    # Skip if the region is too small
                    if (y_max - y_min) < 7 or (x_max - x_min) < 7:
                        continue

                    # Extract region
                    cutout = frame[y_min:y_max, x_min:x_max].copy()

                    # Skip if region has NaN values or is too flat
                    if np.any(np.isnan(cutout)) or np.std(cutout) < 0.5 * std:
                        continue

                    # Background subtract the cutout to improve fitting
                    cutout_mean = np.mean(cutout[0, :].tolist() + cutout[-1, :].tolist() + 
                                         cutout[:, 0].tolist() + cutout[:, -1].tolist())
                    cutout -= cutout_mean

                    # Skip if max value is too low
                    if np.max(cutout) < 5 * std:
                        continue

                    # Calculate centroid on the cutout with error silencing
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        y_centroid, x_centroid = centroid_2dg(cutout)

                    # Ensure centroid is within the cutout
                    if (np.isnan(x_centroid) or np.isnan(y_centroid) or
                        x_centroid < 0 or y_centroid < 0 or
                        x_centroid >= cutout.shape[1] or y_centroid >= cutout.shape[0]):
                        continue

                    # Adjust centroid back to full frame coordinates
                    source['xcentroid'] = float(x_min + x_centroid)
                    source['ycentroid'] = float(y_min + y_centroid)
                    valid_sources.append(source)
                except Exception as e:
                    pass  # Silently skip problematic stars

            if not valid_sources:
                print("No valid star centroids found.")
                return None

            # Convert to table and sort by flux
            from astropy.table import Table
            sources_table = Table(valid_sources)
            sources_table.sort('flux', reverse=True)

            # Limit to requested number of stars
            return sources_table[:min(n_stars, len(sources_table))]

        except Exception as e:
            print(f"Comprehensive star detection error: {e}")
            return None
        
    def align_frame(self, frame, ref_stars, target_stars):
        """
        Align a single frame to the reference frame

        Parameters:
        -----------
        frame : ndarray
            Input frame to align
        ref_stars : Table
            Reference stars
        target_stars : Table
            Target frame stars

        Returns:
        --------
        aligned_frame : ndarray
            Geometrically aligned frame
        """
        # Check for valid star detection
        if ref_stars is None or target_stars is None:
            print("Warning: No stars detected for alignment. Returning original frame.")
            return frame

        # Ensure we have enough stars for alignment
        min_stars = min(len(ref_stars), len(target_stars))
        if min_stars < self.stacking_config['alignment_min_stars']:
            print(f"Insufficient stars for alignment (ref: {len(ref_stars)}, target: {len(target_stars)}). Returning original frame.")
            return frame

        try:
            # Truncate to the minimum number of stars
            ref_stars = ref_stars[:min_stars]
            target_stars = target_stars[:min_stars]

            # Extract star coordinates and ensure they are floating point
            ref_coords = np.array([(float(star['xcentroid']), float(star['ycentroid'])) for star in ref_stars])
            target_coords = np.array([(float(star['xcentroid']), float(star['ycentroid'])) for star in target_stars])

            # Simple shift based on mean positions (fallback method)
            mean_ref = np.mean(ref_coords, axis=0)
            mean_target = np.mean(target_coords, axis=0)
            shift = mean_ref - mean_target

            # Create a simple transformation matrix for the shift
            shift_y, shift_x = shift[1], shift[0]

            # Use a simple coordinate shift for alignment
            def simple_mapping(coordinates):
                """Maps coordinates with a simple shift"""
                # The input coordinates are indices: [y, x]
                y, x = coordinates[0], coordinates[1]

                # Apply shift in reverse (output coords -> input coords)
                input_y = y - shift_y
                input_x = x - shift_x

                # Return mapped coordinates
                return input_y, input_x

            # Apply the transformation
            print(f"Aligning with shift: dx={shift_x:.2f}, dy={shift_y:.2f}")
            aligned_frame = ndimage.geometric_transform(
                frame,
                simple_mapping,
                output_shape=frame.shape,
                order=1  # Linear interpolation
            )

            return aligned_frame

        except Exception as e:
            print(f"Frame alignment error: {e}")
            import traceback
            traceback.print_exc()
            return frame

    def progressive_stack(self, filtered_frames):
        """
        Progressively stack frames while monitoring quality

        Parameters:
        -----------
        filtered_frames : list
            List of frames to stack

        Returns:
        --------
        stacked_image : ndarray
            Final stacked image
        """
        # Make sure frames are sorted by quality score
        quality_df = self.metrics_df[self.metrics_df['filename'].isin(filtered_frames)]
        sorted_frames = quality_df.sort_values('quality_score', ascending=False)['filename'].tolist()

        # Select highest quality frame as reference
        reference_filename = sorted_frames[0]
        reference_path = os.path.join(self.fits_directory, reference_filename)
        self.reference_frame = fits.getdata(reference_path)

        # Detect reference frame stars
        ref_stars = self.detect_alignment_stars(self.reference_frame)

        # Initialize stacked image with reference frame
        stacked_image = self.reference_frame.astype(np.float64)
        frames_stacked = 1

        # Track quality metrics of reference frame
        current_snr = self.metrics_df.loc[self.metrics_df['filename'] == reference_filename, 'mean_snr'].values[0]
        base_snr = current_snr  # Store initial SNR for relative comparison

        print(f"Started stacking with reference frame: {reference_filename} (SNR: {current_snr:.2f})")

        # Keep track of consecutive poor quality frames
        poor_quality_count = 0
        max_poor_quality_frames = 3  # Allow up to 3 consecutive poor quality frames

        # Iterate through remaining frames
        for frame_name in filtered_frames[1:]:
            # Read frame
            frame_path = os.path.join(self.fits_directory, frame_name)
            current_frame = fits.getdata(frame_path)

            # Get frame's SNR
            new_snr = self.metrics_df.loc[self.metrics_df['filename'] == frame_name, 'mean_snr'].values[0]

            # Detect stars in current frame
            target_stars = self.detect_alignment_stars(current_frame)

            # Align frame
            aligned_frame = self.align_frame(current_frame, ref_stars, target_stars)

            # More lenient quality check - only skip frames that are significantly worse
            if new_snr >= base_snr * 0.5:  # Lower threshold to 50% of reference SNR
                # Good quality frame - add to stack
                stacked_image += aligned_frame
                frames_stacked += 1

                # Update running average SNR
                current_snr = (current_snr * (frames_stacked - 1) + new_snr) / frames_stacked
                poor_quality_count = 0  # Reset poor quality counter

                print(f"Stacked frame: {frame_name} (Frames: {frames_stacked}, SNR: {new_snr:.2f}, Running SNR: {current_snr:.2f})")
            else:
                # Poor quality frame
                poor_quality_count += 1
                print(f"Skipping low quality frame: {frame_name} (SNR: {new_snr:.2f} < {base_snr * 0.7:.2f})")

                # Stop if too many consecutive poor quality frames
                if poor_quality_count >= max_poor_quality_frames:
                    print(f"Stopping stacking: {max_poor_quality_frames} consecutive low quality frames detected")
                    break

        # Normalize the stacked image
        print(f"Final stack complete with {frames_stacked} frames")
        self.stacked_image = stacked_image / frames_stacked
        return self.stacked_image

    def save_stacked_image(self, output_path):
        """
        Save the stacked image as a FITS file
        
        Parameters:
        -----------
        output_path : str
            Path to save the stacked image
        """
        if self.stacked_image is None:
            raise ValueError("No stacked image available. Run progressive_stack first.")
        
        # Create a new FITS HDU with the stacked image
        hdu = fits.PrimaryHDU(self.stacked_image)
        hdu.writeto(output_path, overwrite=True)
        print(f"Stacked image saved to {output_path}")

    def process(self, output_path='stacked_image.fits'):
        """
        Full processing pipeline
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the stacked image
        """
        # Filter frames based on quality
        filtered_frames = self.filter_frames()
        print(f"Frames to be stacked: {len(filtered_frames)}")
        
        # Perform progressive stacking
        self.progressive_stack(filtered_frames)
        
        # Save stacked image
        self.save_stacked_image(output_path)

# Example usage
if __name__ == "__main__":
    # Adjust these paths as needed
    csv_path = './quality_results/quality_metrics.csv'
    fits_directory = './lights/temp_290/'
    
    # Initialize and run stacking processor
    processor = AstroStackProcessor(csv_path, fits_directory)
    processor.process_with_fft(
        output_path='stacked_astronomical_image.fits',
        dark_frame='./lights/temp_290/MD-IG_200.0-E10.0s-stellina-5df04c-3072x2080-RGB-session_1.fits',
        dark_scale=1.0,
        debayer=True,
        bayer_pattern='RGGB'
    )
