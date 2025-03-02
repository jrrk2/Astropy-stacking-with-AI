#!/usr/bin/env python3
"""
Complete Stellina image processing script with:
- Advanced dark frame management
- Bayer pattern handling with rotation
- Registration
- Stacking
"""

import os
import numpy as np
import glob
import argparse
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from skimage.registration import phase_cross_correlation
from scipy import ndimage
from skimage.transform import rotate
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from datetime import datetime
from pathlib import Path
import re

class DarkFrameHandler:
    def __init__(self, dark_base_dir=None):
        """
        Initialize dark frame handler
        
        Parameters:
        -----------
        dark_base_dir : str or None
            Base directory containing temp_XXX subdirectories with dark frames
        """
        self.dark_base_dir = Path(dark_base_dir) if dark_base_dir else None
        self.dark_frames = {}  # {temp_k: {'data': array, 'pattern': str}}
        
        if self.dark_base_dir and self.dark_base_dir.exists():
            self._load_dark_frames()
    
    def _load_dark_frames(self):
        """Load master dark frames from temperature directories"""
        for temp_dir in self.dark_base_dir.glob("temp_*"):
            try:
                temp_k = int(temp_dir.name.split('_')[1])
                master_dark_path = temp_dir / "master_dark.fits"
                
                # Check if master dark exists
                if master_dark_path.exists():
                    with fits.open(master_dark_path) as hdul:
                        self.dark_frames[temp_k] = {
                            'data': hdul[0].data.astype(np.float32),
                            'pattern': hdul[0].header.get('BAYERPAT', 'RGGB').strip()
                        }
                    print(f"Loaded existing master dark for {temp_k}K")
                else:
                    # Find any dark frame in the directory
                    dark_files = list(temp_dir.glob("*.fits"))
                    if dark_files:
                        # Use the first one for testing
                        with fits.open(dark_files[0]) as hdul:
                            self.dark_frames[temp_k] = {
                                'data': hdul[0].data.astype(np.float32),
                                'pattern': hdul[0].header.get('BAYERPAT', 'RGGB').strip()
                            }
                        print(f"Loaded sample dark for {temp_k}K from {dark_files[0].name}")
            except Exception as e:
                print(f"Error loading dark from {temp_dir}: {e}")
        
        if self.dark_frames:
            self.temp_range = (min(self.dark_frames.keys()), max(self.dark_frames.keys()))
            print(f"Loaded {len(self.dark_frames)} dark frames, temperature range: {self.temp_range[0]}-{self.temp_range[1]}K")
        else:
            print("No dark frames found")
    
    def celsius_to_kelvin(self, temp_c):
        """Convert Celsius to Kelvin"""
        return temp_c + 273.15
    
    def get_dark_frame(self, temp_c, target_pattern):
        """
        Get appropriate dark frame for given temperature and target pattern
        
        Parameters:
        -----------
        temp_c : float
            Temperature in Celsius
        target_pattern : str
            Target Bayer pattern (e.g., 'BGGR')
            
        Returns:
        --------
        tuple
            (dark_data, info_message) or (None, error_message)
        """
        if not self.dark_frames:
            return None, "No dark frames available"
        
        temp_k = self.celsius_to_kelvin(temp_c)
        
        # Find closest available temperature
        available_temps = np.array(list(self.dark_frames.keys()))
        closest_temp = available_temps[np.abs(available_temps - temp_k).argmin()]
        dark = self.dark_frames[closest_temp]
        info_msg = f"Using dark frame from {closest_temp}K for {temp_k:.1f}K"
        
        # Handle pattern rotation if needed
        if dark['pattern'] != target_pattern:
            if (dark['pattern'] == 'RGGB' and target_pattern == 'BGGR') or \
               (dark['pattern'] == 'BGGR' and target_pattern == 'RGGB'):
                # 180-degree rotation for RGGB<->BGGR conversion
                dark_data = np.rot90(dark['data'], 2)
                info_msg += f" (rotated {dark['pattern']} to {target_pattern})"
            else:
                # For other patterns, we'd need more complex conversion
                return None, f"Unsupported pattern conversion: {dark['pattern']} to {target_pattern}"
        else:
            dark_data = dark['data']
        
        return dark_data, info_msg

def debayer_image(raw_data, bayer_pattern='BGGR'):
    """
    Debayer a raw Bayer pattern image to RGB
    
    Parameters:
    -----------
    raw_data : ndarray
        Raw image data
    bayer_pattern : str
        Bayer pattern (e.g., 'BGGR', 'RGGB', 'GRBG', 'GBRG')
        
    Returns:
    --------
    rgb_data : ndarray
        Debayered RGB image
    """
    # Define OpenCV Bayer pattern codes
    bayer_codes = {
        'BGGR': cv2.COLOR_BAYER_BG2RGB,
        'RGGB': cv2.COLOR_BAYER_RG2RGB,
        'GRBG': cv2.COLOR_BAYER_GB2RGB,
        'GBRG': cv2.COLOR_BAYER_GR2RGB
    }
    
    # Get appropriate code
    if bayer_pattern in bayer_codes:
        code = bayer_codes[bayer_pattern]
    else:
        print(f"Warning: Unknown Bayer pattern '{bayer_pattern}'. Using BGGR.")
        code = cv2.COLOR_BAYER_BG2RGB
    
    # Convert to 8-bit for OpenCV if needed
    if raw_data.dtype == np.uint16:
        # Scale down to 8-bit (preserve dynamic range)
        scaled_data = (raw_data / 256).astype(np.uint8)
    else:
        scaled_data = raw_data.astype(np.uint8)
    
    # Debayer the image
    try:
        rgb_data = cv2.cvtColor(scaled_data, code)
        
        # Convert to float for further processing
        rgb_float = rgb_data.astype(np.float32)
        
        return rgb_float
    except Exception as e:
        print(f"Error in debayering: {e}")
        # Fallback to simple conversion
        print("Using fallback debayering method")
        h, w = raw_data.shape
        rgb_fallback = np.zeros((h, w, 3), dtype=np.float32)
        # Just duplicate the raw data to all channels
        rgb_fallback[:,:,0] = raw_data / 256.0 if raw_data.dtype == np.uint16 else raw_data
        rgb_fallback[:,:,1] = rgb_fallback[:,:,0]
        rgb_fallback[:,:,2] = rgb_fallback[:,:,0]
        return rgb_fallback

def apply_dark_subtraction(image, dark_frame, scale=1.0):
    """
    Apply dark frame subtraction
    
    Parameters:
    -----------
    image : ndarray
        Image to process
    dark_frame : ndarray
        Dark frame
    scale : float
        Scaling factor for dark frame (for exposure time differences)
        
    Returns:
    --------
    calibrated : ndarray
        Dark-subtracted image
    """
    # Check if both are RGB or both are grayscale
    if len(image.shape) != len(dark_frame.shape):
        raise ValueError(f"Image shape {image.shape} is incompatible with dark frame shape {dark_frame.shape}")
    
    # Apply scaling if needed
    if scale != 1.0:
        scaled_dark = dark_frame * scale
    else:
        scaled_dark = dark_frame
    
    # Subtract dark frame
    calibrated = image - scaled_dark
    
    # Clip negative values
    calibrated = np.clip(calibrated, 0, None)
    
    return calibrated

def extract_luminance(rgb_image):
    """
    Extract luminance channel from RGB image
    """
    # Simple average of RGB channels
    luminance = np.mean(rgb_image, axis=2)
    return luminance

def enhance_image_for_registration(image, is_rgb=False):
    """
    Enhance image for registration
    """
    # Get luminance if RGB
    if is_rgb:
        lum = extract_luminance(image)
    else:
        lum = image.copy()
    
    # Convert to float
    lum = lum.astype(np.float64)
    
    # Background statistics
    mean, median, std = sigma_clipped_stats(lum, sigma=3.0)
    
    # Background subtraction
    lum_bg = lum - median
    lum_bg = np.clip(lum_bg, 0, None)
    
    # Stretch
    p_high = np.percentile(lum_bg, 99.5)
    lum_bg = lum_bg / (p_high + 1e-10)
    
    # Apply sqrt stretch (good for astronomy)
    lum_stretch = np.sqrt(lum_bg)
    
    # Mild denoising
    lum_smooth = ndimage.gaussian_filter(lum_stretch, sigma=0.7)
    
    # Normalize
    lum_norm = (lum_smooth - np.min(lum_smooth)) / (np.max(lum_smooth) - np.min(lum_smooth) + 1e-10)
    
    return lum_norm

def register_frame(frame, reference, detect_rotation=True, max_rotation=1.0):
    """
    Register frame to reference
    
    Parameters:
    -----------
    frame : ndarray
        Frame to align
    reference : ndarray
        Reference frame
    detect_rotation : bool
        Whether to detect rotation
    max_rotation : float
        Maximum rotation to test in degrees
        
    Returns:
    --------
    tuple
        (aligned_frame, transform_dict)
    """
    # Determine if RGB
    is_rgb = len(frame.shape) == 3
    
    # Store original frames
    original_frame = frame.copy()
    original_ref = reference.copy()
    
    # Enhance for registration
    frame_enh = enhance_image_for_registration(frame, is_rgb)
    ref_enh = enhance_image_for_registration(reference, is_rgb)
    
    # Detect rotation if requested
    rotation_angle = 0.0
    rotation_confidence = 0.0
    
    if detect_rotation:
        # Grid search for rotation
        angles = np.linspace(-max_rotation, max_rotation, 20)
        best_score = -np.inf
        best_angle = 0.0
        
        for angle in angles:
            # Rotate enhanced frame
            rotated = rotate(frame_enh, angle, preserve_range=True)
            
            # Calculate similarity
            ncc = calculate_ncc(rotated, ref_enh)
            
            if ncc > best_score:
                best_score = ncc
                best_angle = angle
        
        # Fine tune best angle
        if abs(best_angle) < max_rotation - 0.1:
            fine_angles = np.linspace(best_angle - 0.2, best_angle + 0.2, 10)
            
            for angle in fine_angles:
                rotated = rotate(frame_enh, angle, preserve_range=True)
                ncc = calculate_ncc(rotated, ref_enh)
                
                if ncc > best_score:
                    best_score = ncc
                    best_angle = angle
        
        rotation_angle = best_angle
        rotation_confidence = best_score
        
        # Apply rotation if significant
        if abs(rotation_angle) > 0.05:
            if is_rgb:
                # Handle RGB data
                rotated_frame = np.zeros_like(original_frame)
                for i in range(3):
                    rotated_frame[:,:,i] = rotate(original_frame[:,:,i], rotation_angle, preserve_range=True)
                
                # Update enhanced frame for translation
                frame_enh = enhance_image_for_registration(rotated_frame, is_rgb)
            else:
                rotated_frame = rotate(original_frame, rotation_angle, preserve_range=True)
                frame_enh = enhance_image_for_registration(rotated_frame, is_rgb)
        else:
            rotated_frame = original_frame
    else:
        rotated_frame = original_frame
    
    # Use phase correlation for subpixel registration
    try:
        # Apply window to reduce edge effects
        h, w = frame_enh.shape
        y_window = np.hanning(h).reshape(-1, 1)
        x_window = np.hanning(w).reshape(1, -1)
        window = y_window * x_window
        
        frame_windowed = frame_enh * window
        ref_windowed = ref_enh * window
        
        shift, error, diffphase = phase_cross_correlation(
            ref_windowed, frame_windowed, upsample_factor=100
        )
        
        confidence = 1.0 / (1.0 + error)
    except Exception as e:
        print(f"Phase correlation failed: {e}")
        shift = np.array([0.0, 0.0])
        confidence = 0.0
    
    # Apply shift
    if is_rgb:
        # Handle RGB
        aligned_frame = np.zeros_like(rotated_frame)
        for i in range(3):
            aligned_frame[:,:,i] = ndimage.shift(rotated_frame[:,:,i], shift, order=3, mode='constant', cval=0)
    else:
        aligned_frame = ndimage.shift(rotated_frame, shift, order=3, mode='constant', cval=0)
    
    # Calculate quality
    quality = {
        'ncc': float(calculate_ncc(enhance_image_for_registration(aligned_frame, is_rgb), 
                                  enhance_image_for_registration(original_ref, is_rgb)))
    }
    
    # Create transform dictionary
    transform = {
        'shift_y': float(shift[0]),
        'shift_x': float(shift[1]),
        'rotation': float(rotation_angle),
        'confidence': float(confidence),
        'quality': quality
    }
    
    return aligned_frame, transform

def calculate_ncc(img1, img2):
    """
    Calculate normalized cross-correlation
    """
    # Normalize images
    img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-10)
    img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-10)
    
    # Calculate correlation
    return np.mean(img1_norm * img2_norm)

def extract_temperature(fits_header):
    """
    Extract temperature from FITS header
    
    Returns temperature in Celsius
    """
    # Try common temperature keywords
    for keyword in ['TEMP', 'CCD-TEMP', 'CCDTEMP', 'SENSORTE']:
        if keyword in fits_header:
            try:
                return float(fits_header[keyword])
            except (ValueError, TypeError):
                pass
    
    # Try to extract from filename
    if 'FILENAME' in fits_header:
        filename = fits_header['FILENAME']
        # Look for patterns like "temp_290" in filename
        temp_match = re.search(r'temp_(\d+)', filename)
        if temp_match:
            try:
                # Usually in Kelvin, convert to Celsius
                return float(temp_match.group(1)) - 273.15
            except (ValueError, TypeError):
                pass
    
    # Default temperature if not found
    print("Warning: Temperature not found in FITS header, using default")
    return 20.0  # Default to 20°C

def process_stellina_images(input_dir, output_dir, dark_dir=None, 
                          detect_rotation=True, max_rotation=1.0, max_frames=None):
    """
    Process Stellina images with debayering, dark subtraction, and registration
    
    Parameters:
    -----------
    input_dir : str
        Directory with raw Stellina frames
    output_dir : str
        Directory for output
    dark_dir : str or None
        Directory with dark frames, if None will not perform dark subtraction
    detect_rotation : bool
        Whether to detect rotation
    max_rotation : float
        Maximum rotation to test (degrees)
    max_frames : int or None
        Maximum number of frames to process
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dark frame handler if directory provided
    dark_handler = None
    if dark_dir:
        dark_handler = DarkFrameHandler(dark_dir)
    
    # Find FITS files
    fits_files = sorted(glob.glob(os.path.join(input_dir, "*.fits")))
    
    if not fits_files:
        print(f"No FITS files found in {input_dir}")
        return
    
    print(f"Found {len(fits_files)} FITS files")
    
    # Limit number of frames if specified
    if max_frames and max_frames < len(fits_files):
        print(f"Limiting to {max_frames} frames")
        fits_files = fits_files[:max_frames]
    
    # Load reference frame
    try:
        reference_path = fits_files[0]
        print(f"Loading reference frame: {os.path.basename(reference_path)}")
        ref_hdul = fits.open(reference_path)
        reference_raw = ref_hdul[0].data
        ref_header = ref_hdul[0].header
        
        # Get Bayer pattern from header
        bayer_pattern = ref_header.get('BAYERPAT', 'BGGR').strip()
        print(f"Reference Bayer pattern: {bayer_pattern}")
        
        # Get temperature if available
        ref_temp = extract_temperature(ref_header)
        print(f"Reference temperature: {ref_temp:.1f}°C")
        
        ref_hdul.close()
        
        # Process dark frame if available
        dark_data = None
        dark_info = "No dark frame used"
        
        if dark_handler:
            dark_data, dark_info = dark_handler.get_dark_frame(ref_temp, bayer_pattern)
            if dark_data is not None:
                print(f"Using dark frame: {dark_info}")
                
                # Apply dark subtraction to raw data
                reference_dark_sub = apply_dark_subtraction(reference_raw, dark_data)
                print(f"Applied dark subtraction to reference frame")
            else:
                reference_dark_sub = reference_raw
                print(f"Dark frame not available: {dark_info}")
        else:
            reference_dark_sub = reference_raw
        
        # Apply debayering
        print("Debayering reference frame...")
        reference = debayer_image(reference_dark_sub, bayer_pattern)
        
        # Create diagnostic image
        plt.figure(figsize=(12, 10))
        plt.imshow(reference / 255.0)  # Normalize for display
        plt.title(f"Reference Frame: {os.path.basename(reference_path)} (Debayered)")
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, "reference_rgb.png"))
        plt.close()
        
        # Save luminance image
        luminance = extract_luminance(reference)
        plt.figure(figsize=(10, 8))
        plt.imshow(enhance_image_for_registration(luminance), cmap='viridis')
        plt.title("Reference Frame (Luminance)")
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, "reference_lum.png"))
        plt.close()
        
        # Save processed reference frame
        fits.writeto(os.path.join(output_dir, "reference_processed.fits"), reference, overwrite=True)
        
    except Exception as e:
        print(f"Error processing reference frame: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize stacking
    stacked = reference.astype(np.float64)
    frame_count = 1
    
    # Create alignment info list
    alignment_info = [{
        'filename': os.path.basename(reference_path),
        'shift_x': 0.0,
        'shift_y': 0.0,
        'rotation': 0.0,
        'confidence': 1.0,
        'ncc': 1.0,
        'temperature': ref_temp,
        'dark_info': dark_info,
        'is_reference': True
    }]
    
    # Process each frame
    for i, file_path in enumerate(fits_files[1:]):
        try:
            print(f"\nProcessing frame {i+1}/{len(fits_files)-1}: {os.path.basename(file_path)}")
            
            # Load frame
            frame_hdul = fits.open(file_path)
            frame_raw = frame_hdul[0].data
            frame_header = frame_hdul[0].header
            
            # Get frame's Bayer pattern
            frame_pattern = frame_header.get('BAYERPAT', bayer_pattern).strip()
            
            # Get frame's temperature
            frame_temp = extract_temperature(frame_header)
            print(f"Frame temperature: {frame_temp:.1f}°C")
            
            # Check for pattern rotation (happens at 180° from derotator)
            pattern_rotated = False
            if frame_pattern != bayer_pattern:
                print(f"Bayer pattern changed from {bayer_pattern} to {frame_pattern}")
                pattern_rotated = True
            
            frame_hdul.close()
            
            # Process dark frame if available
            frame_dark_info = "No dark frame used"
            if dark_handler:
                # Get appropriate dark frame for this frame's temperature
                dark_data, frame_dark_info = dark_handler.get_dark_frame(frame_temp, frame_pattern)
                if dark_data is not None:
                    print(f"Using dark frame: {frame_dark_info}")
                    
                    # Apply dark subtraction 
                    frame_dark_sub = apply_dark_subtraction(frame_raw, dark_data)
                else:
                    frame_dark_sub = frame_raw
                    print(f"Dark frame not available: {frame_dark_info}")
            else:
                frame_dark_sub = frame_raw
            
            # Apply debayering
            frame = debayer_image(frame_dark_sub, frame_pattern)
            
            # Check shape
            if frame.shape != reference.shape:
                print(f"Skipping frame with incorrect shape: {frame.shape}")
                continue
            
            # Align frame
            aligned, transform = register_frame(
                frame, reference, 
                detect_rotation=detect_rotation,
                max_rotation=max_rotation
            )
            
            # Print alignment info
            print(f"  Shift: ({transform['shift_x']:.2f}, {transform['shift_y']:.2f}) pixels")
            print(f"  Rotation: {transform['rotation']:.3f} degrees")
            print(f"  Confidence: {transform['confidence']:.3f}")
            print(f"  NCC: {transform['quality']['ncc']:.3f}")
            
            # Add to stack if alignment is good
            if transform['quality']['ncc'] > 0.3:  # Adjust threshold as needed
                stacked += aligned
                frame_count += 1
                print(f"  Added to stack (frames: {frame_count})")
                
                # Save aligned frame
                fits.writeto(os.path.join(output_dir, f"aligned_{i+1}.fits"), aligned, overwrite=True)
                
                # Create comparison for first few frames
                if i < 3:
                    # Create enhanced versions for visualization
                    ref_viz = reference / 255.0  # Normalize for display
                    frame_viz = frame / 255.0
                    aligned_viz = aligned / 255.0
                    
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    axes[0].imshow(ref_viz)
                    axes[0].set_title("Reference")
                    axes[0].axis('off')
                    
                    axes[1].imshow(frame_viz)
                    axes[1].set_title("Original")
                    axes[1].axis('off')
                    
                    axes[2].imshow(aligned_viz)
                    axes[2].set_title("Aligned")
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"compare_{i+1}.png"))
                    plt.close()
            else:
                print(f"  Rejected: low alignment quality")
            
            # Add to alignment info
            alignment_info.append({
                'filename': os.path.basename(file_path),
                'shift_x': transform['shift_x'],
                'shift_y': transform['shift_y'],
                'rotation': transform['rotation'],
                'confidence': transform['confidence'],
                'ncc': transform['quality']['ncc'],
                'temperature': frame_temp,
                'dark_info': frame_dark_info,
                'pattern': frame_pattern,
                'pattern_rotated': pattern_rotated
            })
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save final stack
    if frame_count > 0:
        final_stack = stacked / frame_count
        
        # Create HDU with metadata
        hdu = fits.PrimaryHDU(final_stack)
        hdu.header['NFRAMES'] = frame_count
        hdu.header['DEBAYER'] = (True, 'Applied debayering')
        hdu.header['BAYERPAT'] = (bayer_pattern, 'Reference Bayer pattern')
        hdu.header['DARKSUB'] = (dark_handler is not None, 'Applied dark subtraction')
        
        # Save stack
        stack_path = os.path.join(output_dir, "stacked.fits")
        hdu.writeto(stack_path, overwrite=True)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        plt.imshow(final_stack / 255.0)  # Normalize for display
        plt.title(f"Stacked Image ({frame_count} frames)")
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, "stacked_rgb.png"))
        plt.close()
        
        # Create enhanced visualization of luminance for detail
        lum_stack = extract_luminance(final_stack)
        enhanced_lum = enhance_image_for_registration(lum_stack, is_rgb=False)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(enhanced_lum, cmap='viridis')
        plt.title(f"Stacked Image - Enhanced ({frame_count} frames)")
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, "stacked_enhanced.png"))
        plt.close()
        
        print(f"Stacked {frame_count} frames. Output saved to {stack_path}")
        
        # Save alignment info
        pd.DataFrame(alignment_info).to_csv(os.path.join(output_dir, "alignment_info.csv"), index=False)
        
        # Calculate and print statistics
        shifts_x = [info['shift_x'] for info in alignment_info[1:]]
        shifts_y = [info['shift_y'] for info in alignment_info[1:]]
        rotations = [info['rotation'] for info in alignment_info[1:]]
        
        print("\nAlignment Statistics:")
        print(f"  Mean shift X: {np.mean(shifts_x):.3f} ± {np.std(shifts_x):.3f} pixels")
        print(f"  Mean shift Y: {np.mean(shifts_y):.3f} ± {np.std(shifts_y):.3f} pixels")
        print(f"  Mean rotation: {np.mean(rotations):.4f} ± {np.std(rotations):.4f} degrees")
        print(f"  Range: {np.min(rotations):.4f}° to {np.max(rotations):.4f}°")
    else:
        print("No frames were successfully aligned")

def main():
    parser = argparse.ArgumentParser(description="Stellina image processing with dark frame management")
    parser.add_argument("input_dir", help="Directory containing FITS files")
    parser.add_argument("output_dir", help="Directory for output files")
    parser.add_argument("--dark_dir", help="Directory with dark frames organized in temp_XXX folders")
    parser.add_argument("--no_rotation", action="store_true", help="Disable rotation detection")
    parser.add_argument("--max_rotation", type=float, default=1.0, help="Maximum rotation angle to test")
    parser.add_argument("--max_frames", type=int, help="Maximum frames to process")
    
    args = parser.parse_args()
    
    process_stellina_images(
        args.input_dir, 
        args.output_dir, 
        dark_dir=args.dark_dir,
        detect_rotation=not args.no_rotation,
        max_rotation=args.max_rotation,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()
