import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from scipy import ndimage
from skimage.transform import rotate
from skimage.registration import phase_cross_correlation
import astropy.io.fits as fits
import os
import cv2
from datetime import datetime
import pandas as pd

def preprocess_galaxy_image(image, normalize=True, remove_bg=True, denoise=True, 
                           protect_galaxy=True, galaxy_center=None, galaxy_radius=None):
    """
    Specialized preprocessing for galaxy images
    
    Parameters:
    -----------
    image : ndarray
        Input image (can be 2D or 3D for RGB)
    normalize : bool
        Whether to normalize to 0-1 range
    remove_bg : bool
        Whether to remove background
    denoise : bool
        Whether to apply denoising
    protect_galaxy : bool
        Whether to protect galaxy region during processing
    galaxy_center : tuple or None
        (x, y) center of galaxy, if None will try to detect
    galaxy_radius : float or None
        Radius of galaxy in pixels, if None will try to estimate
        
    Returns:
    --------
    processed : ndarray
        Processed image
    """
    # Handle RGB data
    if len(image.shape) == 3:
        # Create luminance channel (simple average)
        lum = np.mean(image, axis=2)
    else:
        lum = image.copy()
    
    # Ensure float data type
    lum = lum.astype(np.float64)
    
    # Detect galaxy center and radius if not provided
    if protect_galaxy and (galaxy_center is None or galaxy_radius is None):
        # Simple galaxy detection - find brightest extended region
        # Smooth heavily to emphasize extended structures over stars
        smoothed = ndimage.gaussian_filter(lum, sigma=15.0)
        
        # Create star mask (to ignore bright stars)
        _, median, std = sigma_clipped_stats(lum, sigma=3.0)
        star_mask = lum > (median + 10*std)
        star_mask = ndimage.binary_dilation(star_mask, iterations=2)
        
        # Apply star mask
        smoothed_masked = smoothed.copy()
        smoothed_masked[star_mask] = median
        
        # Find center as maximum of smoothed image
        y_max, x_max = np.unravel_index(np.argmax(smoothed_masked), smoothed_masked.shape)
        
        # Estimate radius based on moment of inertia
        y_coords, x_coords = np.mgrid[:smoothed_masked.shape[0], :smoothed_masked.shape[1]]
        # Calculate distance from center
        r = np.sqrt((x_coords - x_max)**2 + (y_coords - y_max)**2)
        
        # Create mask for galaxy region estimation
        galaxy_mask = smoothed_masked > (median + 1*std)
        
        # Estimate radius from masked region
        if np.sum(galaxy_mask) > 0:
            galaxy_radius = np.sqrt(np.sum(galaxy_mask)) / np.pi  # Equivalent circular radius
        else:
            # Default radius if detection fails
            galaxy_radius = min(lum.shape) / 10
        
        galaxy_center = (x_max, y_max)
    
    # Create galaxy mask if center and radius are available
    if protect_galaxy and galaxy_center is not None and galaxy_radius is not None:
        y_coords, x_coords = np.mgrid[:lum.shape[0], :lum.shape[1]]
        r = np.sqrt((x_coords - galaxy_center[0])**2 + (y_coords - galaxy_center[1])**2)
        galaxy_mask = r < galaxy_radius
        # Expand mask a bit
        galaxy_mask = ndimage.binary_dilation(galaxy_mask, iterations=5)
    else:
        galaxy_mask = np.zeros_like(lum, dtype=bool)
    
    # Calculate background statistics with sigma clipping
    mean, median, std = sigma_clipped_stats(lum, sigma=3.0)
    
    # Remove background
    if remove_bg:
        # Remove global background level
        lum_bg = lum - median
        lum_bg = np.clip(lum_bg, 0, None)
        
        # Optional: remove large-scale gradients outside galaxy
        if protect_galaxy:
            # Create smoothed version for gradient estimation
            smoothed_bg = ndimage.gaussian_filter(lum, sigma=50.0)
            
            # Only apply gradient correction outside galaxy
            lum_nobg = lum.copy()
            # Apply correction outside galaxy
            lum_nobg[~galaxy_mask] = lum_bg[~galaxy_mask] - (smoothed_bg[~galaxy_mask] - median) 
            # Keep original data in galaxy region
            lum_nobg[galaxy_mask] = lum_bg[galaxy_mask]
            
            # Clip negative values
            lum_nobg = np.clip(lum_nobg, 0, None)
        else:
            lum_nobg = lum_bg
    else:
        lum_nobg = lum
    
    # Apply denoising if requested
    if denoise:
        # Apply stronger denoising outside galaxy
        if protect_galaxy:
            # Create a temporary copy
            denoised = lum_nobg.copy()
            
            # Apply strong denoising outside galaxy
            outside_galaxy = denoised[~galaxy_mask]
            outside_denoised = ndimage.gaussian_filter(outside_galaxy, sigma=1.0)
            denoised[~galaxy_mask] = outside_denoised
            
            # Apply gentle denoising inside galaxy
            inside_galaxy = denoised[galaxy_mask]
            inside_denoised = ndimage.gaussian_filter(inside_galaxy, sigma=0.5)
            denoised[galaxy_mask] = inside_denoised
        else:
            # Apply uniform denoising
            denoised = ndimage.gaussian_filter(lum_nobg, sigma=0.7)
    else:
        denoised = lum_nobg
    
    # Enhance contrast
    p_low, p_high = np.percentile(denoised, (0.5, 99.5))
    enhanced = np.clip((denoised - p_low) / (p_high - p_low), 0, 1)
    
    # Apply asinh stretch (particularly good for galaxies)
    enhanced = np.arcsinh(5 * enhanced) / np.arcsinh(5)
    
    # Normalize if requested
    if normalize:
        processed = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced) + 1e-10)
    else:
        processed = enhanced
    
    return processed

def register_galaxy_frame(frame, reference_frame, detect_rotation=True, max_rotation=1.0):
    """
    Register a frame to a reference frame with specialized galaxy processing
    
    Parameters:
    -----------
    frame : ndarray
        Frame to register
    reference_frame : ndarray
        Reference frame
    detect_rotation : bool
        Whether to detect rotation
    max_rotation : float
        Maximum rotation angle to test in degrees
        
    Returns:
    --------
    tuple
        (aligned_frame, transform_dict)
    """
    # Handle RGB data
    is_rgb = len(frame.shape) == 3
    
    # Store original frames
    original_frame = frame.copy()
    original_ref = reference_frame.copy()
    
    # Preprocess frames for registration
    ref_processed = preprocess_galaxy_image(reference_frame)
    
    # Detect rotation if requested
    rotation_angle = 0.0
    rotation_confidence = 0.0
    
    if detect_rotation:
        # Try to detect rotation angle
        angles = np.linspace(-max_rotation, max_rotation, 20)
        best_score = -np.inf
        best_angle = 0.0
        
        # Process frame once (avoid repeating for each angle)
        frame_processed = preprocess_galaxy_image(frame)
        
        for angle in angles:
            # Rotate processed frame
            rotated = rotate(frame_processed, angle, preserve_range=True)
            
            # Calculate similarity score
            score = calculate_ncc(rotated, ref_processed)
            
            if score > best_score:
                best_score = score
                best_angle = angle
        
        # Fine tune around best angle
        if abs(best_angle) < max_rotation - 0.1:
            fine_angles = np.linspace(best_angle - 0.2, best_angle + 0.2, 10)
            
            for angle in fine_angles:
                rotated = rotate(frame_processed, angle, preserve_range=True)
                score = calculate_ncc(rotated, ref_processed)
                
                if score > best_score:
                    best_score = score
                    best_angle = angle
        
        rotation_angle = best_angle
        rotation_confidence = best_score
        
        # Apply rotation to original frame
        if abs(rotation_angle) > 0.05:
            if is_rgb:
                # Handle RGB data
                rotated_frame = np.zeros_like(original_frame)
                for i in range(3):  # Process each channel
                    rotated_frame[:,:,i] = rotate(original_frame[:,:,i], rotation_angle, preserve_range=True)
            else:
                rotated_frame = rotate(original_frame, rotation_angle, preserve_range=True)
                
            # Reprocess for translation detection
            frame_processed = preprocess_galaxy_image(rotated_frame)
        else:
            rotated_frame = original_frame
            # No need to reprocess if no rotation was applied
    else:
        rotated_frame = original_frame
        # Process frame for translation detection
        frame_processed = preprocess_galaxy_image(frame)
    
    # Use phase cross-correlation for subpixel registration
    try:
        # Apply adaptive windowing to reduce edge effects
        h, w = ref_processed.shape
        y_window = np.hanning(h).reshape(-1, 1)
        x_window = np.hanning(w).reshape(1, -1)
        window = y_window * x_window
        
        ref_windowed = ref_processed * window
        frame_windowed = frame_processed * window
        
        # Compute shift with high precision
        shift, error, diffphase = phase_cross_correlation(
            ref_windowed, frame_windowed, upsample_factor=100
        )
        
        # Calculate confidence
        confidence = 1.0 / (1.0 + error)
    except Exception as e:
        print(f"Phase correlation failed: {e}")
        # Fallback to basic registration
        shift = np.array([0.0, 0.0])
        confidence = 0.0
    
    # Apply shift to rotated frame
    if is_rgb:
        # Handle RGB data
        shifted_frame = np.zeros_like(rotated_frame)
        for i in range(3):  # Process each channel
            shifted_frame[:,:,i] = ndimage.shift(
                rotated_frame[:,:,i], shift, order=3, mode='constant', cval=0
            )
    else:
        shifted_frame = ndimage.shift(
            rotated_frame, shift, order=3, mode='constant', cval=0
        )
    
    # Calculate quality metrics
    quality = calculate_alignment_quality(shifted_frame, original_ref)
    
    # Create transform dictionary
    transform = {
        'method': 'galaxy_registration',
        'shift_y': float(shift[0]),
        'shift_x': float(shift[1]),
        'rotation': float(rotation_angle),
        'translation_confidence': float(confidence),
        'rotation_confidence': float(rotation_confidence),
        'quality': quality
    }
    
    return shifted_frame, transform

def calculate_ncc(img1, img2):
    """
    Calculate normalized cross-correlation
    """
    # Ensure same shape
    if img1.shape != img2.shape:
        # Resize to smallest common shape
        min_shape = np.minimum(img1.shape, img2.shape)
        img1_cropped = img1[:min_shape[0], :min_shape[1]]
        img2_cropped = img2[:min_shape[0], :min_shape[1]]
    else:
        img1_cropped = img1
        img2_cropped = img2
    
    # Normalize images
    img1_norm = (img1_cropped - np.mean(img1_cropped)) / (np.std(img1_cropped) + 1e-10)
    img2_norm = (img2_cropped - np.mean(img2_cropped)) / (np.std(img2_cropped) + 1e-10)
    
    # Calculate correlation
    return np.mean(img1_norm * img2_norm)

def calculate_alignment_quality(aligned_frame, reference_frame):
    """
    Calculate alignment quality metrics
    """
    # Handle RGB data
    if len(aligned_frame.shape) == 3:
        # Create luminance channels
        aligned_lum = np.mean(aligned_frame, axis=2)
        ref_lum = np.mean(reference_frame, axis=2)
    else:
        aligned_lum = aligned_frame
        ref_lum = reference_frame
    
    # Process both frames the same way
    aligned_proc = preprocess_galaxy_image(aligned_lum)
    ref_proc = preprocess_galaxy_image(ref_lum)
    
    # Calculate NCC
    ncc = calculate_ncc(aligned_proc, ref_proc)
    
    # Calculate SSIM if available
    ssim = 0.0
    try:
        from skimage.metrics import structural_similarity
        ssim = structural_similarity(aligned_proc, ref_proc)
    except:
        pass
    
    # Calculate MSE
    mse = np.mean((aligned_proc - ref_proc) ** 2)
    
    return {
        'ncc': float(ncc),
        'ssim': float(ssim),
        'mse': float(mse)
    }

def process_fits_sequence(input_directory, output_directory=None, detect_rotation=True, max_rotation=1.0):
    """
    Process a sequence of FITS files containing galaxy images
    
    Parameters:
    -----------
    input_directory : str
        Directory containing FITS files
    output_directory : str or None
        Directory to save aligned frames and stacked image
    detect_rotation : bool
        Whether to detect rotation
    max_rotation : float
        Maximum rotation angle to test in degrees
        
    Returns:
    --------
    tuple
        (stacked_image, alignment_info)
    """
    # Create output directory if specified
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    
    # Find FITS files
    fits_files = [f for f in os.listdir(input_directory) 
                 if f.lower().endswith(('.fits', '.fit', '.fts'))]
    
    if not fits_files:
        print(f"No FITS files found in {input_directory}")
        return None, None
    
    print(f"Found {len(fits_files)} FITS files")
    
    # Sort files by name for consistent processing
    fits_files.sort()
    
    # Load reference frame (first file)
    reference_path = os.path.join(input_directory, fits_files[0])
    reference_frame = fits.getdata(reference_path)
    
    # Detect if RGB data
    is_rgb = len(reference_frame.shape) == 3
    
    print(f"Reference frame: {fits_files[0]}, shape: {reference_frame.shape}, RGB: {is_rgb}")
    
    # Preprocess reference frame for visualization
    ref_processed = preprocess_galaxy_image(reference_frame)
    
    # Initialize tracking variables
    alignment_info = []
    frames_aligned = 0
    
    # Add reference frame to alignment info
    alignment_info.append({
        'filename': fits_files[0],
        'method': 'reference',
        'shift_x': 0.0,
        'shift_y': 0.0,
        'rotation': 0.0,
        'translation_confidence': 1.0,
        'rotation_confidence': 1.0,
        'ncc': 1.0,
        'ssim': 1.0,
        'mse': 0.0
    })
    
    # Initialize RGB or grayscale stack
    if is_rgb:
        stacked_image = reference_frame.copy().astype(np.float64)
    else:
        stacked_image = reference_frame.copy().astype(np.float64)
    
    frames_stacked = 1
    
    # Save reference frame if output directory specified
    if output_directory:
        ref_out_path = os.path.join(output_directory, f"aligned_{fits_files[0]}")
        fits.writeto(ref_out_path, reference_frame, overwrite=True)
    
    # Process each remaining frame
    for i, filename in enumerate(fits_files[1:]):
        print(f"\nProcessing frame {i+1}/{len(fits_files)-1}: {filename}")
        
        # Load frame
        frame_path = os.path.join(input_directory, filename)
        try:
            frame = fits.getdata(frame_path)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
        
        # Check if frame has compatible shape with reference
        if frame.shape != reference_frame.shape:
            print(f"Frame {filename} has incompatible shape {frame.shape}, skipping")
            continue
        
        # Register frame
        try:
            aligned_frame, transform = register_galaxy_frame(
                frame, 
                reference_frame,
                detect_rotation=detect_rotation,
                max_rotation=max_rotation
            )
            
            # Print alignment results
            print(f"  Shift: ({transform['shift_x']:.2f}, {transform['shift_y']:.2f}) pixels")
            print(f"  Rotation: {transform['rotation']:.3f} degrees")
            print(f"  Translation confidence: {transform['translation_confidence']:.3f}")
            print(f"  Quality: NCC={transform['quality']['ncc']:.3f}, SSIM={transform['quality']['ssim']:.3f}")
            
            # Check if alignment quality is good enough
            if transform['quality']['ncc'] > 0.3:  # Adjust threshold as needed
                # Add to stack
                stacked_image += aligned_frame
                frames_stacked += 1
                frames_aligned += 1
                print(f"  Added to stack (frames: {frames_stacked})")
                
                # Save aligned frame if output directory specified
                if output_directory:
                    aligned_path = os.path.join(output_directory, f"aligned_{filename}")
                    fits.writeto(aligned_path, aligned_frame, overwrite=True)
                    
                    # Create visualization for the first few frames
                    if i < 3:  # Only for first 3 frames
                        viz_path = os.path.join(output_directory, f"viz_{filename.split('.')[0]}.png")
                        visualize_alignment(
                            reference_frame, 
                            frame, 
                            aligned_frame,
                            transform,
                            output_path=viz_path
                        )
            else:
                print(f"  Rejected: poor alignment quality (NCC={transform['quality']['ncc']:.3f})")
            
            # Add to alignment info
            alignment_info.append({
                'filename': filename,
                'method': transform['method'],
                'shift_x': transform['shift_x'],
                'shift_y': transform['shift_y'],
                'rotation': transform['rotation'],
                'translation_confidence': transform['translation_confidence'],
                'rotation_confidence': transform.get('rotation_confidence', 0.0),
                'ncc': transform['quality']['ncc'],
                'ssim': transform['quality']['ssim'],
                'mse': transform['quality']['mse']
            })
            
        except Exception as e:
            print(f"Error aligning {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Normalize stacked image
    if frames_stacked > 0:
        stacked_image /= frames_stacked
        print(f"\nStacking complete: {frames_stacked} frames stacked ({frames_aligned} aligned)")
        
        # Save stacked image if output directory specified
        if output_directory:
            stack_path = os.path.join(output_directory, "stacked_image.fits")
            
            # Create HDU
            hdu = fits.PrimaryHDU(stacked_image)
            
            # Add metadata
            hdu.header['NFRAMES'] = (frames_stacked, 'Number of frames stacked')
            hdu.header['HIERARCH ALIGNMENT METHOD'] = ('GALAXY', 'Galaxy specialized registration')
            hdu.header['HIERARCH ALIGNMENT ROTATION'] = (str(detect_rotation), 'Rotation detection enabled')
            hdu.header['HIERARCH ALIGNMENT MAXROT'] = (max_rotation, 'Maximum rotation angle')
            hdu.header['DATE'] = (datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'Processing date')
            
            # Write file
            hdu.writeto(stack_path, overwrite=True)
            print(f"Stacked image saved to {stack_path}")
            
            # Generate enhanced visualization of stacked result
            viz_stack_path = os.path.join(output_directory, "stacked_enhanced.png")
            visualize_stack(stacked_image, output_path=viz_stack_path)
    else:
        print("No frames were successfully aligned and stacked")
        stacked_image = None
    
    # Save alignment info if output directory specified
    if output_directory and alignment_info:
        info_path = os.path.join(output_directory, "alignment_info.csv")
        pd.DataFrame(alignment_info).to_csv(info_path, index=False)
        print(f"Alignment info saved to {info_path}")
    
    return stacked_image, alignment_info

def visualize_alignment(reference_frame, frame, aligned_frame, transform, output_path=None):
    """
    Create a visualization of the alignment process
    
    Parameters:
    -----------
    reference_frame : ndarray
        Reference frame
    frame : ndarray
        Original frame
    aligned_frame : ndarray
        Aligned frame
    transform : dict
        Transform dictionary with alignment parameters
    output_path : str or None
        Path to save visualization, if None will display
    """
    # Handle RGB data
    is_rgb = len(reference_frame.shape) == 3
    
    # Process frames for visualization
    if is_rgb:
        # Create luminance channels for visualization
        ref_viz = preprocess_galaxy_image(reference_frame)
        frame_viz = preprocess_galaxy_image(frame)
        aligned_viz = preprocess_galaxy_image(aligned_frame)
    else:
        ref_viz = preprocess_galaxy_image(reference_frame)
        frame_viz = preprocess_galaxy_image(frame)
        aligned_viz = preprocess_galaxy_image(aligned_frame)
    
    # Create difference images
    diff_before = ref_viz - frame_viz
    diff_after = ref_viz - aligned_viz
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot images
    axes[0, 0].imshow(ref_viz, cmap='viridis')
    axes[0, 0].set_title('Reference')
    
    axes[0, 1].imshow(frame_viz, cmap='viridis')
    axes[0, 1].set_title('Original Frame')
    
    axes[0, 2].imshow(aligned_viz, cmap='viridis')
    axes[0, 2].set_title('Aligned Frame')
    
    # Plot differences
    axes[1, 0].imshow(diff_before, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title('Difference Before Alignment')
    
    axes[1, 1].imshow(diff_after, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title('Difference After Alignment')
    
    # Plot text info
    axes[1, 2].axis('off')
    info_text = f"Rotation: {transform['rotation']:.3f}°\n" \
                f"Shift X: {transform['shift_x']:.2f} pixels\n" \
                f"Shift Y: {transform['shift_y']:.2f} pixels\n\n" \
                f"Translation confidence: {transform['translation_confidence']:.3f}\n" \
                f"NCC before: {calculate_ncc(frame_viz, ref_viz):.3f}\n" \
                f"NCC after: {transform['quality']['ncc']:.3f}\n" \
                f"SSIM: {transform['quality']['ssim']:.3f}"
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=12, transform=axes[1, 2].transAxes)
    
    # Remove ticks
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()

def visualize_stack(stacked_image, output_path=None):
    """
    Create an enhanced visualization of the stacked image
    
    Parameters:
    -----------
    stacked_image : ndarray
        Stacked image data
    output_path : str or None
        Path to save visualization, if None will display
    """
    # Handle RGB or grayscale
    is_rgb = len(stacked_image.shape) == 3
    
    # Create figure
    plt.figure(figsize=(12, 12))
    
    if is_rgb:
        # For RGB, apply different stretches to show different details
        # Apply standard processing
        standard = preprocess_galaxy_image(stacked_image)
        
        # Create an asinh stretch with stronger parameters for faint details
        def enhance_faint_details(img):
            # Convert to luminance if RGB
            if len(img.shape) == 3:
                lum = np.mean(img, axis=2)
            else:
                lum = img.copy()
                
            # Background removal with sigma clipping
            mean, median, std = sigma_clipped_stats(lum, sigma=3.0)
            lum_bg = lum - median
            lum_bg = np.clip(lum_bg, 0, None)
            
            # Apply stronger stretch for faint details
            enhanced = np.arcsinh(10 * lum_bg / std) / np.arcsinh(10)
            
            # Normalize
            enhanced = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced) + 1e-10)
            
            return enhanced
        
        enhanced = enhance_faint_details(stacked_image)
        
        # Create a subplot for each visualization
        plt.subplot(2, 2, 1)
        plt.imshow(stacked_image)
        plt.title('RGB Stack')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(standard, cmap='viridis')
        plt.title('Standard Processing')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(enhanced, cmap='hot')
        plt.title('Enhanced for Faint Details')
        plt.axis('off')
        
        # Show RGB channels separately
        plt.subplot(2, 2, 4)
        r_enhanced = enhance_faint_details(stacked_image[:,:,0])
        g_enhanced = enhance_faint_details(stacked_image[:,:,1])
        b_enhanced = enhance_faint_details(stacked_image[:,:,2])
        
        # Create false color image
        false_color = np.stack([r_enhanced, g_enhanced, b_enhanced], axis=2)
        plt.imshow(false_color)
        plt.title('False Color')
        plt.axis('off')
    else:
        # For grayscale, apply different stretches to show different details
        # Standard processing
        standard = preprocess_galaxy_image(stacked_image)
        
        # Stronger stretch for faint details
        def enhance_faint_details(img):
            # Background removal with sigma clipping
            mean, median, std = sigma_clipped_stats(img, sigma=3.0)
            img_bg = img - median
            img_bg = np.clip(img_bg, 0, None)
            
            # Apply stronger stretch for faint details
            enhanced = np.arcsinh(10 * img_bg / std) / np.arcsinh(10)
            
            # Normalize
            enhanced = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced) + 1e-10)
            
            return enhanced
        
        enhanced = enhance_faint_details(stacked_image)
        
        # Linear stretch for bright details
        p_low, p_high = np.percentile(stacked_image, (50, 99.5))
        linear = np.clip((stacked_image - p_low) / (p_high - p_low), 0, 1)
        
        # Log stretch for medium details
        mean, median, std = sigma_clipped_stats(stacked_image, sigma=3.0)
        bg_sub = np.clip(stacked_image - median, 0, None)
        log_stretch = np.log1p(bg_sub / std)
        log_stretch = (log_stretch - np.min(log_stretch)) / (np.max(log_stretch) - np.min(log_stretch) + 1e-10)
        
        # Create a subplot for each visualization
        plt.subplot(2, 2, 1)
        plt.imshow(standard, cmap='viridis')
        plt.title('Standard Processing')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(enhanced, cmap='inferno')
        plt.title('Enhanced for Faint Details')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(linear, cmap='gray')
        plt.title('Linear Stretch (Bright Details)')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(log_stretch, cmap='plasma')
        plt.title('Log Stretch (Medium Details)')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()

def analyze_rotation_statistics(alignment_info):
    """
    Analyze rotation angles from alignment results
    
    Parameters:
    -----------
    alignment_info : list of dict
        Alignment information
        
    Returns:
    --------
    dict
        Statistics of rotation angles
    """
    # Extract rotation angles (skip reference frame)
    rotations = [info['rotation'] for info in alignment_info[1:] if 'rotation' in info]
    
    if len(rotations) < 2:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'range': 0.0
        }
    
    # Calculate statistics
    mean_rot = np.mean(rotations)
    median_rot = np.median(rotations)
    std_rot = np.std(rotations)
    min_rot = np.min(rotations)
    max_rot = np.max(rotations)
    range_rot = max_rot - min_rot
    
    return {
        'mean': float(mean_rot),
        'median': float(median_rot),
        'std': float(std_rot),
        'min': float(min_rot),
        'max': float(max_rot),
        'range': float(range_rot)
    }

def main():
    """
    Main function for command line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Galaxy Image Registration and Stacking")
    parser.add_argument("input_dir", help="Directory containing FITS files")
    parser.add_argument("--output_dir", help="Directory to save aligned frames and stack")
    parser.add_argument("--rotation", action="store_true", help="Enable rotation detection")
    parser.add_argument("--max_rotation", type=float, default=1.0, help="Maximum rotation angle (degrees)")
    
    args = parser.parse_args()
    
    # Process sequence
    stacked_image, alignment_info = process_fits_sequence(
        args.input_dir,
        args.output_dir,
        detect_rotation=args.rotation,
        max_rotation=args.max_rotation
    )
    
    # Analyze rotation statistics if alignment was successful
    if alignment_info:
        rotation_stats = analyze_rotation_statistics(alignment_info)
        print("\nRotation Statistics:")
        print(f"  Mean rotation: {rotation_stats['mean']:.3f}°")
        print(f"  Median rotation: {rotation_stats['median']:.3f}°")
        print(f"  Standard deviation: {rotation_stats['std']:.3f}°")
        print(f"  Range: {rotation_stats['min']:.3f}° to {rotation_stats['max']:.3f}°")
        print(f"  Span: {rotation_stats['range']:.3f}°")

if __name__ == "__main__":
    main()