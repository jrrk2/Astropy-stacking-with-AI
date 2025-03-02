import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import pandas as pd
from datetime import datetime

def drizzle_integration(aligned_frames, shifts, rotations=None, output_shape=None, 
                       pixfrac=0.7, scale=1.5, kernel='square'):
    """
    Implement Drizzle algorithm for combining aligned frames with subpixel accuracy
    
    Parameters:
    -----------
    aligned_frames : list of ndarray
        List of aligned frames
    shifts : list of tuple
        List of (y, x) shifts used for alignment
    rotations : list of float or None
        List of rotation angles (degrees) used for alignment
    output_shape : tuple or None
        Shape of output image (height, width), if None will be calculated
    pixfrac : float
        Pixel fraction - size of drop relative to input pixel (0.5-1.0)
    scale : float
        Scale factor for output pixel grid (1.0-3.0)
    kernel : str
        Drizzle kernel shape ('square', 'gaussian', 'lanczos')
        
    Returns:
    --------
    drizzled : ndarray
        Drizzled image
    """
    # Make sure we have arrays
    frames = [np.array(frame) for frame in aligned_frames]
    
    # Get information from frames
    n_frames = len(frames)
    is_rgb = len(frames[0].shape) == 3
    
    if is_rgb:
        h, w, c = frames[0].shape
    else:
        h, w = frames[0].shape
    
    # Determine output shape
    if output_shape is None:
        out_h = int(h * scale)
        out_w = int(w * scale)
        if is_rgb:
            output_shape = (out_h, out_w, c)
        else:
            output_shape = (out_h, out_w)
    
    # Initialize output arrays
    if is_rgb:
        drizzled = np.zeros(output_shape, dtype=np.float64)
        weight = np.zeros(output_shape[:2], dtype=np.float64)
    else:
        drizzled = np.zeros(output_shape, dtype=np.float64)
        weight = np.zeros(output_shape, dtype=np.float64)
    
    # Pixel scale
    in_scale = 1.0
    out_scale = in_scale / scale
    
    # Calculate drop size
    drop_size = in_scale * pixfrac
    
    # Prepare output pixel grid
    out_h, out_w = output_shape[:2]
    out_y_grid, out_x_grid = np.mgrid[0:out_h, 0:out_w]
    out_y_grid = out_y_grid * out_scale
    out_x_grid = out_x_grid * out_scale
    
    # Process each frame
    print(f"Drizzling {n_frames} frames with pixfrac={pixfrac}, scale={scale}")
    
    for i, frame in enumerate(frames):
        # Get shift for this frame
        y_shift, x_shift = shifts[i]
        
        # Get rotation for this frame
        angle = 0.0
        if rotations is not None and i < len(rotations):
            angle = rotations[i]
        
        # Drizzle this frame to output grid
        drizzle_frame(
            frame, drizzled, weight, 
            y_shift, x_shift, angle,
            in_scale, out_scale, drop_size,
            kernel
        )
        
        # Progress update
        if (i+1) % 10 == 0 or i+1 == n_frames:
            print(f"  Processed {i+1}/{n_frames} frames")
    
    # Normalize by weight
    # Avoid division by zero
    if is_rgb:
        for c_idx in range(drizzled.shape[2]):
            nonzero = weight > 0
            drizzled[..., c_idx][nonzero] /= weight[nonzero]
    else:
        nonzero = weight > 0
        drizzled[nonzero] /= weight[nonzero]
    
    print("Drizzling complete")
    
    return drizzled

def drizzle_frame(frame, output, weight, y_shift, x_shift, angle, 
                 in_scale, out_scale, drop_size, kernel):
    """
    Drizzle a single frame onto the output grid
    
    Parameters:
    -----------
    frame : ndarray
        Input frame
    output : ndarray
        Output drizzled image (modified in-place)
    weight : ndarray
        Output weight image (modified in-place)
    y_shift, x_shift : float
        Shift of this frame relative to reference
    angle : float
        Rotation angle (degrees)
    in_scale : float
        Scale of input pixels
    out_scale : float
        Scale of output pixels
    drop_size : float
        Size of drizzle drop
    kernel : str
        Drizzle kernel shape
    """
    # Get frame shape
    if len(frame.shape) == 3:
        h, w, c = frame.shape
        is_rgb = True
    else:
        h, w = frame.shape
        is_rgb = False
    
    # Create input pixel grid
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Apply transformation to input pixel coordinates
    if angle != 0:
        # Rotation center (center of image)
        y_center = h / 2
        x_center = w / 2
        
        # Shift to origin, rotate, shift back
        y_rot = (y_grid - y_center) * np.cos(angle_rad) - (x_grid - x_center) * np.sin(angle_rad) + y_center
        x_rot = (y_grid - y_center) * np.sin(angle_rad) + (x_grid - x_center) * np.cos(angle_rad) + x_center
        
        # Apply alignment shift
        y_trans = y_rot - y_shift
        x_trans = x_rot - x_shift
    else:
        # Just apply alignment shift
        y_trans = y_grid - y_shift
        x_trans = x_grid - x_shift
    
    # Scale to output pixel units
    y_out = y_trans / out_scale
    x_out = x_trans / out_scale
    
    # Calculate drop overlap with output pixels
    # This is the computationally intensive part
    
    # For each input pixel, calculate output pixels that overlap with the drop
    drop_radius = drop_size / out_scale / 2
    
    # Process by chunks to avoid excessive memory use
    chunk_size = 1000  # Adjust based on available memory
    n_pixels = h * w
    n_chunks = int(np.ceil(n_pixels / chunk_size))
    
    for chunk in range(n_chunks):
        # Get pixel indices for this chunk
        start = chunk * chunk_size
        end = min((chunk + 1) * chunk_size, n_pixels)
        indices = np.unravel_index(np.arange(start, end), (h, w))
        
        # Get transformed coordinates for these pixels
        chunk_y = y_out[indices]
        chunk_x = x_out[indices]
        
        # Determine output pixels that need to be updated
        min_y = np.floor(chunk_y - drop_radius).astype(int)
        max_y = np.ceil(chunk_y + drop_radius).astype(int)
        min_x = np.floor(chunk_x - drop_radius).astype(int)
        max_x = np.ceil(chunk_x + drop_radius).astype(int)
        
        # Ensure bounds are within output image
        min_y = np.maximum(min_y, 0)
        max_y = np.minimum(max_y, output.shape[0] - 1)
        min_x = np.maximum(min_x, 0)
        max_x = np.minimum(max_x, output.shape[1] - 1)
        
        # For each pixel in chunk, update output pixels that overlap with its drop
        for i in range(end - start):
            y_in, x_in = indices[0][i], indices[1][i]
            y_center, x_center = chunk_y[i], chunk_x[i]
            
            # Skip if out of bounds
            if (min_y[i] > max_y[i]) or (min_x[i] > max_x[i]):
                continue
            
            # Get pixel value
            if is_rgb:
                pixel_value = frame[y_in, x_in, :]
            else:
                pixel_value = frame[y_in, x_in]
            
            # Calculate overlap for each output pixel in range
            for y_out in range(min_y[i], max_y[i] + 1):
                for x_out in range(min_x[i], max_x[i] + 1):
                    # Calculate distance from drop center
                    dy = (y_out + 0.5) - y_center
                    dx = (x_out + 0.5) - x_center
                    
                    # Determine overlap based on kernel
                    if kernel == 'square':
                        # Square drop
                        if (abs(dy) <= drop_radius) and (abs(dx) <= drop_radius):
                            area = 1.0  # Full overlap
                        else:
                            continue  # No overlap
                    elif kernel == 'gaussian':
                        # Gaussian drop
                        sigma = drop_radius / 2
                        r_sq = dy*dy + dx*dx
                        area = np.exp(-0.5 * r_sq / (sigma*sigma))
                        
                        # Skip if negligible
                        if area < 0.01:
                            continue
                    elif kernel == 'lanczos':
                        # Lanczos kernel
                        r = np.sqrt(dy*dy + dx*dx) / drop_radius
                        if r < 1e-10:
                            area = 1.0
                        elif r >= 1.0:
                            continue  # No overlap
                        else:
                            area = np.sinc(r) * np.sin(np.pi * r) / (np.pi * r)
                    else:
                        # Default to square
                        if (abs(dy) <= drop_radius) and (abs(dx) <= drop_radius):
                            area = 1.0
                        else:
                            continue
                    
                    # Update output and weight
                    if is_rgb:
                        output[y_out, x_out, :] += pixel_value * area
                        weight[y_out, x_out] += area
                    else:
                        output[y_out, x_out] += pixel_value * area
                        weight[y_out, x_out] += area

def drizzle_from_alignment_info(input_directory, alignment_file, output_directory, 
                              pixfrac=0.7, scale=1.5, kernel='square'):
    """
    Apply drizzle algorithm using alignment info from previous registration
    
    Parameters:
    -----------
    input_directory : str
        Directory containing original FITS files
    alignment_file : str
        Path to CSV file with alignment information
    output_directory : str
        Directory to save drizzled output
    pixfrac : float
        Pixel fraction for drizzle
    scale : float
        Scale factor for output
    kernel : str
        Drizzle kernel shape
        
    Returns:
    --------
    drizzled : ndarray
        Drizzled image
    """
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Load alignment info
    alignment_info = pd.read_csv(alignment_file)
    
    # Get list of files
    filenames = alignment_info['filename'].tolist()
    
    # Get shifts and rotations
    shifts = []
    rotations = []
    
    for _, row in alignment_info.iterrows():
        shifts.append((row['shift_y'], row['shift_x']))
        if 'rotation' in row:
            rotations.append(row['rotation'])
    
    # Load frames
    frames = []
    for filename in filenames:
        try:
            frame_path = os.path.join(input_directory, filename)
            frame = fits.getdata(frame_path)
            frames.append(frame)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"Loaded {len(frames)} frames for drizzle")
    
    # Apply drizzle
    drizzled = drizzle_integration(
        frames, shifts, rotations,
        pixfrac=pixfrac, scale=scale, kernel=kernel
    )
    
    # Save drizzled image
    output_path = os.path.join(output_directory, "drizzled.fits")
    hdu = fits.PrimaryHDU(drizzled)
    
    # Add metadata
    hdu.header['NFRAMES'] = (len(frames), 'Number of frames drizzled')
    hdu.header['PIXFRAC'] = (pixfrac, 'Drizzle pixel fraction')
    hdu.header['SCALE'] = (scale, 'Drizzle scale factor')
    hdu.header['KERNEL'] = (kernel, 'Drizzle kernel')
    hdu.header['DATE'] = (datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'Processing date')
    
    # Write file
    hdu.writeto(output_path, overwrite=True)
    print(f"Drizzled image saved to {output_path}")
    
    # Create visualization
    visualize_drizzle(frames[0], drizzled, output_directory)
    
    return drizzled

def visualize_drizzle(original, drizzled, output_directory):
    """
    Create a visualization comparing original and drizzled images
    
    Parameters:
    -----------
    original : ndarray
        Original reference frame
    drizzled : ndarray
        Drizzled image
    output_directory : str
        Directory to save visualization
    """
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Determine if RGB
    is_rgb = len(original.shape) == 3
    
    # Process for better visualization
    from galaxy_registration import preprocess_galaxy_image
    
    if is_rgb:
        # Process RGB images
        original_viz = preprocess_galaxy_image(original)
        drizzled_viz = preprocess_galaxy_image(drizzled)
        
        # Show original
        plt.subplot(2, 2, 1)
        plt.imshow(original)
        plt.title("Original Frame")
        plt.axis('off')
        
        # Show drizzled
        plt.subplot(2, 2, 2)
        plt.imshow(drizzled)
        plt.title("Drizzled Result")
        plt.axis('off')
        
        # Show enhanced original
        plt.subplot(2, 2, 3)
        plt.imshow(original_viz, cmap='viridis')
        plt.title("Original (Enhanced)")
        plt.axis('off')
        
        # Show enhanced drizzled
        plt.subplot(2, 2, 4)
        plt.imshow(drizzled_viz, cmap='viridis')
        plt.title("Drizzled (Enhanced)")
        plt.axis('off')
    else:
        # Process grayscale images
        original_viz = preprocess_galaxy_image(original)
        drizzled_viz = preprocess_galaxy_image(drizzled)
        
        # Show original
        plt.subplot(2, 2, 1)
        plt.imshow(original_viz, cmap='viridis')
        plt.title("Original Frame")
        plt.axis('off')
        
        # Show drizzled
        plt.subplot(2, 2, 2)
        plt.imshow(drizzled_viz, cmap='viridis')
        plt.title("Drizzled Result")
        plt.axis('off')
        
        # Zoom in on a region with interesting detail
        h, w = original.shape
        center_y, center_x = h // 2, w // 2
        size = min(h, w) // 4
        
        # Extract regions (compensate for scale difference)
        scale_factor = drizzled.shape[0] / original.shape[0]
        
        original_region = original_viz[
            center_y-size//2:center_y+size//2, 
            center_x-size//2:center_x+size//2
        ]
        
        drizzled_region = drizzled_viz[
            int((center_y-size//2) * scale_factor):int((center_y+size//2) * scale_factor),
            int((center_x-size//2) * scale_factor):int((center_x+size//2) * scale_factor)
        ]
        
        # Show zoomed regions
        plt.subplot(2, 2, 3)
        plt.imshow(original_region, cmap='inferno')
        plt.title("Original (Zoomed)")
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(drizzled_region, cmap='inferno')
        plt.title("Drizzled (Zoomed)")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_directory, "drizzle_comparison.png"), dpi=150)
    plt.close()

def main():
    """
    Main function for command line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Drizzle Integration for Aligned Galaxy Images")
    parser.add_argument("input_dir", help="Directory containing original FITS files")
    parser.add_argument("alignment_file", help="CSV file with alignment information")
    parser.add_argument("output_dir", help="Directory to save drizzled output")
    parser.add_argument("--pixfrac", type=float, default=0.7, help="Pixel fraction (0.5-1.0)")
    parser.add_argument("--scale", type=float, default=1.5, help="Scale factor (1.0-3.0)")
    parser.add_argument("--kernel", choices=["square", "gaussian", "lanczos"], default="square", 
                       help="Drizzle kernel shape")
    
    args = parser.parse_args()
    
    # Apply drizzle
    drizzled = drizzle_from_alignment_info(
        args.input_dir,
        args.alignment_file,
        args.output_dir,
        pixfrac=args.pixfrac,
        scale=args.scale,
        kernel=args.kernel
    )

if __name__ == "__main__":
    main()
