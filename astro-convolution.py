import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve, Box2DKernel, Gaussian2DKernel
from astropy.visualization import simple_norm, ZScaleInterval
import os
from tqdm import tqdm
import gc
import multiprocessing as mp
from functools import partial

def load_fits_frames(filenames):
    """Load multiple FITS files into a list of numpy arrays."""
    frames = []
    for filename in filenames:
        with fits.open(filename) as hdul:
            frames.append(hdul[0].data)
    return frames

def register_frames(frames):
    """
    Simple frame registration using cross-correlation.
    In practice, you'd use a more sophisticated registration method.
    """
    from scipy.ndimage import shift
    from skimage.registration import phase_cross_correlation
    
    reference = frames[0]
    aligned_frames = [reference]
    
    for frame in frames[1:]:
        # Find shift between this frame and reference
        shift_vector, error, _ = phase_cross_correlation(reference, frame)
        # Apply shift to align with reference
        shifted_frame = shift(frame, shift_vector)
        aligned_frames.append(shifted_frame)
    
    return aligned_frames

def convolve_frames(frames, kernel=None):
    """
    Convolve each frame with a kernel before stacking.
    This can help weight each frame's contribution based on its quality.
    """
    if kernel is None:
        # Default to a small Gaussian kernel
        kernel = Gaussian2DKernel(x_stddev=1.5)
    
    convolved_frames = []
    for frame in frames:
        convolved = convolve(frame, kernel)
        convolved_frames.append(convolved)
    
    return convolved_frames

def stack_frames(frames, method='mean'):
    """Stack frames using different methods."""
    if method == 'mean':
        return np.mean(frames, axis=0)
    elif method == 'median':
        return np.median(frames, axis=0)
    elif method == 'sum':
        return np.sum(frames, axis=0)
    else:
        raise ValueError(f"Unknown stacking method: {method}")

def process_frame(filename, reference_frame=None, kernel=None):
    """Process a single frame (for parallel processing)"""
    # Load frame
    with fits.open(filename) as hdul:
        frame = hdul[0].data
    
    # Register frame if reference is provided
    if reference_frame is not None:
        from scipy.ndimage import shift
        from skimage.registration import phase_cross_correlation
        
        shift_vector, error, _ = phase_cross_correlation(reference_frame, frame)
        frame = shift(frame, shift_vector)
    
    # Convolve frame if kernel is provided
    if kernel is not None:
        frame = convolve(frame, kernel)
    
    return frame

def convolutional_stacking(filenames, kernel_size=3, kernel_type='gaussian', batch_size=10, n_processes=None):
    """
    Process astronomical frames with convolution between subsequent frames.
    Optimized for large numbers of FITS files using batch processing.
    
    Parameters:
    -----------
    filenames : list
        List of FITS file paths
    kernel_size : float
        Size parameter for the convolution kernel
    kernel_type : str
        Type of kernel ('gaussian' or 'box')
    batch_size : int
        Number of files to process in each batch
    n_processes : int or None
        Number of processes for parallel processing. Default is None (use CPU count).
    
    Returns:
    --------
    stacked_image : ndarray
        Final stacked image
    """
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)  # Leave one CPU free
    
    print(f"Processing {len(filenames)} FITS files using {n_processes} processes")
    
    # Create appropriate kernel
    if kernel_type == 'gaussian':
        kernel = Gaussian2DKernel(x_stddev=kernel_size/2.355)  # FWHM to sigma conversion
    elif kernel_type == 'box':
        kernel = Box2DKernel(kernel_size)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Get reference frame from first file
    with fits.open(filenames[0]) as hdul:
        reference_frame = hdul[0].data.copy()
    
    # Initialize accumulator for running sum and counter
    with fits.open(filenames[0]) as hdul:
        sum_image = np.zeros_like(hdul[0].data, dtype=np.float64)
    
    n_frames_processed = 0
    
    # Process files in batches to manage memory
    for batch_start in tqdm(range(0, len(filenames), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(filenames))
        batch_files = filenames[batch_start:batch_end]
        
        # Process batch in parallel
        with mp.Pool(n_processes) as pool:
            process_func = partial(process_frame, reference_frame=reference_frame, kernel=kernel)
            processed_frames = pool.map(process_func, batch_files)
        
        # Add batch to running sum
        for frame in processed_frames:
            sum_image += frame
            n_frames_processed += 1
        
        # Clear memory
        del processed_frames
        gc.collect()
        
        # Periodically save intermediate results
        if batch_end % (batch_size * 5) == 0 or batch_end == len(filenames):
            intermediate_avg = sum_image / n_frames_processed
            fits.writeto(f"intermediate_stack_{batch_end}.fits", 
                         intermediate_avg, overwrite=True)
    
    # Calculate final average
    stacked_image = sum_image / n_frames_processed
    
    return stacked_image

def plot_results(original_frames, convolved_frames, stacked_image):
    """Plot original, convolved, and stacked frames for comparison."""
    n_frames = len(original_frames)
    fig = plt.figure(figsize=(15, 10))
    
    # Set up scaling for consistent display
    interval = ZScaleInterval()
    
    # Plot original frames
    for i, frame in enumerate(original_frames[:min(3, n_frames)]):
        ax = fig.add_subplot(3, 4, i+1)
        vmin, vmax = interval.get_limits(frame)
        ax.imshow(frame, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"Original Frame {i+1}")
        ax.axis('off')
    
    # Plot convolved frames
    for i, frame in enumerate(convolved_frames[:min(3, n_frames)]):
        ax = fig.add_subplot(3, 4, i+5)
        vmin, vmax = interval.get_limits(frame)
        ax.imshow(frame, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"Convolved Frame {i+1}")
        ax.axis('off')
    
    # Plot stacked image
    ax = fig.add_subplot(3, 4, 9)
    vmin, vmax = interval.get_limits(stacked_image)
    ax.imshow(stacked_image, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title("Stacked Image")
    ax.axis('off')
    
    # Plot difference between stacked and average of originals
    avg_original = np.mean(original_frames, axis=0)
    diff = stacked_image - avg_original
    ax = fig.add_subplot(3, 4, 10)
    vmin, vmax = interval.get_limits(diff)
    ax.imshow(diff, cmap='RdBu_r', vmin=-np.abs(vmin), vmax=np.abs(vmax))
    ax.set_title("Difference (Stacked - Avg)")
    ax.axis('off')
    
    plt.tight_layout()
    return fig

# Add function for parallel processing of large FITS datasets
def process_directory(directory_path, output_file="stacked_result.fits", 
                      pattern="*.fits", kernel_size=3, kernel_type="gaussian",
                      batch_size=10, n_processes=None):
    """
    Process all FITS files in a directory using convolutional stacking.
    
    Parameters:
    -----------
    directory_path : str
        Path to directory containing FITS files
    output_file : str
        Path to save the final stacked image
    pattern : str
        Glob pattern to match FITS files
    kernel_size, kernel_type, batch_size, n_processes:
        See convolutional_stacking function
    """
    import glob
    
    # Find all FITS files matching the pattern
    fits_files = sorted(glob.glob(os.path.join(directory_path, pattern)))
    if not fits_files:
        raise ValueError(f"No FITS files found in {directory_path} matching {pattern}")
    
    print(f"Found {len(fits_files)} FITS files to process")
    
    # Process files
    stacked_image = convolutional_stacking(
        fits_files, 
        kernel_size=kernel_size,
        kernel_type=kernel_type,
        batch_size=batch_size,
        n_processes=n_processes
    )
    
    # Save result
    fits.writeto(output_file, stacked_image, overwrite=True)
    print(f"Stacked image saved to {output_file}")
    
    return stacked_image

# Quality assessment function
def assess_frame_quality(filename):
    """
    Assess the quality of a FITS frame, returning a quality score.
    This can be used to weight frames or exclude low-quality frames.
    
    Parameters:
    -----------
    filename : str
        Path to FITS file
        
    Returns:
    --------
    quality_score : float
        Quality score (higher is better)
    """
    with fits.open(filename) as hdul:
        data = hdul[0].data
    
    # Calculate some quality metrics
    mean = np.mean(data)
    std = np.std(data)
    snr = mean / std if std > 0 else 0
    
    # Calculate sharpness (using Laplacian variance)
    from scipy import ndimage
    laplacian = ndimage.laplace(data)
    sharpness = np.var(laplacian)
    
    # Combine metrics into a single quality score
    # This is a simple example - you may need to adjust weights
    quality_score = 0.5 * snr + 0.5 * sharpness
    
    return quality_score

# Adaptive convolution based on frame quality
def adaptive_convolutional_stacking(filenames, min_kernel_size=1, max_kernel_size=5, 
                                   kernel_type='gaussian', quality_threshold=0.0,
                                   batch_size=10, n_processes=None):
    """
    Stack frames with adaptive convolution based on frame quality.
    
    Parameters:
    -----------
    filenames : list
        List of FITS file paths
    min_kernel_size, max_kernel_size : float
        Minimum and maximum kernel sizes
    kernel_type : str
        Type of kernel ('gaussian' or 'box')
    quality_threshold : float
        Minimum quality score for frames to be included
    batch_size, n_processes:
        See convolutional_stacking function
    """
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    # Assess quality of all frames
    print("Assessing frame quality...")
    with mp.Pool(n_processes) as pool:
        quality_scores = pool.map(assess_frame_quality, filenames)
    
    # Normalize quality scores to 0-1 range if needed
    if max(quality_scores) > min(quality_scores):
        min_qual = min(quality_scores)
        max_qual = max(quality_scores)
        norm_scores = [(q - min_qual) / (max_qual - min_qual) for q in quality_scores]
    else:
        norm_scores = [1.0] * len(quality_scores)
    
    # Filter out low-quality frames
    good_frames = [f for f, q in zip(filenames, norm_scores) if q >= quality_threshold]
    good_scores = [q for q in norm_scores if q >= quality_threshold]
    
    print(f"Using {len(good_frames)} of {len(filenames)} frames after quality filtering")
    
    # Get reference frame (highest quality frame)
    ref_idx = norm_scores.index(max(norm_scores))
    with fits.open(filenames[ref_idx]) as hdul:
        reference_frame = hdul[0].data.copy()
    
    # Initialize accumulator and weighting
    with fits.open(filenames[0]) as hdul:
        sum_image = np.zeros_like(hdul[0].data, dtype=np.float64)
        weight_sum = 0.0
    
    # Process files in batches
    for batch_idx in range(0, len(good_frames), batch_size):
        batch_files = good_frames[batch_idx:batch_idx + batch_size]
        batch_scores = good_scores[batch_idx:batch_idx + batch_size]
        
        # Process each file in the batch with adaptive kernel size
        batch_results = []
        for file_idx, (filename, quality) in enumerate(zip(batch_files, batch_scores)):
            # Scale kernel size based on quality (better quality = smaller kernel)
            kernel_size = max_kernel_size - quality * (max_kernel_size - min_kernel_size)
            
            if kernel_type == 'gaussian':
                kernel = Gaussian2DKernel(x_stddev=kernel_size/2.355)
            else:
                kernel = Box2DKernel(kernel_size)
            
            # Process frame
            processed_frame = process_frame(filename, reference_frame, kernel)
            
            # Weight by quality
            weight = quality
            sum_image += processed_frame * weight
            weight_sum += weight
            
            # Clear memory
            del processed_frame
            
        # Garbage collect
        gc.collect()
    
    # Calculate weighted average
    stacked_image = sum_image / weight_sum if weight_sum > 0 else sum_image
    
    return stacked_image

# Example usage with synthetic data (since we don't have actual FITS files)
def create_synthetic_frames(n_frames=5, size=100, snr_range=(5, 15)):
    """Create synthetic astronomical frames with varying noise levels."""
    frames = []
    filenames = []
    
    # Create base image with a few stars
    base = np.zeros((size, size))
    
    # Add some stars
    for _ in range(20):
        x, y = np.random.randint(10, size-10, 2)
        intensity = np.random.uniform(0.5, 1.0)
        base[y, x] = intensity
    
    # Create frames with different noise levels and small offsets
    os.makedirs("synthetic_fits", exist_ok=True)
    
    for i in range(n_frames):
        # Add random jitter (simulating telescope movement)
        offset_x = np.random.randint(-2, 3)
        offset_y = np.random.randint(-2, 3)
        frame = np.roll(base, (offset_y, offset_x), axis=(0, 1))
        
        # Add noise
        snr = np.random.uniform(snr_range[0], snr_range[1])
        noise_level = np.mean(frame[frame > 0]) / snr
        noise = np.random.normal(0, noise_level, frame.shape)
        frame = frame + noise
        
        # Save to temporary FITS file
        filename = f"synthetic_fits/frame_{i:03d}.fits"
        fits.writeto(filename, frame, overwrite=True)
        
        frames.append(frame)
        filenames.append(filename)
    
    return frames, filenames

# Demo function
def demo(n_frames=100):
    """
    Run a demonstration with synthetic data.
    """
    print(f"Creating {n_frames} synthetic FITS files...")
    frames, filenames = create_synthetic_frames(n_frames=n_frames, size=200, snr_range=(3, 20))
    
    print("Processing with standard convolution stacking...")
    stacked_standard = convolutional_stacking(
        filenames, 
        kernel_size=2, 
        batch_size=10
    )
    
    print("Processing with adaptive convolution stacking...")
    stacked_adaptive = adaptive_convolutional_stacking(
        filenames,
        min_kernel_size=1,
        max_kernel_size=3,
        quality_threshold=0.2,
        batch_size=10
    )
    
    # Compare results
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 3, 1)
    plt.title("Single Frame (First)")
    plt.imshow(frames[0], cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title("Standard Convolution Stacking")
    plt.imshow(stacked_standard, cmap='viridis')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.title("Adaptive Convolution Stacking")
    plt.imshow(stacked_adaptive, cmap='viridis')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("stacking_comparison.png", dpi=200)
    plt.show()
    
    # Clean up synthetic files
    for filename in filenames:
        os.remove(filename)
    os.rmdir("synthetic_fits")
    
    return stacked_standard, stacked_adaptive

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Astronomical image stacking with convolution")
    parser.add_argument("--directory", type=str, help="Directory containing FITS files")
    parser.add_argument("--output", type=str, default="stacked_result.fits", 
                        help="Output FITS file")
    parser.add_argument("--pattern", type=str, default="*.fits", 
                        help="Pattern to match FITS files")
    parser.add_argument("--kernel-size", type=float, default=2.0, 
                        help="Convolution kernel size")
    parser.add_argument("--kernel-type", type=str, default="gaussian", 
                        choices=["gaussian", "box"], help="Convolution kernel type")
    parser.add_argument("--batch-size", type=int, default=10, 
                        help="Batch size for processing")
    parser.add_argument("--processes", type=int, default=None, 
                        help="Number of parallel processes")
    parser.add_argument("--adaptive", action="store_true", 
                        help="Use adaptive convolution based on frame quality")
    parser.add_argument("--demo", action="store_true", 
                        help="Run demonstration with synthetic data")
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    elif args.directory:
        if args.adaptive:
            frames = glob.glob(os.path.join(args.directory, args.pattern))
            result = adaptive_convolutional_stacking(
                frames,
                min_kernel_size=args.kernel_size / 2,
                max_kernel_size=args.kernel_size * 2,
                kernel_type=args.kernel_type,
                batch_size=args.batch_size,
                n_processes=args.processes
            )
            fits.writeto(args.output, result, overwrite=True)
        else:
            process_directory(
                args.directory,
                output_file=args.output,
                pattern=args.pattern,
                kernel_size=args.kernel_size,
                kernel_type=args.kernel_type,
                batch_size=args.batch_size,
                n_processes=args.processes
            )
    else:
        print("Either --directory or --demo must be specified")
        parser.print_help()
