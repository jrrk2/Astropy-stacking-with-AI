import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astropy.visualization import ZScaleInterval
from scipy.spatial import cKDTree
from astropy.wcs import WCS
from skimage.transform import SimilarityTransform, warp
from scipy.ndimage import median_filter, gaussian_filter
import os
import multiprocessing as mp
from tqdm import tqdm
import gc

def filter_hot_pixels(data, filter_size=3, hot_pixel_threshold=5.0):
    """
    Filter out hot pixels from an astronomical image.
    
    Parameters:
    -----------
    data : ndarray
        Image data
    filter_size : int
        Size of the median filter kernel
    hot_pixel_threshold : float
        Threshold in sigma for detecting hot pixels
        
    Returns:
    --------
    filtered_data : ndarray
        Image with hot pixels removed
    hot_pixel_mask : ndarray
        Boolean mask of hot pixels (True = hot pixel)
    """
    # Apply median filter to get reference image without hot pixels
    median_filtered = median_filter(data, size=filter_size)
    
    # Calculate deviation from median filter
    deviation = data - median_filtered
    
    # Calculate statistics of the deviation using sigma clipping
    mean, median, std = sigma_clipped_stats(deviation, sigma=3.0)
    
    # Identify hot pixels as those significantly above median in deviation
    hot_pixel_mask = deviation > (median + hot_pixel_threshold * std)
    
    # Replace hot pixels with median filtered values
    filtered_data = data.copy()
    filtered_data[hot_pixel_mask] = median_filtered[hot_pixel_mask]
    
    return filtered_data, hot_pixel_mask

def detect_stars(data, fwhm=3.0, threshold=5.0, filter_hot_pixels=True, 
                filter_size=3, hot_pixel_threshold=5.0):
    """
    Detect stars in an astronomical image.
    
    Parameters:
    -----------
    data : ndarray
        Image data
    fwhm : float
        Full-width half-maximum (FWHM) of the PSF in pixels
    threshold : float
        Detection threshold in sigma above background
    filter_hot_pixels : bool
        Whether to filter hot pixels before detection
    filter_size : int
        Size of the median filter kernel for hot pixel detection
    hot_pixel_threshold : float
        Threshold in sigma for detecting hot pixels
        
    Returns:
    --------
    stars : table
        Table of detected stars with positions and fluxes
    filtered_data : ndarray or None
        Filtered image data if filter_hot_pixels is True, else None
    """
    # Filter hot pixels if requested
    if filter_hot_pixels:
        filtered_data, hot_pixel_mask = filter_hot_pixels(data, filter_size, hot_pixel_threshold)
        # Use filtered data for star detection
        detection_data = filtered_data
    else:
        filtered_data = None
        detection_data = data
    
    # Calculate background statistics with sigma clipping
    mean, median, std = sigma_clipped_stats(detection_data, sigma=3.0)
    
    # Create DAOStarFinder instance
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    
    # Find stars
    sources = daofind(detection_data - median)
    
    return sources, filtered_data

def match_stars(ref_stars, stars, max_distance=10.0):
    """
    Match stars between two images using a KD-tree.
    
    Parameters:
    -----------
    ref_stars : table
        Reference star catalog
    stars : table
        Star catalog to match against reference
    max_distance : float
        Maximum distance in pixels for a match
        
    Returns:
    --------
    matches : list of tuples
        List of (ref_idx, star_idx) pairs for matched stars
    """
    # Extract coordinates
    ref_coords = np.array([ref_stars['xcentroid'], ref_stars['ycentroid']]).T
    coords = np.array([stars['xcentroid'], stars['ycentroid']]).T
    
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(coords)
    
    # Find nearest neighbors for each reference star
    matches = []
    
    for i, ref_pos in enumerate(ref_coords):
        # Find nearest star
        distances, indices = tree.query(ref_pos, k=1)
        
        # Keep match if distance is below threshold
        if distances < max_distance:
            matches.append((i, indices))
    
    return matches

def estimate_transform(ref_stars, stars, matches):
    """
    Estimate transformation between two sets of stars.
    
    Parameters:
    -----------
    ref_stars : table
        Reference star catalog
    stars : table
        Star catalog to transform
    matches : list of tuples
        List of (ref_idx, star_idx) pairs for matched stars
        
    Returns:
    --------
    transform : SimilarityTransform
        Transformation to align stars with reference stars
    """
    if len(matches) < 3:
        print(f"Warning: Only {len(matches)} matches found. Transformation may be unreliable.")
        # Fall back to identity transformation if not enough matches
        if len(matches) <= 1:
            transform = SimilarityTransform()
            return transform
    
    # Extract matched coordinates
    src_coords = np.array([(stars['xcentroid'][j], stars['ycentroid'][j]) for _, j in matches])
    dst_coords = np.array([(ref_stars['xcentroid'][i], ref_stars['ycentroid'][i]) for i, _ in matches])
    
    # Estimate transformation
    transform = SimilarityTransform()
    transform.estimate(src_coords, dst_coords)
    
    return transform

def align_image(image, transform):
    """
    Apply transformation to align an image.
    
    Parameters:
    -----------
    image : ndarray
        Image to align
    transform : SimilarityTransform
        Transformation to apply
        
    Returns:
    --------
    aligned_image : ndarray
        Aligned image
    """
    # Apply transformation
    aligned_image = warp(image, transform.inverse, preserve_range=True)
    
    return aligned_image.astype(image.dtype)

def process_frame(filename, ref_stars=None, ref_image=None, fwhm=3.0, threshold=5.0,
                filter_hot_pixels=True, filter_size=3, hot_pixel_threshold=5.0,
                cosmic_ray_filter=True, cosmic_ray_threshold=10.0):
    """
    Process a single frame for star-based alignment.
    
    Parameters:
    -----------
    filename : str
        Path to FITS file
    ref_stars : table or None
        Reference star catalog
    ref_image : ndarray or None
        Reference image (used if no reference stars provided)
    fwhm, threshold : float
        Parameters for star detection
    filter_hot_pixels : bool
        Whether to filter hot pixels
    filter_size : int
        Size of the median filter kernel
    hot_pixel_threshold : float
        Threshold for hot pixel detection
    cosmic_ray_filter : bool
        Whether to filter cosmic rays
    cosmic_ray_threshold : float
        Threshold for cosmic ray detection
        
    Returns:
    --------
    result : dict
        Dictionary with processed data
    """
    # Load image
    with fits.open(filename) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy() if hdul[0].header else None
    
    # Detect stars (includes hot pixel filtering)
    stars, filtered_data = detect_stars(
        data, 
        fwhm=fwhm, 
        threshold=threshold,
        filter_hot_pixels=filter_hot_pixels,
        filter_size=filter_size,
        hot_pixel_threshold=hot_pixel_threshold
    )
    
    # Use filtered data if available, otherwise use original
    if filtered_data is not None:
        processed_data = filtered_data
    else:
        processed_data = data
    
    # Filter cosmic rays if requested
    if cosmic_ray_filter:
        # Use Laplacian-based detection for cosmic rays
        from scipy import ndimage
        
        # Laplacian filter helps detect sharp edges characteristic of cosmic rays
        laplacian = ndimage.laplace(processed_data)
        
        # Calculate statistics of Laplacian using sigma clipping
        l_mean, l_median, l_std = sigma_clipped_stats(laplacian, sigma=3.0)
        
        # Identify cosmic rays as pixels with unusually high Laplacian values
        cosmic_ray_mask = laplacian > (l_median + cosmic_ray_threshold * l_std)
        
        # Apply small dilation to ensure full cosmic ray is covered
        cosmic_ray_mask = ndimage.binary_dilation(cosmic_ray_mask, structure=np.ones((3, 3)))
        
        # Replace cosmic rays with local median values
        local_median = median_filter(processed_data, size=5)
        processed_data = processed_data.copy()
        processed_data[cosmic_ray_mask] = local_median[cosmic_ray_mask]
    
    # If no reference stars provided, return unaligned image and stars
    if ref_stars is None:
        return {
            'image': processed_data,
            'original_image': data,
            'stars': stars,
            'header': header,
            'aligned': False
        }
    
    # Match stars with reference
    matches = match_stars(ref_stars, stars)
    
    # Estimate and apply transformation if enough matches
    if len(matches) >= 3:
        transform = estimate_transform(ref_stars, stars, matches)
        aligned_image = align_image(processed_data, transform)
        
        return {
            'image': aligned_image,
            'original_image': data,
            'stars': stars,
            'matches': matches,
            'transform': transform,
            'header': header,
            'aligned': True
        }
    else:
        print(f"Warning: Not enough matches in {filename}. Using unaligned image.")
        return {
            'image': processed_data,
            'original_image': data,
            'stars': stars,
            'header': header,
            'aligned': False
        }

def stack_frames(processed_frames, method='median', weights=None):
    """
    Stack processed frames using specified method.
    
    Parameters:
    -----------
    processed_frames : list
        List of dictionaries with processed frame data
    method : str
        Stacking method ('mean', 'median', 'sum')
    weights : list or None
        Optional weights for weighted mean
        
    Returns:
    --------
    stacked_image : ndarray
        Stacked image
    """
    # Extract images
    images = [frame['image'] for frame in processed_frames if frame['aligned'] or 'aligned' not in frame]
    
    if not images:
        raise ValueError("No images to stack")
    
    # Stack images using specified method
    if method == 'mean':
        if weights is not None:
            # Weighted mean
            stacked = np.average(images, axis=0, weights=weights)
        else:
            stacked = np.mean(images, axis=0)
    elif method == 'median':
        stacked = np.median(images, axis=0)
    elif method == 'sum':
        stacked = np.sum(images, axis=0)
    else:
        raise ValueError(f"Unknown stacking method: {method}")
    
    return stacked

def star_align_stack(filenames, output_file="star_aligned_stack.fits", 
                    fwhm=3.0, threshold=5.0, method='median', 
                    ref_idx=0, batch_size=10, n_processes=None,
                    filter_hot_pixels=True, filter_size=3, hot_pixel_threshold=5.0,
                    cosmic_ray_filter=True, cosmic_ray_threshold=10.0):
    """
    Align and stack FITS images based on star positions.
    
    Parameters:
    -----------
    filenames : list
        List of FITS file paths
    output_file : str
        Path to save the final stacked image
    fwhm : float
        FWHM parameter for star detection
    threshold : float
        Detection threshold in sigma above background
    method : str
        Stacking method ('mean', 'median', 'sum')
    ref_idx : int
        Index of reference file in filenames
    batch_size : int
        Number of files to process in each batch
    n_processes : int or None
        Number of processes for parallel processing
        
    Returns:
    --------
    stacked_image : ndarray
        Stacked image
    """
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    print(f"Processing {len(filenames)} FITS files using {n_processes} processes")
    
    # Get reference frame
    ref_filename = filenames[ref_idx]
    print(f"Using {ref_filename} as reference frame")
    
    ref_result = process_frame(ref_filename, ref_stars=None, ref_image=None, 
                              fwhm=fwhm, threshold=threshold,
                              filter_hot_pixels=filter_hot_pixels, 
                              filter_size=filter_size,
                              hot_pixel_threshold=hot_pixel_threshold,
                              cosmic_ray_filter=cosmic_ray_filter,
                              cosmic_ray_threshold=cosmic_ray_threshold)
    ref_image = ref_result['image']
    ref_stars = ref_result['stars']
    ref_header = ref_result['header']
    
    if ref_stars is None or len(ref_stars) < 5:
        print(f"Warning: Few stars detected in reference image ({len(ref_stars) if ref_stars else 0})")
        if ref_stars is None or len(ref_stars) < 3:
            raise ValueError("Not enough stars in reference image for alignment")
    
    print(f"Detected {len(ref_stars)} stars in reference image")
    
    # Process remaining frames in batches
    all_processed = [ref_result]  # Include reference frame
    remaining_files = filenames.copy()
    remaining_files.pop(ref_idx)  # Remove reference frame
    
    for batch_start in tqdm(range(0, len(remaining_files), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(remaining_files))
        batch_files = remaining_files[batch_start:batch_end]
        
        # Process batch in parallel
        with mp.Pool(n_processes) as pool:
            batch_results = pool.starmap(
                process_frame,
                [(filename, ref_stars, ref_image, fwhm, threshold,
                 filter_hot_pixels, filter_size, hot_pixel_threshold,
                 cosmic_ray_filter, cosmic_ray_threshold) for filename in batch_files]
            )
        
        # Add to all processed frames
        all_processed.extend(batch_results)
        
        # Clear memory
        del batch_results
        gc.collect()
        
        # Periodically save intermediate results
        if batch_end % (batch_size * 5) == 0 or batch_end == len(remaining_files):
            intermediate_stack = stack_frames(all_processed, method=method)
            intermediate_name = f"intermediate_stack_{batch_end}.fits"
            fits.writeto(intermediate_name, intermediate_stack, header=ref_header, overwrite=True)
            print(f"Intermediate stack saved to {intermediate_name}")
    
    # Stack all processed frames
    stacked_image = stack_frames(all_processed, method=method)
    
    # Save final stacked image
    fits.writeto(output_file, stacked_image, header=ref_header, overwrite=True)
    print(f"Stacked image saved to {output_file}")
    
    return stacked_image, all_processed

def choose_best_reference(filenames, fwhm=3.0, threshold=5.0, n_candidates=5, n_processes=None,
                     filter_hot_pixels=True, filter_size=3, hot_pixel_threshold=5.0,
                     cosmic_ray_filter=True, cosmic_ray_threshold=10.0):
    """
    Choose the best reference frame based on number and brightness of stars.
    
    Parameters:
    -----------
    filenames : list
        List of FITS file paths
    fwhm, threshold : float
        Parameters for star detection
    n_candidates : int
        Number of files to evaluate as potential reference frames
    n_processes : int or None
        Number of processes for parallel processing
        
    Returns:
    --------
    best_idx : int
        Index of the best reference frame in filenames
    """
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    # Randomly select candidate files if there are many
    import random
    if len(filenames) <= n_candidates:
        candidates = filenames
        candidate_indices = list(range(len(filenames)))
    else:
        candidate_indices = sorted(random.sample(range(len(filenames)), n_candidates))
        candidates = [filenames[i] for i in candidate_indices]
    
    print(f"Evaluating {len(candidates)} candidate reference frames")
    
    # Process candidate frames in parallel
    with mp.Pool(n_processes) as pool:
        candidate_results = pool.starmap(
            process_frame,
            [(filename, None, None, fwhm, threshold,
              filter_hot_pixels, filter_size, hot_pixel_threshold,
              cosmic_ray_filter, cosmic_ray_threshold) for filename in candidates]
        )
    
    # Score candidates based on number and quality of stars
    scores = []
    for result in candidate_results:
        stars = result['stars']
        if stars is None:
            scores.append(0)
            continue
        
        # Score based on number of stars and their brightness
        n_stars = len(stars)
        if n_stars == 0:
            scores.append(0)
            continue
        
        avg_flux = np.mean(stars['peak'])
        score = n_stars * avg_flux
        scores.append(score)
    
    # Find best candidate
    if not scores:
        best_candidate_idx = 0
    else:
        best_candidate_idx = np.argmax(scores)
    best_idx = candidate_indices[best_candidate_idx]
    
    print(f"Selected {filenames[best_idx]} as best reference frame")
    print(f"It contains {len(candidate_results[best_candidate_idx]['stars'])} stars")
    
    return best_idx

def visualize_alignment(reference_result, aligned_result, title="Alignment Visualization", show_hot_pixels=True):
    """
    Visualize the alignment between reference and aligned frames.
    
    Parameters:
    -----------
    reference_result : dict
        Reference frame processing result
    aligned_result : dict
        Aligned frame processing result
    title : str
        Plot title
    """
    if show_hot_pixels and 'original_image' in reference_result:
        # Show original and filtered images to visualize hot pixel removal
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Set up scaling for consistent display
    interval = ZScaleInterval()
    vmin1, vmax1 = interval.get_limits(reference_result['image'])
    vmin2, vmax2 = interval.get_limits(aligned_result['image'])
    
    # Plot reference image
    axes[0, 0].imshow(reference_result['image'], cmap='viridis', vmin=vmin1, vmax=vmax1)
    axes[0, 0].set_title("Reference Image")
    
    # Plot detected stars in reference image
    ref_stars = reference_result['stars']
    if ref_stars:
        axes[0, 0].scatter(ref_stars['xcentroid'], ref_stars['ycentroid'], 
                          s=50, edgecolor='red', facecolor='none')
    
    # Plot aligned image
    axes[0, 1].imshow(aligned_result['image'], cmap='viridis', vmin=vmin2, vmax=vmax2)
    axes[0, 1].set_title("Aligned Image")
    
    # Plot detected stars in aligned image
    aligned_stars = aligned_result['stars']
    if aligned_stars:
        axes[0, 1].scatter(aligned_stars['xcentroid'], aligned_stars['ycentroid'], 
                          s=50, edgecolor='blue', facecolor='none')
    
    # Plot difference image
    diff = reference_result['image'] - aligned_result['image']
    vmin_diff, vmax_diff = interval.get_limits(diff)
    axes[1, 0].imshow(diff, cmap='RdBu_r', vmin=-max(abs(vmin_diff), abs(vmax_diff)), 
                     vmax=max(abs(vmin_diff), abs(vmax_diff)))
    axes[1, 0].set_title("Difference (Reference - Aligned)")
    
    # Plot star matches if available
    if 'matches' in aligned_result:
        axes[1, 1].imshow(reference_result['image'], cmap='viridis', alpha=0.5, 
                         vmin=vmin1, vmax=vmax1)
        
        matches = aligned_result['matches']
        for ref_idx, star_idx in matches:
            ref_x = ref_stars['xcentroid'][ref_idx]
            ref_y = ref_stars['ycentroid'][ref_idx]
            star_x = aligned_stars['xcentroid'][star_idx]
            star_y = aligned_stars['ycentroid'][star_idx]
            
            axes[1, 1].plot([ref_x, star_x], [ref_y, star_y], 'g-', alpha=0.5)
            axes[1, 1].plot(ref_x, ref_y, 'ro', markersize=5)
            axes[1, 1].plot(star_x, star_y, 'bo', markersize=5)
        
        axes[1, 1].set_title(f"Star Matches ({len(matches)} pairs)")
    else:
        axes[1, 1].axis('off')
    
    # Show hot pixel filtering results if requested and available
    if show_hot_pixels and 'original_image' in reference_result:
        # Plot original reference image
        if axes.shape[1] > 2:  # We have 3 columns
            vmin_orig, vmax_orig = interval.get_limits(reference_result['original_image'])
            axes[0, 2].imshow(reference_result['original_image'], cmap='viridis', 
                             vmin=vmin_orig, vmax=vmax_orig)
            axes[0, 2].set_title("Original Reference (Unfiltered)")
            
            # Plot difference to show removed hot pixels
            diff_hot = reference_result['original_image'] - reference_result['image']
            vmin_hot, vmax_hot = interval.get_limits(diff_hot)
            axes[1, 2].imshow(diff_hot, cmap='hot', vmin=vmin_hot, vmax=vmax_hot)
            axes[1, 2].set_title("Removed Hot Pixels/Cosmic Rays")
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

def multi_reference_align_stack(filenames, output_file="multi_ref_stack.fits", 
                               fwhm=3.0, threshold=5.0, method='median',
                               n_refs=3, batch_size=10, n_processes=None,
                               filter_hot_pixels=True, filter_size=3, hot_pixel_threshold=5.0,
                               cosmic_ray_filter=True, cosmic_ray_threshold=10.0):
    """
    Align and stack using multiple reference frames for improved robustness.
    
    Parameters:
    -----------
    filenames : list
        List of FITS file paths
    output_file : str
        Path to save the final stacked image
    fwhm : float
        FWHM parameter for star detection
    threshold : float
        Detection threshold in sigma above background
    method : str
        Stacking method ('mean', 'median', 'sum')
    n_refs : int
        Number of reference frames to use
    batch_size : int
        Number of files to process in each batch
    n_processes : int or None
        Number of processes for parallel processing
        
    Returns:
    --------
    stacked_image : ndarray
        Stacked image
    """
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    print(f"Processing {len(filenames)} FITS files using {n_processes} processes")
    
    # Choose best reference frames
    ref_indices = []
    
    # Choose first reference
    ref_idx = choose_best_reference(filenames, fwhm, threshold, 
                                   n_candidates=min(10, len(filenames)),
                                   filter_hot_pixels=filter_hot_pixels,
                                   filter_size=filter_size,
                                   hot_pixel_threshold=hot_pixel_threshold,
                                   cosmic_ray_filter=cosmic_ray_filter,
                                   cosmic_ray_threshold=cosmic_ray_threshold)
    ref_indices.append(ref_idx)
    
    # Choose additional references
    remaining = filenames.copy()
    for _ in range(1, n_refs):
        # Remove already selected references
        for idx in sorted(ref_indices, reverse=True):
            if idx < len(remaining):
                remaining.pop(idx)
        
        if not remaining:
            break
            
        # Choose next reference
        next_ref_idx = choose_best_reference(remaining, fwhm, threshold, 
                                            n_candidates=min(5, len(remaining)))
        
        # Convert index in remaining to index in original filenames
        original_idx = 0
        skipped = 0
        for i in range(len(filenames)):
            if i in ref_indices:
                skipped += 1
                continue
            if original_idx == next_ref_idx:
                ref_indices.append(i)
                break
            original_idx += 1
    
    print(f"Selected {len(ref_indices)} reference frames: {[filenames[i] for i in ref_indices]}")
    
    # Process each reference frame
    ref_results = []
    for ref_idx in ref_indices:
        ref_result = process_frame(filenames[ref_idx], ref_stars=None, ref_image=None, 
                                  fwhm=fwhm, threshold=threshold)
        ref_results.append(ref_result)
    
    # Use first reference's header for output
    ref_header = ref_results[0]['header']
    
    # Process all frames against each reference
    all_alignments = []
    
    for i, ref_result in enumerate(ref_results):
        ref_stars = ref_result['stars']
        ref_image = ref_result['image']
        
        print(f"Aligning against reference {i+1}/{len(ref_results)}")
        
        # Process files against this reference
        alignments = []
        
        for batch_start in tqdm(range(0, len(filenames), batch_size), 
                              desc=f"Processing batch (ref {i+1})"):
            batch_end = min(batch_start + batch_size, len(filenames))
            batch_files = filenames[batch_start:batch_end]
            
            # Process batch in parallel
            with mp.Pool(n_processes) as pool:
                batch_results = pool.starmap(
                    process_frame,
                    [(filename, ref_stars, ref_image, fwhm, threshold) for filename in batch_files]
                )
            
            # Add to alignments
            alignments.extend(batch_results)
            
            # Clear memory
            del batch_results
            gc.collect()
        
        all_alignments.append(alignments)
    
    # For each frame, choose the best alignment
    best_alignments = []
    
    for frame_idx in range(len(filenames)):
        frame_alignments = [alignments[frame_idx] for alignments in all_alignments]
        
        # Choose best alignment based on number of star matches
        best_idx = 0
        most_matches = 0
        
        for i, alignment in enumerate(frame_alignments):
            if 'matches' in alignment and alignment['aligned']:
                n_matches = len(alignment['matches'])
                if n_matches > most_matches:
                    most_matches = n_matches
                    best_idx = i
        
        best_alignments.append(frame_alignments[best_idx])
    
    # Stack best alignments
    stacked_image = stack_frames(best_alignments, method=method)
    
    # Save final stacked image
    fits.writeto(output_file, stacked_image, header=ref_header, overwrite=True)
    print(f"Stacked image saved to {output_file}")
    
    return stacked_image, best_alignments

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Star-based alignment and stacking for astronomical images")
    parser.add_argument("--directory", type=str, help="Directory containing FITS files")
    parser.add_argument("--output", type=str, default="star_aligned_stack.fits", 
                        help="Output FITS file")
    parser.add_argument("--pattern", type=str, default="*.fits", 
                        help="Pattern to match FITS files")
    parser.add_argument("--fwhm", type=float, default=3.0, 
                        help="FWHM for star detection (pixels)")
    parser.add_argument("--threshold", type=float, default=5.0, 
                        help="Detection threshold (sigma)")
    parser.add_argument("--method", type=str, default="median", 
                        choices=["mean", "median", "sum"], help="Stacking method")
    parser.add_argument("--batch-size", type=int, default=10, 
                        help="Batch size for processing")
    parser.add_argument("--processes", type=int, default=None, 
                        help="Number of parallel processes")
    parser.add_argument("--multi-ref", action="store_true", 
                        help="Use multiple reference frames")
    parser.add_argument("--n-refs", type=int, default=3, 
                        help="Number of reference frames to use with --multi-ref")
    parser.add_argument("--no-hot-pixel-filter", action="store_true",
                        help="Disable hot pixel filtering")
    parser.add_argument("--hot-pixel-threshold", type=float, default=5.0,
                        help="Threshold for hot pixel detection")
    parser.add_argument("--filter-size", type=int, default=3,
                        help="Size of median filter for hot pixel removal")
    parser.add_argument("--no-cosmic-ray-filter", action="store_true",
                        help="Disable cosmic ray filtering")
    parser.add_argument("--cosmic-ray-threshold", type=float, default=10.0,
                        help="Threshold for cosmic ray detection")
    
    args = parser.parse_args()
    
    if args.directory:
        import glob
        
        # Find all FITS files matching the pattern
        fits_files = sorted(glob.glob(os.path.join(args.directory, args.pattern)))
        if not fits_files:
            raise ValueError(f"No FITS files found in {args.directory} matching {args.pattern}")
        
        print(f"Found {len(fits_files)} FITS files to process")
        
        # Set filtering parameters based on command line arguments
        filter_hot_pixels = not args.no_hot_pixel_filter
        filter_size = args.filter_size
        hot_pixel_threshold = args.hot_pixel_threshold
        cosmic_ray_filter = not args.no_cosmic_ray_filter
        cosmic_ray_threshold = args.cosmic_ray_threshold
        
        if args.multi_ref:
            multi_reference_align_stack(
                fits_files,
                output_file=args.output,
                fwhm=args.fwhm,
                threshold=args.threshold,
                method=args.method,
                n_refs=args.n_refs,
                batch_size=args.batch_size,
                n_processes=args.processes,
                filter_hot_pixels=filter_hot_pixels,
                filter_size=filter_size,
                hot_pixel_threshold=hot_pixel_threshold,
                cosmic_ray_filter=cosmic_ray_filter,
                cosmic_ray_threshold=cosmic_ray_threshold
            )
        else:
            # Choose best reference frame
            ref_idx = choose_best_reference(
                fits_files, 
                fwhm=args.fwhm, 
                threshold=args.threshold,
                filter_hot_pixels=filter_hot_pixels,
                filter_size=filter_size,
                hot_pixel_threshold=hot_pixel_threshold,
                cosmic_ray_filter=cosmic_ray_filter,
                cosmic_ray_threshold=cosmic_ray_threshold
            )
            
            # Align and stack
            star_align_stack(
                fits_files,
                output_file=args.output,
                fwhm=args.fwhm,
                threshold=args.threshold,
                method=args.method,
                ref_idx=ref_idx,
                batch_size=args.batch_size,
                n_processes=args.processes,
                filter_hot_pixels=filter_hot_pixels,
                filter_size=filter_size,
                hot_pixel_threshold=hot_pixel_threshold,
                cosmic_ray_filter=cosmic_ray_filter,
                cosmic_ray_threshold=cosmic_ray_threshold
            )
    else:
        parser.print_help()
