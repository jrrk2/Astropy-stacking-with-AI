import numpy as np
import cv2
from scipy import ndimage
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.spatial import KDTree
from scipy.ndimage import shift as ndimage_shift
from skimage.transform import warp_polar, rotate
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt

def enhanced_registration(frame, reference_frame, detect_rotation=True, max_rotation=1.0):
    """
    Enhanced registration algorithm that combines star detection, feature matching,
    and phase correlation with rotation detection capability.
    
    Parameters:
    -----------
    frame : ndarray
        Input frame to align
    reference_frame : ndarray
        Reference frame to align to
    detect_rotation : bool
        Whether to detect and correct for rotation
    max_rotation : float
        Maximum rotation angle to test in degrees
        
    Returns:
    --------
    aligned_frame : ndarray
        Aligned frame
    transformation : dict
        Dictionary containing alignment parameters
    """
    # Ensure frames are float64
    frame = frame.astype(np.float64)
    reference = reference_frame.astype(np.float64)
    
    # Normalize frames to 0-1 range for preprocessing
    frame_norm = normalize_frame(frame)
    ref_norm = normalize_frame(reference)
    
    # Step 1: Try star-based registration with robust matching
    transformation = star_based_registration(frame_norm, ref_norm)
    
    # Check if star registration was successful
    if transformation['success']:
        # Apply star-based transformation as initial alignment
        if transformation['rotation'] != 0 and detect_rotation:
            # Apply rotation and translation
            aligned_frame = rotate_and_shift_frame(
                frame, 
                transformation['rotation'],
                transformation['shift_y'],
                transformation['shift_x']
            )
        else:
            # Apply just translation
            aligned_frame = ndimage_shift(
                frame, 
                (transformation['shift_y'], transformation['shift_x']), 
                order=3, 
                mode='constant', 
                cval=0
            )
            
        # Calculate quality metrics
        quality = calculate_alignment_quality(aligned_frame, reference)
        transformation['quality'] = quality
        
        # If quality is good, return the result
        if quality['ncc'] > 0.8:  # Good alignment threshold
            transformation['method'] = 'star'
            return aligned_frame, transformation
    
    # Step 2: If star registration failed or quality is poor, try FFT-based registration
    # First correct rotation if needed
    rotation_angle = 0
    if detect_rotation:
        rotation_angle = detect_rotation_angle(frame_norm, ref_norm, max_angle=max_rotation)
        if abs(rotation_angle) > 0.05:  # Only rotate if angle is significant
            frame_rot = rotate(frame, rotation_angle, preserve_range=True)
        else:
            frame_rot = frame
    else:
        frame_rot = frame
    
    # Then detect translation using phase correlation
    shift, error, diffphase = phase_cross_correlation(
        ref_norm, 
        normalize_frame(frame_rot),
        upsample_factor=100  # For subpixel accuracy
    )
    
    # Apply the full transformation
    if abs(rotation_angle) > 0.05:
        aligned_frame = rotate_and_shift_frame(frame, rotation_angle, shift[0], shift[1])
    else:
        aligned_frame = ndimage_shift(frame, shift, order=3, mode='constant', cval=0)
    
    # Calculate quality metrics for FFT-based alignment
    quality = calculate_alignment_quality(aligned_frame, reference)
    
    # Return the transformation details
    fft_transformation = {
        'method': 'fft',
        'shift_y': float(shift[0]),
        'shift_x': float(shift[1]),
        'rotation': float(rotation_angle),
        'success': True,
        'quality': quality
    }
    
    return aligned_frame, fft_transformation

def normalize_frame(frame):
    """
    Normalize and preprocess frame for better registration
    """
    # Calculate background statistics
    mean, median, std = sigma_clipped_stats(frame, sigma=3.0)
    
    # Background subtraction
    frame_bg = frame - median
    
    # Clip negative values
    frame_bg = np.clip(frame_bg, 0, None)
    
    # Apply mild stretch to enhance contrast (square root stretch works well for astronomical images)
    frame_norm = np.sqrt(frame_bg / (np.percentile(frame_bg, 99) + 1e-10))
    
    # Apply Gaussian blur to reduce noise (optional)
    frame_smooth = ndimage.gaussian_filter(frame_norm, sigma=1.0)
    
    # Ensure values are in range 0-1
    frame_final = (frame_smooth - np.min(frame_smooth)) / (np.max(frame_smooth) - np.min(frame_smooth) + 1e-10)
    
    return frame_final

def star_based_registration(frame, reference_frame, num_stars=30):
    """
    Perform star-based registration with robust matching
    """
    # Detect stars in both frames
    ref_stars = detect_stars(reference_frame, n_stars=num_stars)
    frame_stars = detect_stars(frame, n_stars=num_stars)
    
    # Initialize transformation dict
    transformation = {
        'shift_x': 0.0,
        'shift_y': 0.0,
        'rotation': 0.0,
        'success': False
    }
    
    # Check if enough stars were detected
    if ref_stars is None or frame_stars is None or len(ref_stars) < 5 or len(frame_stars) < 5:
        return transformation
    
    # Extract star coordinates
    ref_coords = np.array([(s['x'], s['y']) for s in ref_stars])
    frame_coords = np.array([(s['x'], s['y']) for s in frame_stars])
    
    # Match stars between frames using a triangle-based approach
    matched_pairs = match_star_patterns(ref_coords, frame_coords)
    
    # If enough matches found, calculate transformation
    if len(matched_pairs) >= 3:
        # Extract matched coordinates
        ref_matched = np.array([ref_coords[idx] for idx, _ in matched_pairs])
        frame_matched = np.array([frame_coords[idx] for _, idx in matched_pairs])
        
        # Calculate affine transformation matrix
        transformation_matrix = calculate_affine_transform(frame_matched, ref_matched)
        
        if transformation_matrix is not None:
            # Extract rotation and translation from matrix
            rotation, shift_x, shift_y = extract_transform_parameters(transformation_matrix)
            
            transformation['shift_x'] = shift_x
            transformation['shift_y'] = shift_y
            transformation['rotation'] = rotation
            transformation['success'] = True
            transformation['num_matched_stars'] = len(matched_pairs)
    
    return transformation

def detect_stars(frame, n_stars=30):
    """
    Detect stars in frame using DAOStarFinder
    
    Returns:
    --------
    stars : list of dict
        List of detected stars with x, y coordinates and flux
    """
    try:
        # Calculate background statistics
        mean, median, std = sigma_clipped_stats(frame, sigma=3.0)
        
        # Create star finder
        daofind = DAOStarFinder(
            fwhm=3.0,  # Adjust based on your image characteristics
            threshold=5.0 * std,
            sharplo=0.2,
            sharphi=1.0,
            roundlo=-0.5,
            roundhi=0.5
        )
        
        # Find stars
        sources = daofind(frame - median)
        
        if sources is None or len(sources) == 0:
            return None
            
        # Convert to list of dictionaries for easier handling
        stars = []
        for i in range(len(sources)):
            stars.append({
                'x': float(sources['xcentroid'][i]),
                'y': float(sources['ycentroid'][i]),
                'flux': float(sources['flux'][i])
            })
        
        # Sort by flux (brightest first) and take the N brightest
        stars.sort(key=lambda s: s['flux'], reverse=True)
        return stars[:n_stars]
        
    except Exception as e:
        print(f"Star detection error: {e}")
        return None

def match_star_patterns(ref_coords, frame_coords, max_matches=10):
    """
    Match stars between frames using geometric pattern matching
    
    This uses triangle similarity to establish robust correspondences
    between stars in the two frames, even with rotation.
    """
    # Need at least 3 stars in each frame to form triangles
    if len(ref_coords) < 3 or len(frame_coords) < 3:
        return []
    
    # Function to calculate triangle features (3 sides and area)
    def triangle_features(points, idx1, idx2, idx3):
        p1, p2, p3 = points[idx1], points[idx2], points[idx3]
        
        # Calculate side lengths
        side1 = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        side2 = np.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
        side3 = np.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
        
        # Sort sides for invariance to vertex ordering
        sides = sorted([side1, side2, side3])
        
        # Calculate area using Heron's formula
        s = (side1 + side2 + side3) / 2
        area = np.sqrt(s * (s - side1) * (s - side2) * (s - side3))
        
        # Features: normalized sides and log(area) for scale invariance
        if side1 > 0:
            features = [
                sides[1] / sides[0],
                sides[2] / sides[0],
                np.log10(area + 1e-10)
            ]
        else:
            features = [1.0, 1.0, 0.0]  # Default for degenerate triangle
            
        return features
    
    # Generate triangles from reference stars (using brightest stars)
    ref_triangles = []
    ref_indices = []
    
    # Limit number of triangles to avoid combinatorial explosion
    max_stars = min(10, len(ref_coords))
    
    for i in range(max_stars):
        for j in range(i+1, max_stars):
            for k in range(j+1, max_stars):
                features = triangle_features(ref_coords, i, j, k)
                ref_triangles.append(features)
                ref_indices.append((i, j, k))
    
    # Generate triangles from frame stars
    frame_triangles = []
    frame_indices = []
    
    max_stars = min(10, len(frame_coords))
    
    for i in range(max_stars):
        for j in range(i+1, max_stars):
            for k in range(j+1, max_stars):
                features = triangle_features(frame_coords, i, j, k)
                frame_triangles.append(features)
                frame_indices.append((i, j, k))
    
    # Find matching triangles
    matched_triangles = []
    
    for i, ref_feat in enumerate(ref_triangles):
        for j, frame_feat in enumerate(frame_triangles):
            # Calculate feature distance
            dist = np.sqrt(
                (ref_feat[0] - frame_feat[0])**2 +
                (ref_feat[1] - frame_feat[1])**2 +
                (5.0 * (ref_feat[2] - frame_feat[2]))**2  # Weight area difference more
            )
            
            # If features are close, consider it a match
            if dist < 0.1:  # Threshold for triangle similarity
                matched_triangles.append((i, j, dist))
    
    # Sort matches by feature distance
    matched_triangles.sort(key=lambda x: x[2])
    
    # Extract star correspondences from triangle matches
    star_votes = {}
    
    for ref_idx, frame_idx, _ in matched_triangles[:20]:  # Use top 20 triangle matches
        ref_tri = ref_indices[ref_idx]
        frame_tri = frame_indices[frame_idx]
        
        # Each triangle gives 3 potential star matches
        for r, f in zip(ref_tri, frame_tri):
            pair = (r, f)
            star_votes[pair] = star_votes.get(pair, 0) + 1
    
    # Select pairs with most votes
    pairs = sorted(star_votes.items(), key=lambda x: x[1], reverse=True)
    
    # Create final list of matches, ensuring each star is used only once
    matched_pairs = []
    used_ref = set()
    used_frame = set()
    
    for (r, f), votes in pairs:
        if votes < 2:  # Require at least 2 votes for reliability
            continue
            
        if r not in used_ref and f not in used_frame:
            matched_pairs.append((r, f))
            used_ref.add(r)
            used_frame.add(f)
            
        if len(matched_pairs) >= max_matches:
            break
    
    return matched_pairs

def calculate_affine_transform(src_points, dst_points):
    """
    Calculate affine transformation matrix from matched points
    """
    if len(src_points) < 3 or len(dst_points) < 3:
        return None
        
    # Use OpenCV's estimateAffinePartial2D to allow only rotation and translation
    # This is more stable than full affine which can introduce scaling and shearing
    try:
        matrix, inliers = cv2.estimateAffinePartial2D(
            src_points, 
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99,
            maxIters=2000
        )
        
        # Check if enough inliers
        if matrix is not None and np.sum(inliers) >= 3:
            return matrix
        else:
            return None
    except Exception as e:
        print(f"Affine calculation error: {e}")
        return None

def extract_transform_parameters(matrix):
    """
    Extract rotation angle and translation from affine matrix
    """
    # Extract rotation
    rotation_rad = np.arctan2(matrix[1, 0], matrix[0, 0])
    rotation_deg = np.degrees(rotation_rad)
    
    # Extract translation
    shift_x = matrix[0, 2]
    shift_y = matrix[1, 2]
    
    return rotation_deg, shift_x, shift_y

def rotate_and_shift_frame(frame, angle, shift_y, shift_x):
    """
    Apply rotation and translation to frame
    """
    # Rotate first
    rotated = rotate(frame, angle, preserve_range=True, order=3)
    
    # Then shift
    aligned = ndimage_shift(rotated, (shift_y, shift_x), order=3, mode='constant', cval=0)
    
    return aligned

def detect_rotation_angle(frame, reference, max_angle=1.0, angle_steps=20):
    """
    Detect rotation angle between frames using polar transformation and phase correlation
    
    Parameters:
    -----------
    frame : ndarray
        Input frame
    reference : ndarray
        Reference frame
    max_angle : float
        Maximum rotation to check in degrees
    angle_steps : int
        Number of angle steps to test
        
    Returns:
    --------
    float
        Detected rotation angle in degrees
    """
    # Define angle search space
    angles = np.linspace(-max_angle, max_angle, angle_steps)
    
    # Initialize variables to track best correlation
    best_corr = -1
    best_angle = 0
    
    # Try different rotation angles
    for angle in angles:
        # Rotate frame
        frame_rotated = rotate(frame, angle, preserve_range=True)
        
        # Calculate correlation
        corr = calculate_ncc(reference, frame_rotated)
        
        # Update best if improved
        if corr > best_corr:
            best_corr = corr
            best_angle = angle
    
    # Fine-tune angle with smaller steps around best angle
    if abs(best_angle) < max_angle - 0.1:  # Not at the edge of search space
        fine_angles = np.linspace(best_angle - 0.1, best_angle + 0.1, 10)
        
        for angle in fine_angles:
            # Rotate frame
            frame_rotated = rotate(frame, angle, preserve_range=True)
            
            # Calculate correlation
            corr = calculate_ncc(reference, frame_rotated)
            
            # Update best if improved
            if corr > best_corr:
                best_corr = corr
                best_angle = angle
    
    return best_angle

def calculate_ncc(img1, img2):
    """
    Calculate normalized cross-correlation between two images
    """
    # Normalize images
    img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-10)
    img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-10)
    
    # Calculate correlation
    return np.mean(img1_norm * img2_norm)

def calculate_alignment_quality(aligned_frame, reference_frame):
    """
    Calculate various metrics to assess alignment quality
    
    Returns:
    --------
    dict
        Dictionary with quality metrics
    """
    # Normalized Cross-Correlation (NCC)
    ncc = calculate_ncc(aligned_frame, reference_frame)
    
    # Mean Squared Error (MSE)
    # Normalize frames to 0-1 range first
    aligned_norm = (aligned_frame - np.min(aligned_frame)) / (np.max(aligned_frame) - np.min(aligned_frame) + 1e-10)
    ref_norm = (reference_frame - np.min(reference_frame)) / (np.max(reference_frame) - np.min(reference_frame) + 1e-10)
    mse = np.mean((aligned_norm - ref_norm) ** 2)
    
    # Structural Similarity Index (SSIM)
    try:
        from skimage.metrics import structural_similarity
        ssim = structural_similarity(aligned_norm, ref_norm)
    except:
        ssim = 0.0
    
    return {
        'ncc': ncc,
        'mse': mse,
        'ssim': ssim
    }

def visualize_alignment(reference, frame, aligned, stars_ref=None, stars_frame=None, matched_pairs=None):
    """
    Visualize alignment results for debugging
    """
    plt.figure(figsize=(15, 10))
    
    # Plot reference frame
    plt.subplot(2, 2, 1)
    plt.imshow(reference, cmap='viridis')
    plt.title('Reference Frame')
    
    # Plot original frame
    plt.subplot(2, 2, 2)
    plt.imshow(frame, cmap='viridis')
    plt.title('Original Frame')
    
    # Plot aligned frame
    plt.subplot(2, 2, 3)
    plt.imshow(aligned, cmap='viridis')
    plt.title('Aligned Frame')
    
    # Plot difference
    plt.subplot(2, 2, 4)
    diff = reference - aligned
    plt.imshow(diff, cmap='RdBu_r')
    plt.title('Difference (Reference - Aligned)')
    
    plt.tight_layout()
    plt.savefig('alignment_visualization.png')
    plt.close()
    
    # If star matches are provided, visualize them
    if stars_ref is not None and stars_frame is not None and matched_pairs is not None:
        plt.figure(figsize=(10, 5))
        
        # Extract coordinates
        ref_coords = np.array([(s['x'], s['y']) for s in stars_ref])
        frame_coords = np.array([(s['x'], s['y']) for s in stars_frame])
        
        # Plot reference frame with stars
        plt.subplot(1, 2, 1)
        plt.imshow(reference, cmap='viridis')
        plt.scatter(ref_coords[:, 0], ref_coords[:, 1], c='red', s=50, alpha=0.7, marker='+')
        
        # Highlight matched stars
        for r, _ in matched_pairs:
            plt.scatter(ref_coords[r, 0], ref_coords[r, 1], c='yellow', s=100, alpha=0.7, facecolors='none')
        
        plt.title('Reference Frame Stars')
        
        # Plot original frame with stars
        plt.subplot(1, 2, 2)
        plt.imshow(frame, cmap='viridis')
        plt.scatter(frame_coords[:, 0], frame_coords[:, 1], c='red', s=50, alpha=0.7, marker='+')
        
        # Highlight matched stars
        for _, f in matched_pairs:
            plt.scatter(frame_coords[f, 0], frame_coords[f, 1], c='yellow', s=100, alpha=0.7, facecolors='none')
        
        plt.title('Frame Stars')
        
        plt.tight_layout()
        plt.savefig('star_matching.png')
        plt.close()

def multi_method_registration(frame, reference_frame, methods=['enhanced', 'star', 'fft']):
    """
    Try multiple registration methods and select the best result
    """
    results = []
    
    for method in methods:
        if method == 'enhanced':
            aligned, transform = enhanced_registration(frame, reference_frame)
        elif method == 'star':
            # Normalize frames
            frame_norm = normalize_frame(frame)
            ref_norm = normalize_frame(reference_frame)
            
            # Detect stars
            ref_stars = detect_stars(ref_norm)
            frame_stars = detect_stars(frame_norm)
            
            # Match stars
            if ref_stars and frame_stars:
                ref_coords = np.array([(s['x'], s['y']) for s in ref_stars])
                frame_coords = np.array([(s['x'], s['y']) for s in frame_stars])
                matched_pairs = match_star_patterns(ref_coords, frame_coords)
                
                # Calculate transformation
                if len(matched_pairs) >= 3:
                    ref_matched = np.array([ref_coords[idx] for idx, _ in matched_pairs])
                    frame_matched = np.array([frame_coords[idx] for _, idx in matched_pairs])
                    
                    matrix = calculate_affine_transform(frame_matched, ref_matched)
                    if matrix is not None:
                        rotation, shift_x, shift_y = extract_transform_parameters(matrix)
                        aligned = rotate_and_shift_frame(frame, rotation, shift_y, shift_x)
                        
                        quality = calculate_alignment_quality(aligned, reference_frame)
                        transform = {
                            'method': 'star',
                            'shift_x': shift_x,
                            'shift_y': shift_y,
                            'rotation': rotation,
                            'success': True,
                            'quality': quality
                        }
                    else:
                        # Fallback to no transformation
                        aligned = frame.copy()
                        transform = {
                            'method': 'star',
                            'success': False,
                            'quality': {'ncc': 0, 'mse': float('inf'), 'ssim': 0}
                        }
                else:
                    # Not enough matches
                    aligned = frame.copy()
                    transform = {
                        'method': 'star',
                        'success': False,
                        'quality': {'ncc': 0, 'mse': float('inf'), 'ssim': 0}
                    }
            else:
                # Star detection failed
                aligned = frame.copy()
                transform = {
                    'method': 'star',
                    'success': False,
                    'quality': {'ncc': 0, 'mse': float('inf'), 'ssim': 0}
                }
        elif method == 'fft':
            # Normalize frames
            frame_norm = normalize_frame(frame)
            ref_norm = normalize_frame(reference_frame)
            
            # Use phase correlation for translation
            shift, error, diffphase = phase_cross_correlation(
                ref_norm, 
                frame_norm,
                upsample_factor=100
            )
            
            aligned = ndimage_shift(frame, shift, order=3, mode='constant', cval=0)
            quality = calculate_alignment_quality(aligned, reference_frame)
            
            transform = {
                'method': 'fft',
                'shift_y': float(shift[0]),
                'shift_x': float(shift[1]),
                'rotation': 0.0,
                'success': True,
                'quality': quality
            }
        
        # Add to results
        results.append((aligned, transform))
    
    # Select best result based on quality
    best_ncc = -1
    best_result = None
    
    for aligned, transform in results:
        if transform['success']:
            ncc = transform['quality']['ncc']
            if ncc > best_ncc:
                best_ncc = ncc
                best_result = (aligned, transform)
    
    # Return best or original if all failed
    if best_result:
        return best_result
    else:
        return frame, {'method': 'none', 'success': False}

# Example usage
if __name__ == "__main__":
    import astropy.io.fits as fits
    
    # Load frames
    ref_path = "reference_frame.fits"
    frame_path = "frame_to_align.fits"
    
    reference_frame = fits.getdata(ref_path)
    frame = fits.getdata(frame_path)
    
    # Register frame
    aligned_frame, transformation = enhanced_registration(frame, reference_frame, detect_rotation=True)
    
    print(f"Alignment method: {transformation['method']}")
    print(f"Shift: ({transformation['shift_x']:.2f}, {transformation['shift_y']:.2f}) pixels")
    print(f"Rotation: {transformation['rotation']:.3f} degrees")
    print(f"Quality: NCC={transformation['quality']['ncc']:.3f}, SSIM={transformation['quality']['ssim']:.3f}")
    
    # Visualize result
    visualize_alignment(normalize_frame(reference_frame), normalize_frame(frame), normalize_frame(aligned_frame))
