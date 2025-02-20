from astropy.io import fits
import numpy as np
import cv2
import argparse
from scipy import interpolate

def parse_arguments():
    """Parse command-line arguments for background gradient elimination method."""
    parser = argparse.ArgumentParser(description='Image Processing with Background Gradient Elimination')
    parser.add_argument('--method', type=str, default='none', 
                        choices=['none', 'median', 'polynomial', 'local', 'wavelet'],
                        help='Background gradient elimination method')
    return parser.parse_args()

# Read the individual stacks
r = fits.getdata('stacked_r.fits')
g = fits.getdata('stacked_g.fits')
b = fits.getdata('stacked_b.fits')

# Function to estimate background level
def get_background_level(data):
    valid_data = data[~np.isnan(data)]
    # Use the 10th percentile as an estimate of the background
    background = np.percentile(valid_data, 10)
    print(f"Background level: {background:.2f}")
    return background

# Fill NaN values with background level for each channel
r_bg = get_background_level(r)
g_bg = get_background_level(g)
b_bg = get_background_level(b)

r[np.isnan(r)] = r_bg
g[np.isnan(g)] = g_bg
b[np.isnan(b)] = b_bg

# Function to apply custom stretch with careful histogram matching
def custom_stretch(image, black_point=0.1, white_point=0.999, gamma=0.5):
    """
    Apply a custom stretch with histogram equalization
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image array
    black_point : float, optional
        Percentile to use as black point (default: 0.1)
    white_point : float, optional
        Percentile to use as white point (default: 0.999)
    gamma : float, optional
        Gamma correction factor (default: 0.5)
    
    Returns:
    --------
    numpy.ndarray
        Stretched image
    """
    # Remove extreme outliers
    valid_data = image[~np.isnan(image)]
    
    # Calculate black and white points
    black = np.percentile(valid_data, black_point * 100)
    white = np.percentile(valid_data, white_point * 100)
    
    # Clip and normalize
    stretched = np.clip((image - black) / (white - black), 0, 1)
    
    # Apply gamma correction
    stretched = np.power(stretched, gamma)
    
    return stretched

# Background gradient elimination methods
def eliminate_background_median(image, kernel_size=101):
    """
    Remove background using median filtering
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    kernel_size : int, optional
        Size of the median filter kernel (default: 101)
    
    Returns:
    --------
    numpy.ndarray
        Background-subtracted image
    """
    # Ensure kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Apply median filter
    background = cv2.medianBlur(image.astype(np.float32), kernel_size)
    
    # Subtract background
    return image - background + np.median(background)

def eliminate_background_polynomial(image, degree=2):
    """
    Remove background using polynomial surface fitting
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    degree : int, optional
        Degree of polynomial surface (default: 2)
    
    Returns:
    --------
    numpy.ndarray
        Background-subtracted image
    """
    # Create coordinate grid
    height, width = image.shape
    y, x = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    
    # Flatten the arrays
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = image.ravel()
    
    # Prepare polynomial basis
    A = np.zeros((x_flat.size, (degree+1)**2))
    for i in range(degree+1):
        for j in range(degree+1):
            A[:, i*(degree+1) + j] = (x_flat**i) * (y_flat**j)
    
    # Solve for polynomial coefficients
    coeffs = np.linalg.lstsq(A, z_flat, rcond=None)[0]
    
    # Reconstruct background surface
    background = np.zeros_like(image)
    for i in range(degree+1):
        for j in range(degree+1):
            background += coeffs[i*(degree+1) + j] * (x**i) * (y**j)
    
    # Subtract background
    return image - background + np.median(background)

def eliminate_background_local(image, block_size=101):
    """
    Remove background using local background equalization
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    block_size : int, optional
        Size of local background estimation block (default: 101)
    
    Returns:
    --------
    numpy.ndarray
        Background-subtracted image
    """
    # Ensure block size is odd
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    
    # Apply local background estimation
    background = cv2.blur(image.astype(np.float32), (block_size, block_size))
    
    # Subtract background
    return image - background + np.median(background)

def eliminate_background_wavelet(image, levels=3):
    """
    Remove background using wavelet transform
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    levels : int, optional
        Number of wavelet decomposition levels (default: 3)
    
    Returns:
    --------
    numpy.ndarray
        Background-subtracted image
    """
    import pywt
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(image, 'db4', level=levels)
    
    # Modify approximation coefficients (lowest frequency)
    coeffs[0] = np.zeros_like(coeffs[0])
    
    # Reconstruct image
    background = pywt.waverec2(coeffs, 'db4')
    
    # Ensure background has same shape as original
    background = background[:image.shape[0], :image.shape[1]]
    
    # Subtract background
    return image - background + np.median(background)

# Stretch each channel
r_stretched = custom_stretch(r)
g_stretched = custom_stretch(g)
b_stretched = custom_stretch(b)

# Parse arguments
args = parse_arguments()

# Apply background gradient elimination if specified
if args.method == 'median':
    r_stretched = eliminate_background_median(r_stretched)
    g_stretched = eliminate_background_median(g_stretched)
    b_stretched = eliminate_background_median(b_stretched)
elif args.method == 'polynomial':
    r_stretched = eliminate_background_polynomial(r_stretched)
    g_stretched = eliminate_background_polynomial(g_stretched)
    b_stretched = eliminate_background_polynomial(b_stretched)
elif args.method == 'local':
    r_stretched = eliminate_background_local(r_stretched)
    g_stretched = eliminate_background_local(g_stretched)
    b_stretched = eliminate_background_local(b_stretched)
elif args.method == 'wavelet':
    r_stretched = eliminate_background_wavelet(r_stretched)
    g_stretched = eliminate_background_wavelet(g_stretched)
    b_stretched = eliminate_background_wavelet(b_stretched)

# Stack stretched channels
rgb_stretched = np.stack((r_stretched, g_stretched, b_stretched), axis=0)

# Save as FITS
hdu = fits.PrimaryHDU(rgb_stretched)
hdu.writeto('merged_rgb_stretched.fits', overwrite=True)

# Improved PNG saving method with border trimming
def save_16bit_png(data, filename, trim_percent=0.05):
    """
    Save a normalized 16-bit PNG with border trimming
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input image data (channels, height, width)
    filename : str
        Output filename
    trim_percent : float, optional
        Percentage of border to trim (default: 0.05 = 5%)
    """
    # Transpose to (height, width, channels)
    data_transposed = data.transpose(1, 2, 0)
    
    # Calculate trim sizes
    height, width, _ = data_transposed.shape
    trim_height = int(height * trim_percent)
    trim_width = int(width * trim_percent)
    
    # Trim the border
    trimmed_data = data_transposed[
        trim_height:height-trim_height, 
        trim_width:width-trim_width
    ]
    
    # Scale to 16-bit range
    scaled_data = (trimmed_data * 65535).astype(np.uint16)
    
    # Save with compression
    cv2.imwrite(filename, scaled_data, [
        cv2.IMWRITE_PNG_COMPRESSION, 9,  # Maximum compression
        cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT
    ])
    
    # Print file size
    import os
    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"PNG file size: {file_size:.2f} MB")

# Save 16-bit PNG
save_16bit_png(rgb_stretched, "merged_rgb_stretched16.png")

print("Image processing complete.")
print("Created: merged_rgb_stretched.fits")
print("Created: merged_rgb_stretched16.png")
print("Stretch Statistics:")
for channel, name in zip([r_stretched, g_stretched, b_stretched], ['R', 'G', 'B']):
    print(f"\n{name} Channel:")
    print(f"Mean: {channel.mean():.4f}")
    print(f"Median: {np.median(channel):.4f}")
    print(f"Min: {channel.min():.4f}")
    print(f"Max: {channel.max():.4f}")
