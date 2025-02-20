from astropy.io import fits
import numpy as np
from scipy import stats
from PIL import Image

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

# Stretch each channel
r_stretched = custom_stretch(r)
g_stretched = custom_stretch(g)
b_stretched = custom_stretch(b)

# Stack stretched channels
rgb_stretched = np.stack((r_stretched, g_stretched, b_stretched), axis=0)

# Save as FITS
hdu = fits.PrimaryHDU(rgb_stretched)
hdu.writeto('merged_rgb_stretched.fits', overwrite=True)

# Correct PNG saving method
def save_16bit_png(data, filename):
    """
    Save a normalized 16-bit PNG with proper scaling
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input image data (0-1 range)
    filename : str
        Output filename
    """
    # Scale to 16-bit range and convert
    scaled_data = (data * 65535).astype(np.uint16)
    
    # Reshape for PIL (width, height, channels)
    scaled_image = scaled_data.transpose(1, 2, 0)
    
    # Save using PIL
    im = Image.fromarray(scaled_image, mode='I;16')
    im.save(filename)

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
