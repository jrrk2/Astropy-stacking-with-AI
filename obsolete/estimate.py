from astropy.io import fits
import numpy as np
from scipy import stats

# Read the individual stacks
r = fits.getdata('stacked_r.fits')
g = fits.getdata('stacked_g.fits')
b = fits.getdata('stacked_b.fits')

# Function to estimate background level more aggressively
def get_background_level(data, label=''):
    valid_data = data[~np.isnan(data)]
    # Try multiple methods to estimate background
    p1 = np.percentile(valid_data, 1)  # 1st percentile
    p5 = np.percentile(valid_data, 5)  # 5th percentile
    # Use histogram to find mode of lower values
    hist, bin_edges = np.histogram(valid_data, bins=100)
    mode_bin = bin_edges[np.argmax(hist)]
    
    print(f"{label} background estimates:")
    print(f"  1st percentile: {p1:.2f}")
    print(f"  5th percentile: {p5:.2f}")
    print(f"  Mode estimate: {mode_bin:.2f}")
    
    # Use the mode estimate as it should better represent the true background
    return mode_bin

# Get and apply background levels
r_bg = get_background_level(r, 'Red')
g_bg = get_background_level(g, 'Green')
b_bg = get_background_level(b, 'Blue')

# Fill NaN values with background
r[np.isnan(r)] = r_bg
g[np.isnan(g)] = g_bg
b[np.isnan(b)] = b_bg

# Normalize each channel with more aggressive scaling
def normalize_channel(data, bg_level):
    # Subtract background
    data_sub = data - bg_level
    # Scale using 99th percentile instead of max to avoid hot pixels
    scale_max = np.percentile(data_sub, 99)
    return np.clip(data_sub / scale_max, 0, 1)

r_norm = normalize_channel(r, r_bg)
g_norm = normalize_channel(g, g_bg)
b_norm = normalize_channel(b, b_bg)

# Combine into RGB
rgb = np.zeros((2080, 3072, 3))
rgb[:,:,0] = r_norm
rgb[:,:,1] = g_norm
rgb[:,:,2] = b_norm

# Save as a new FITS file with RGB data
hdu = fits.PrimaryHDU(rgb)
hdu.writeto('merged_rgb_normalized.fits', overwrite=True)
