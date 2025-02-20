from astropy.io import fits
import numpy as np
from scipy import stats

# Read the individual stacks
r = fits.getdata('stacked_r.fits')
g = fits.getdata('stacked_g.fits')
b = fits.getdata('stacked_b.fits')

# Function to estimate background level
def get_background_level(data):
    valid_data = data[~np.isnan(data)]
    # Use the mode as an estimate of the background
    # Or could use a low percentile like 10th
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

# Normalize each channel
r_norm = np.clip((r - r.min()) / (r.max() - r.min()), 0, 1)
g_norm = np.clip((g - g.min()) / (g.max() - g.min()), 0, 1)
b_norm = np.clip((b - b.min()) / (b.max() - b.min()), 0, 1)

# Stack with correct dimensions
rgb = np.stack((r_norm, g_norm, b_norm), axis=0)
# rgb = np.transpose(rgb, (1, 2, 0))

# Save as a new FITS file with RGB data
hdu = fits.PrimaryHDU(rgb)
hdu.writeto('merged_rgb_normalized.fits', overwrite=True)
