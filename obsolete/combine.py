from astropy.io import fits
import numpy as np

# Read the individual stacks
r = fits.getdata('stacked_r.fits')
g = fits.getdata('stacked_g.fits')
b = fits.getdata('stacked_b.fits')

# Print statistics for each channel
print("Red channel:", "min:", r.min(), "max:", r.max(), "mean:", r.mean())
print("Green channel:", "min:", g.min(), "max:", g.max(), "mean:", g.mean())
print("Blue channel:", "min:", b.min(), "max:", b.max(), "mean:", b.mean())

# Normalize each channel independently 
r_norm = (r - r.min()) / (r.max() - r.min())
g_norm = (g - g.min()) / (g.max() - g.min())
b_norm = (b - b.min()) / (b.max() - b.min())

# Stack with correct dimensions
rgb = np.stack((r_norm, g_norm, b_norm), axis=0)
rgb = np.transpose(rgb, (1, 2, 0))

# Save as a new FITS file with RGB data
hdu = fits.PrimaryHDU(rgb)
hdu.writeto('merged_rgb_normalized.fits', overwrite=True)
