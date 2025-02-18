from astropy.io import fits
import numpy as np

# Read the stacked channel files
r_data = fits.getdata('stacked_r.fits')
g_data = fits.getdata('stacked_g.fits')
b_data = fits.getdata('stacked_b.fits')

# Stack with correct axis order
rgb = np.stack([r_data, g_data, b_data], axis=0)

# Normalize to 16-bit
rgb_min = np.min(rgb)
rgb_max = np.max(rgb)
rgb_norm = ((rgb - rgb_min) / (rgb_max - rgb_min) * 65535).astype(np.uint16)

# Save result
hdu = fits.PrimaryHDU(rgb_norm)
hdu.header['COLORIMG'] = True
hdu.writeto('final_rgb.fits', overwrite=True)
