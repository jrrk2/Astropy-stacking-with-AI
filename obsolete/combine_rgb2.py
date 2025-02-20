from astropy.io import fits
import numpy as np

# Read the stacked channel files
r_data = fits.getdata('stacked_r.fits')
g_data = fits.getdata('stacked_g.fits')
b_data = fits.getdata('stacked_b.fits')

# Normalize each channel separately before combining
def normalize_channel(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return ((data - data_min) / (data_max - data_min) * 65535).astype(np.uint16)

r_norm = normalize_channel(r_data)
g_norm = normalize_channel(g_data)
b_norm = normalize_channel(b_data)

# Stack with correct axis order and balanced weights
rgb = np.stack([r_norm, g_norm, b_norm], axis=0)

# Save result
hdu = fits.PrimaryHDU(rgb)
hdu.header['COLORIMG'] = True
hdu.writeto('final_rgb_balanced.fits', overwrite=True)
