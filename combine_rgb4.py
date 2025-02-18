from astropy.io import fits
import numpy as np

# Read the stacked channel files
r_data = fits.getdata('stacked_r.fits')
g_data = fits.getdata('stacked_g.fits')
b_data = fits.getdata('stacked_b.fits')

# Compensate for double green pixels in Bayer matrix
g_data = g_data * 0.5  # Halve the green channel

# Normalize each channel independently
def normalize_channel(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return ((data - data_min) / (data_max - data_min))

r_norm = normalize_channel(r_data)
g_norm = normalize_channel(g_data)  # Green already halved
b_norm = normalize_channel(b_data)

# Stack with correct axis order
rgb = np.stack([r_norm, g_norm, b_norm], axis=0)

# Final scaling to 16-bit
rgb = (rgb * 65535).astype(np.uint16)

# Save result
hdu = fits.PrimaryHDU(rgb)
hdu.header['COLORIMG'] = True
hdu.writeto('final_rgb_green_corrected.fits', overwrite=True)
