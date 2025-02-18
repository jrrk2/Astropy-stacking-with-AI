from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from PIL import Image

# First analyze the reference image
ref_img = plt.imread('reference_m27.jpg')  # Or whatever format your reference image is

# Get color statistics from the nebula region (approximate center coordinates)
h, w = ref_img.shape[:2]
center_y, center_x = h//2, w//2
radius = min(h, w)//6  # Approximate nebula size

# Create a mask for the nebula region
y, x = np.ogrid[:h, :w]
mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2

# Calculate ratios from reference
ref_ratios = {
    'R': np.median(ref_img[mask, 0]),
    'G': np.median(ref_img[mask, 1]),
    'B': np.median(ref_img[mask, 2])
}

print("Reference image ratios:")
for channel, ratio in ref_ratios.items():
    print(f"{channel}: {ratio:.3f}")

# Now process the stacked channels
r_data = fits.getdata('stacked_r.fits')
g_data = fits.getdata('stacked_g.fits') * 0.5  # Compensate for double green pixels
b_data = fits.getdata('stacked_b.fits')

# Normalize each channel
def normalize_channel(data):
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    data = data - median  # Remove background
    return np.clip(data, 0, None)  # Clip negative values

r_norm = normalize_channel(r_data)
g_norm = normalize_channel(g_data)
b_norm = normalize_channel(b_data)

# Apply reference ratios
max_val = max(ref_ratios.values())
weights = {
    'R': ref_ratios['R'] / max_val,
    'G': ref_ratios['G'] / max_val,
    'B': ref_ratios['B'] / max_val
}

print("\nCalculated weights:")
for channel, weight in weights.items():
    print(f"{channel}: {weight:.3f}")

r_weighted = r_norm * weights['R']
g_weighted = g_norm * weights['G']
b_weighted = b_norm * weights['B']

# Stack and normalize
rgb = np.stack([r_weighted, g_weighted, b_weighted], axis=0)

# Apply final stretching to match reference
def stretch_to_match_reference(data, black_point=0.1, white_point=0.995):
    percentiles = np.percentile(data, [black_point*100, white_point*100])
    normalized = (data - percentiles[0]) / (percentiles[1] - percentiles[0])
    normalized = np.clip(normalized, 0, 1)
    # Apply slight gamma correction to match reference
    normalized = normalized ** 0.85
    return (normalized * 65535).astype(np.uint16)

rgb = stretch_to_match_reference(rgb)

# Save result
hdu = fits.PrimaryHDU(rgb)
hdu.header['COLORIMG'] = True
for channel, weight in weights.items():
    hdu.header[f'WEIGHT_{channel}'] = weight
hdu.writeto('final_rgb_matched.fits', overwrite=True)
