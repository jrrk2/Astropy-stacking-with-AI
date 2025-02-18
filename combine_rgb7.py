from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clipped_stats
from PIL import Image

# [previous code remains the same until the stretching]

def stretch_and_save(rgb_data, fits_name, png_name):
    # Stretch the data
    def stretch_to_match_reference(data, black_point=0.1, white_point=0.995):
        percentiles = np.percentile(data, [black_point*100, white_point*100])
        normalized = (data - percentiles[0]) / (percentiles[1] - percentiles[0])
        normalized = np.clip(normalized, 0, 1)
        normalized = normalized ** 0.85
        return normalized

    rgb_stretched = stretch_to_match_reference(rgb_data)
    rgb_16bit = (rgb_stretched * 65535).astype(np.uint16)
    
    # Save FITS
    hdu = fits.PrimaryHDU(rgb_16bit)
    hdu.header['COLORIMG'] = True
    for channel, weight in weights.items():
        hdu.header[f'WEIGHT_{channel}'] = weight
    hdu.writeto(fits_name, overwrite=True)
    
    # Save 16-bit PNG
    # Transpose from (3,H,W) to (H,W,3) for image format
    rgb_png = np.transpose(rgb_16bit, (1, 2, 0))
    # Create PIL Image in 16-bit RGB mode
    img = Image.fromarray(rgb_png, mode='RGB;16')
    img.save(png_name, format='PNG', optimize=False)
    
    print(f"Saved {fits_name} and {png_name} in 16-bit format")

# Save both formats
stretch_and_save(rgb, 'final_rgb_matched.fits', 'final_rgb_matched_16bit.png')
