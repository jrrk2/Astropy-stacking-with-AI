from astropy.io import fits
import numpy as np

# Function to safely load and check a FITS file
def check_fits(filename):
    with fits.open(filename) as hdul:
        data = hdul[0].data
        print(f"\n{filename}:")
        print("Shape:", data.shape)
        print("Data type:", data.dtype)
        print("Number of NaN values:", np.sum(np.isnan(data)))
        print("Number of Inf values:", np.sum(np.isinf(data)))
        if not np.all(np.isnan(data)):  # If not all values are NaN
            print("Min (excluding NaN):", np.nanmin(data))
            print("Max (excluding NaN):", np.nanmax(data))
            print("Mean (excluding NaN):", np.nanmean(data))
        return data

# Check each stack
r = check_fits('stacked_r.fits')
g = check_fits('stacked_g.fits')
b = check_fits('stacked_b.fits')
