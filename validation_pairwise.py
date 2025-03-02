import numpy as np
import os
import glob
import sys
import numpy as np
from astropy.io import fits
from skimage.registration import phase_cross_correlation

def load_fits_image(filename):
    """Load the first channel of a FITS image."""
    with fits.open(filename) as hdul:
        data = hdul[0].data  # Extract image data
        if data.ndim == 3:
            return data[0, :, :]  # Use only the first channel
        return data

def compute_shift(image1, image2):
    shift, error, diffphase = phase_cross_correlation(image1, image2, upsample_factor=10)
    return shift

def main():
    """Process FITS files given as command-line arguments."""
    if len(sys.argv) < 3:
        print("Usage: python script.py file1.fits file2.fits [file3.fits ...]")
        return

    filenames = sys.argv[1:]
    images = [load_fits_image(f) for f in filenames]

    shifts = []
    for i in range(len(images) - 1):
        shift_y, shift_x = compute_shift(images[i], images[i + 1])
        shifts.append((filenames[i], filenames[i + 1], shift_y, shift_x))

    # Print results
    print("Computed Shifts:")
    for f1, f2, dy, dx in shifts:
        print(f"{f1} -> {f2}: Δy = {dy}, Δx = {dx}")

if __name__ == "__main__":
    main()
