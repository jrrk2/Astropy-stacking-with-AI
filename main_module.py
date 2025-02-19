import numpy as np
from astropy.wcs import WCS
from arguments_module import parse_arguments
from astropy.io import fits
from astrometry_module import solve_astrometry
from utils_module import find_valid_region
from utils_module import perform_photometric_calibration
from image_processing_module import custom_stretch
from image_processing_module import save_16bit_png
from image_processing_module import eliminate_background_wavelet
from catalog_module import get_catalog_stars

def main():
    args = parse_arguments()
    
    # Read the individual stacks with WCS information
    r_hdu = fits.open('stacked_r.fits')[0]
    g_hdu = fits.open('stacked_g.fits')[0]
    b_hdu = fits.open('stacked_b.fits')[0]
    
    r = r_hdu.data
    g = g_hdu.data
    b = b_hdu.data
    
    # Run astrometry.net on the green channel
    if solve_astrometry('stacked_g.fits'):
        # Re-read the solved file
        g_hdu = fits.open('stacked_g.fits')[0]
        wcs = WCS(g_hdu.header)
        print("Using new astrometric solution from solve-field")
    else:
        print("Warning: Astrometric solution failed, using original WCS")
        wcs = WCS(g_hdu.header)
    
    print("\nAnalyzing input images...")
    try:
        y_start, y_end, x_start, x_end = find_valid_region([r, g, b])
        print("\nCropping images to valid region...")
        
        # Crop all channels to valid region
        r = r[y_start:y_end, x_start:x_end]
        g = g[y_start:y_end, x_start:x_end]
        b = b[y_start:y_end, x_start:x_end]
        
        # Update WCS for cropped region
        wcs = wcs.slice((slice(y_start, y_end), slice(x_start, x_end)))
    except ValueError as e:
        print(f"\nError: {str(e)}")
        print("Attempting to proceed with uncropped images...")
        # Fill NaN values with zeros instead
        r[np.isnan(r)] = 0
        g[np.isnan(g)] = 0
        b[np.isnan(b)] = 0
    
    # Stretch each channel
    r_stretched = custom_stretch(r)
    g_stretched = custom_stretch(g)
    b_stretched = custom_stretch(b)
    
    # Apply background gradient elimination if specified
    if args.method == 'wavelet':
        r_stretched = eliminate_background_wavelet(r_stretched)
        g_stretched = eliminate_background_wavelet(g_stretched)
        b_stretched = eliminate_background_wavelet(b_stretched)
    
    # Perform photometric calibration using catalog stars
    try:
        print(f"Querying {args.catalog} catalog for reference stars...")
        reference_stars = get_catalog_stars(wcs, catalog=args.catalog, mag_limit=args.mag_limit)
        print("First few star entries:")
        for star in reference_stars[:5]:
            print(star)
        
        if reference_stars is None or len(reference_stars) == 0:
            print(f"No suitable reference stars found in {args.catalog} catalog")
        else:
            print(f"Found {len(reference_stars)} reference stars")
            
            # Calibrate each channel
            r_calibrated, r_zp = perform_photometric_calibration(r_stretched, wcs, reference_stars)
            g_calibrated, g_zp = perform_photometric_calibration(g_stretched, wcs, reference_stars)
            b_calibrated, b_zp = perform_photometric_calibration(b_stretched, wcs, reference_stars)
            
            if all(zp is not None for zp in [r_zp, g_zp, b_zp]):
                print(f"Zero points - R: {r_zp:.2f}, G: {g_zp:.2f}, B: {b_zp:.2f}")
                r_stretched, g_stretched, b_stretched = r_calibrated, g_calibrated, b_calibrated
            else:
                print("Warning: Photometric calibration failed for one or more channels")
    except Exception as e:
        print(f"Error during photometric calibration: {str(e)}")
    
    # Stack stretched channels
    rgb_stretched = np.stack((r_stretched, g_stretched, b_stretched), axis=0)
    
    # Save outputs with updated WCS
    new_header = r_hdu.header.copy()
    new_header.update(wcs.to_header())
    
    # Update NAXIS1/NAXIS2 to match cropped size
    new_header['NAXIS1'] = rgb_stretched.shape[2]
    new_header['NAXIS2'] = rgb_stretched.shape[1]
    
    # Save FITS with updated header
    hdu = fits.PrimaryHDU(rgb_stretched, header=new_header)
    hdu.writeto('merged_rgb_stretched.fits', overwrite=True)
    
    # Save PNG
    save_16bit_png(rgb_stretched, "merged_rgb_stretched16.png")
    
    print("Image processing complete.")

if __name__ == "__main__":
    main()
