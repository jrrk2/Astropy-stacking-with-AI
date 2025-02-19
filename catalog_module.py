import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

def get_catalog_stars(wcs, catalog='apass', mag_limit=16.0):
    """
    Query online star catalog for reference stars in the image field
    """
    # Get image corners
    naxis1, naxis2 = wcs.pixel_shape
    corners = np.array([[0, 0], [0, naxis2], [naxis1, 0], [naxis1, naxis2]])
    ra_dec_corners = wcs.all_pix2world(corners, 0)
    
    # Calculate field center and radius
    center_ra = np.mean(ra_dec_corners[:, 0])
    center_dec = np.mean(ra_dec_corners[:, 1])
    radius = np.max(np.sqrt(
        (ra_dec_corners[:, 0] - center_ra)**2 +
        (ra_dec_corners[:, 1] - center_dec)**2
    )) * u.deg
    
    center = SkyCoord(center_ra, center_dec, unit=(u.deg, u.deg))
    
    # Configure Vizier
    vizier = Vizier(column_filters={"Vmag": f"<{mag_limit}"})
    vizier.ROW_LIMIT = -1  # No row limit
    
    if catalog == 'apass':
        # Query APASS DR9
        catalog_query = vizier.query_region(
            center, 
            radius=radius,
            catalog="II/336/apass9"  # APASS DR9
        )
        
        if len(catalog_query) > 0 and len(catalog_query[0]) > 0:
            stars = []
            for row in catalog_query[0]:
                stars.append({
                    'ra': float(row['RAJ2000']),
                    'dec': float(row['DEJ2000']),
                    'magnitude_B': float(row['Bmag']),
                    'magnitude_V': float(row['Vmag']),
                    'magnitude_g': float(row['g_mag']),
                    'magnitude_r': float(row['r_mag']),
                    'magnitude_i': float(row['i_mag'])
                })
            return stars
            
    elif catalog == 'sdss':
        # Query SDSS photometry
        catalog_query = vizier.query_region(
            center,
            radius=radius,
            catalog="V/147/sdss12"  # SDSS DR12
        )
        
        if len(catalog_query) > 0 and len(catalog_query[0]) > 0:
            stars = []
            for row in catalog_query[0]:
                stars.append({
                    'ra': float(row['RA_ICRS']),
                    'dec': float(row['DE_ICRS']),
                    'magnitude_u': float(row['umag']),
                    'magnitude_g': float(row['gmag']),
                    'magnitude_r': float(row['rmag']),
                    'magnitude_i': float(row['imag']),
                    'magnitude_z': float(row['zmag'])
                })
            return stars
            
    elif catalog == 'panstarrs':
        # Query Pan-STARRS DR1
        catalog_query = vizier.query_region(
            center,
            radius=radius,
            catalog="II/349/ps1"  # Pan-STARRS DR1
        )
        
        if len(catalog_query) > 0 and len(catalog_query[0]) > 0:
            stars = []
            for row in catalog_query[0]:
                stars.append({
                    'ra': float(row['RAJ2000']),
                    'dec': float(row['DEJ2000']),
                    'magnitude_g': float(row['gmag']),
                    'magnitude_r': float(row['rmag']),
                    'magnitude_i': float(row['imag']),
                    'magnitude_z': float(row['zmag']),
                    'magnitude_y': float(row['ymag'])
                })
            return stars
    
    return None

def perform_photometric_calibration(image, wcs, reference_stars):
    """
    Perform photometric calibration using reference stars
    """
    # Extract star positions and fluxes
    star_positions = []
    star_mags = []
    measured_fluxes = []
    
    for star in reference_stars:
        # Convert sky coordinates to pixel coordinates
        coord = SkyCoord(star['ra'], star['dec'], unit=(u.deg, u.deg))
        x, y = wcs.world_to_pixel(coord)
        
        # Extract flux in aperture
        aperture_radius = 5  # pixels
        y_int, x_int = int(round(y)), int(round(x))
        y_grid, x_grid = np.ogrid[-aperture_radius:aperture_radius+1, 
                                 -aperture_radius:aperture_radius+1]
        aperture_mask = x_grid*x_grid + y_grid*y_grid <= aperture_radius*aperture_radius
        
        # Measure flux
        try:
            flux = np.sum(image[y_int-aperture_radius:y_int+aperture_radius+1,
                               x_int-aperture_radius:x_int+aperture_radius+1][aperture_mask])
            if flux > 0:  # Only use valid measurements
                measured_fluxes.append(flux)
                star_mags.append(star['magnitude_r'])  # Use r-band magnitude for calibration
                star_positions.append((x, y))
        except IndexError:
            continue
    
    # Calculate zero point
    if len(measured_fluxes) > 0:
        zero_point = np.median(star_mags + 2.5 * np.log10(measured_fluxes))
        
        # Apply calibration
        calibrated_image = image.copy()
        valid_pixels = calibrated_image > 0
        calibrated_image[valid_pixels] = zero_point - 2.5 * np.log10(calibrated_image[valid_pixels])
        
        return calibrated_image, zero_point
    else:
        return image, None
