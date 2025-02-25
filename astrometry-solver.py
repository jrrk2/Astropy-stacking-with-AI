import os
import sys
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astropy.wcs import WCS
from astropy.io import fits

def get_target_coordinates(target_name):
    """
    Get coordinates for a specific astronomical object using SIMBAD.
    
    Parameters:
    -----------
    target_name : str
        Name of the astronomical object (e.g., "M31", "NGC 224", "Andromeda")
        
    Returns:
    --------
    SkyCoord object with the coordinates
    """
    try:
        result_table = Simbad.query_object(target_name)
        if result_table is None:
            print(f"Target {target_name} not found in SIMBAD database.")
            return None
            
        ra = result_table['RA'][0]
        dec = result_table['DEC'][0]
        
        # Convert from SIMBAD string format to SkyCoord
        coords = SkyCoord(ra + " " + dec, unit=(u.hourangle, u.deg))
        print(f"Coordinates for {target_name}: RA={coords.ra.deg:.6f}°, Dec={coords.dec.deg:.6f}°")
        return coords
        
    except Exception as e:
        print(f"Error retrieving coordinates: {e}")
        return None

def solve_field_with_coordinates(image_path, coords, tolerance=1.0):
    """
    Solve the astrometry of an image using a known starting position with tolerance.
    
    Parameters:
    -----------
    image_path : str
        Path to the FITS or image file to solve
    coords : SkyCoord
        Starting coordinates for the solver
    tolerance : float
        Search radius in degrees
        
    Returns:
    --------
    Path to the solved file if successful, None otherwise
    """
    # Convert coordinates to degrees for the solver
    ra_deg = coords.ra.deg
    dec_deg = coords.dec.deg
    
    # Build the command for astrometry.net's solve-field
    output_dir = os.path.dirname(image_path) or '.'
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    cmd = [
        "solve-field",
        "--ra", f"{ra_deg:.6f}",
        "--dec", f"{dec_deg:.6f}",
        "--radius", f"{tolerance:.2f}",
        "--no-plots",
        "--overwrite",
        "--dir", output_dir,
        "--new-fits", f"{output_dir}/{base_filename}.solved.fits",
        image_path
    ]
    
    cmd_str = " ".join(cmd)
    print(f"Running astrometry solver with command:\n{cmd_str}")
    
    # Execute the command
    import subprocess
    result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        solved_path = f"{output_dir}/{base_filename}.solved.fits"
        if os.path.exists(solved_path):
            print(f"Successfully solved the field. Solution saved to: {solved_path}")
            return solved_path
    
    print("Astrometry solver failed with following output:")
    print(result.stdout)
    print(result.stderr)
    return None

def analyze_solution(solved_fits_path):
    """
    Analyze the astrometric solution to extract useful information.
    
    Parameters:
    -----------
    solved_fits_path : str
        Path to the solved FITS file
        
    Returns:
    --------
    Dictionary with solution details
    """
    try:
        with fits.open(solved_fits_path) as hdul:
            header = hdul[0].header
            
            # Create WCS object from the header
            wcs = WCS(header)
            
            # Extract key information
            info = {
                "CRVAL1": header.get("CRVAL1", None),  # RA of reference pixel
                "CRVAL2": header.get("CRVAL2", None),  # Dec of reference pixel
                "CRPIX1": header.get("CRPIX1", None),  # X reference pixel
                "CRPIX2": header.get("CRPIX2", None),  # Y reference pixel
                "CD1_1": header.get("CD1_1", None),    # Transformation matrix
                "CD1_2": header.get("CD1_2", None),
                "CD2_1": header.get("CD2_1", None),
                "CD2_2": header.get("CD2_2", None),
                "PIXSCALE": header.get("PIXSCALE", None),  # Pixel scale in arcsec/pixel
                "ORIENTAT": header.get("ORIENTAT", None),  # Field orientation
            }
            
            # Calculate field of view if pixel dimensions are available
            if hdul[0].data is not None:
                ny, nx = hdul[0].data.shape[-2:]
                corners = [wcs.pixel_to_world(0, 0),
                           wcs.pixel_to_world(nx-1, 0),
                           wcs.pixel_to_world(nx-1, ny-1),
                           wcs.pixel_to_world(0, ny-1)]
                
                # Calculate diagonal field of view
                diag_fov = corners[0].separation(corners[2])
                info["FIELD_OF_VIEW"] = diag_fov.deg
                
                # Calculate pixel scale manually if not in header
                if info["PIXSCALE"] is None:
                    pixel_scale = corners[0].separation(corners[1]) / nx
                    info["CALCULATED_PIXSCALE"] = pixel_scale.arcsec
            
            return info
            
    except Exception as e:
        print(f"Error analyzing solution: {e}")
        return None

def main():
    # Choose a well-known target
    target = "M51"  # Whirlpool Galaxy
    
    # Get coordinates
    coords = get_target_coordinates(target)
    if coords is None:
        return
    
    # Example usage with an image file
    # Note: User would need to provide their own image file path
    image_path = input("Enter the path to your astronomical image file: ")
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        return
    
    # Set tolerance (search radius in degrees)
    tolerance = 1.0
    
    # Solve the field
    solved_path = solve_field_with_coordinates(image_path, coords, tolerance)
    
    # If solved, analyze the solution
    if solved_path:
        solution_info = analyze_solution(solved_path)
        if solution_info:
            print("\nAstrometric solution details:")
            for key, value in solution_info.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
