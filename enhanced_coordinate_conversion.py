def enhanced_altaz_to_radec(fits_path, precise_lat=52.24509819278444, precise_long=0.0795527475486776):
    """
    Enhanced conversion from ALT/AZ to RA/DEC using precise location and FITS header keywords
    
    Parameters:
    -----------
    fits_path : str
        Path to the FITS file
    precise_lat : float
        Precise latitude in degrees
    precise_long : float
        Precise longitude in degrees
        
    Returns:
    --------
    dict
        Dictionary containing both calculated and header RA/DEC values, ALT/AZ values,
        and their differences
    """
    import ephem
    import astropy.io.fits as fits
    from datetime import datetime
    import numpy as np
    
    # Read FITS header
    header = fits.getheader(fits_path)
    
    # Extract ALT/AZ coordinates from FITS header
    alt = header.get('ALT', header.get('ALTITUDE', None))
    az = header.get('AZ', header.get('AZIMUTH', None))
    
    # Extract timestamp
    date_obs = header.get('DATE-OBS', None)
    if not date_obs:
        # Try alternative keywords
        date_obs = header.get('DATE', None)
    
    # Extract existing RA/DEC from header (if available)
    header_ra = header.get('CRVAL1', header.get('RA', None))
    header_dec = header.get('CRVAL2', header.get('DEC', None))
    
    # Check if we have all required values
    if None in [alt, az, date_obs]:
        print(f"Missing required header keywords in {fits_path}")
        return None
    
    # Setup observer with precise coordinates
    observer = ephem.Observer()
    observer.lat = str(precise_lat)
    observer.lon = str(precise_long)
    observer.elevation = 20  # Elevation in meters
    
    # Add atmospheric parameters (standard values if not in header)
    observer.pressure = header.get('PRESSURE', 1013.0)  # millibars (standard pressure at sea level)
    observer.temp = header.get('TEMPERAT', 15.0)  # degrees Celsius
    
    # Parse date
    try:
        # Handle different date formats
        dt = None
        date_formats = [
            '%Y-%m-%dT%H:%M:%S.%f',  # With microseconds
            '%Y-%m-%dT%H:%M:%S',     # Without microseconds
            '%Y-%m-%d %H:%M:%S.%f',  # Space separator with microseconds
            '%Y-%m-%d %H:%M:%S'      # Space separator without microseconds
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_obs, fmt)
                break
            except ValueError:
                continue
        
        if dt is None:
            raise ValueError(f"Could not parse date: {date_obs}")
            
        # Set date in observer
        observer.date = ephem.Date(dt)
        
        # Convert ALT/AZ to radians for ephem
        alt_rad = alt * ephem.pi/180
        az_rad = az * ephem.pi/180
        
        # Use the same parameter order as in the original script to maintain consistency
        ra, dec = observer.radec_of(az_rad, alt_rad)
        
        # Convert from ephem angle objects to degrees
        calc_ra = float(ra) * 180/ephem.pi
        calc_dec = float(dec) * 180/ephem.pi
        
        # Calculate differences (if header values exist)
        ra_diff_arcsec = None
        dec_diff_arcsec = None
        total_diff_arcsec = None
        
        if header_ra is not None and header_dec is not None:
            # Handle RA wrap-around (0-360 degrees)
            if calc_ra > 270 and header_ra < 90:
                # RA wrapped around 0 hour
                ra_diff_deg = (header_ra + 360) - calc_ra
            elif calc_ra < 90 and header_ra > 270:
                # RA wrapped around 0 hour
                ra_diff_deg = header_ra - (calc_ra + 360)
            else:
                ra_diff_deg = header_ra - calc_ra
                
            # Account for cos(DEC) factor in RA differences
            ra_diff_arcsec = ra_diff_deg * 3600 * np.cos(np.radians(header_dec))
            dec_diff_arcsec = (header_dec - calc_dec) * 3600
            total_diff_arcsec = np.sqrt(ra_diff_arcsec**2 + dec_diff_arcsec**2)
        
        # Prepare results dictionary
        results = {
            'filename': fits_path,
            'date_obs': date_obs,
            'alt': alt,
            'az': az,
            'header_ra': header_ra,
            'header_dec': header_dec,
            'calc_ra': calc_ra,
            'calc_dec': calc_dec,
            'ra_diff_arcsec': ra_diff_arcsec,
            'dec_diff_arcsec': dec_diff_arcsec,
            'total_diff_arcsec': total_diff_arcsec
        }
        
        return results
        
    except Exception as e:
        print(f"Error converting coordinates for {fits_path}: {str(e)}")
        return None

def analyze_coordinate_accuracy(directory, output_csv='coordinate_analysis.csv', 
                               lat=52.24509819278444, long=0.0795527475486776):
    """
    Analyze coordinate transformation accuracy for all FITS files in a directory
    
    Parameters:
    -----------
    directory : str
        Directory containing FITS files
    output_csv : str
        Path to save CSV results
    lat : float
        Precise latitude in degrees
    long : float
        Precise longitude in degrees
        
    Returns:
    --------
    DataFrame
        Pandas DataFrame with analysis results
    """
    import os
    import pandas as pd
    import numpy as np
    
    # Find all FITS files
    fits_files = [f for f in os.listdir(directory) if f.endswith(('.fits', '.fit', '.fts'))]
    
    if not fits_files:
        print(f"No FITS files found in {directory}")
        return None
    
    # Analyze all files
    results = []
    for fits_file in fits_files:
        fits_path = os.path.join(directory, fits_file)
        result = enhanced_altaz_to_radec(fits_path, lat, long)  # Fixed function name here
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate statistics
    valid_diffs = df[df['total_diff_arcsec'].notna()]
    
    if len(valid_diffs) > 0:
        median_diff = valid_diffs['total_diff_arcsec'].median()
        mean_diff = valid_diffs['total_diff_arcsec'].mean()
        std_diff = valid_diffs['total_diff_arcsec'].std()
        max_diff = valid_diffs['total_diff_arcsec'].max()
        min_diff = valid_diffs['total_diff_arcsec'].min()
        
        # Print summary statistics
        print("\nCoordinate Transformation Analysis")
        print("=" * 50)
        print(f"Number of files analyzed: {len(valid_diffs)}")
        print(f"Median difference: {median_diff:.2f} arcseconds")
        print(f"Mean difference: {mean_diff:.2f} arcseconds")
        print(f"Standard deviation: {std_diff:.2f} arcseconds")
        print(f"Range: {min_diff:.2f} - {max_diff:.2f} arcseconds")
        
        # Look for patterns
        print("\nPatterns in Coordinate Differences:")
        
        # Check for ALT correlation
        alt_corr = valid_diffs[['alt', 'total_diff_arcsec']].corr().iloc[0, 1]
        print(f"ALT vs. error correlation: {alt_corr:.2f}")
        
        # Check for AZ correlation
        az_corr = valid_diffs[['az', 'total_diff_arcsec']].corr().iloc[0, 1]
        print(f"AZ vs. error correlation: {az_corr:.2f}")
        
    # Save results
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
    
    return df
