import os
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
import logging
from pathlib import Path
import argparse
from scipy.stats import pearsonr

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_sources(fits_file: str, fwhm: float = 3.0, threshold_sigma: float = 5.0,
                   sharplo: float = 0.5, sharphi: float = 2.0,
                   roundlo: float = 0.5, roundhi: float = 1.5,
                   minsize: int = 2, min_flux_ratio: float = 0.1) -> Table | None:
    """
    Extract sources from the image using photutils with improved filtering.
    
    Args:
        fits_file: Path to the FITS file
        fwhm: Full width at half maximum of stars in pixels
        threshold_sigma: Detection threshold in sigma above background
        sharplo: Lower bound on sharpness for star detection
        sharphi: Upper bound on sharpness for star detection
        roundlo: Lower bound on roundness for star detection
        roundhi: Upper bound on roundness for star detection
        minsize: Minimum size of stars in pixels
        min_flux_ratio: Minimum flux ratio compared to brightest star
        
    Returns:
        astropy.table.Table: Table of extracted sources or None if extraction fails
    """
    try:
        if not os.path.exists(fits_file):
            logger.error(f"FITS file not found: {fits_file}")
            return None
            
        # Read FITS file
        with fits.open(fits_file) as hdul:
            # Get primary HDU data
            data = hdul[0].data
            header = hdul[0].header
            
            # Extract field rotation angle and mount position if available
            rotation_angle = header.get('DER', None)
            alt = header.get('ALT', None)
            az = header.get('AZ', None)
            
            if rotation_angle is not None:
                logger.info(f"Derotator angle: {rotation_angle:.2f}°")
            else:
                logger.info("No derotator angle (DER) found in header")
                
            if alt is not None and az is not None:
                logger.info(f"Mount position: ALT={alt:.2f}°, AZ={az:.2f}°")
            else:
                logger.info("No ALT/AZ position found in header")
            
            if data is None:
                # Try the first extension if primary HDU has no data
                if len(hdul) > 1 and hdul[1].data is not None:
                    data = hdul[1].data
                    header = hdul[1].header
                else:
                    logger.error(f"No image data found in {fits_file}")
                    return None
            
            # Get WCS information
            try:
                wcs = WCS(header)
            except Exception as e:
                logger.error(f"Error parsing WCS: {str(e)}")
                return None
        
        # Estimate background - FIXED: Use SigmaClip class instead of float value
        try:
            sigma_clip = SigmaClip(sigma=3.0)  # Fixed: Using SigmaClip object
            bkg_estimator = MedianBackground()
            bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                              sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
            data_sub = data - bkg.background
            threshold = threshold_sigma * bkg.background_rms.mean()
        except Exception as e:
            logger.warning(f"Error estimating background: {str(e)}. Using simple statistics.")
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            data_sub = data - median
            threshold = threshold_sigma * std
        
        # Source detection with improved filtering for hot pixels
        daofind = DAOStarFinder(
            fwhm=fwhm,
            threshold=threshold,
            sharplo=sharplo,
            sharphi=sharphi,
            roundlo=roundlo,
            roundhi=roundhi,
            peakmax=None  # No upper limit on peak flux
        )
        
        sources = daofind(data_sub)
        
        if sources is None or len(sources) == 0:
            logger.warning(f"No sources detected in {fits_file}")
            return None
            
        # Filter out faint sources that might be noise
        if 'flux' in sources.colnames:
            max_flux = np.max(sources['flux'])
            flux_threshold = max_flux * min_flux_ratio
            bright_sources = sources[sources['flux'] > flux_threshold]
            
            # Skip if we've filtered out too many sources
            if len(bright_sources) < 10:
                logger.warning(f"Only {len(bright_sources)} sources above flux threshold. Using all sources.")
            else:
                sources = bright_sources
                logger.info(f"Filtered to {len(sources)} sources based on minimum flux threshold")
            
        # Apply additional filtering based on star shape if needed
        if 'sharpness' in sources.colnames and 'roundness1' in sources.colnames:
            good_sources = np.logical_and(
                np.logical_and(sources['sharpness'] > sharplo, sources['sharpness'] < sharphi),
                np.logical_and(sources['roundness1'] > roundlo, sources['roundness1'] < roundhi)
            )
            
            filtered_sources = sources[good_sources]
            
            # Skip if we've filtered out too many sources
            if len(filtered_sources) < 10:
                logger.warning(f"Only {len(filtered_sources)} sources after shape filtering. Using all sources.")
            else:
                sources = filtered_sources
                logger.info(f"Filtered to {len(sources)} sources based on star shape")
            
        # Add RA/Dec columns using WCS
        pixel_positions = np.column_stack((sources['xcentroid'], sources['ycentroid']))
        world_positions = wcs.pixel_to_world(pixel_positions[:, 0], pixel_positions[:, 1])
        
        sources['ALPHA_J2000'] = world_positions.ra.deg
        sources['DELTA_J2000'] = world_positions.dec.deg
        
        logger.info(f"Successfully extracted {len(sources)} sources")
        return sources
        
    except Exception as e:
        logger.error(f"Error extracting sources: {str(e)}")
        return None

def apply_rotation_correction(image_coords, center_coords, rotation_angle):
    """
    Apply a rotation correction to image coordinates based on field rotation angle.
    
    Args:
        image_coords: SkyCoord object with image source coordinates
        center_coords: SkyCoord object with field center coordinates
        rotation_angle: Rotation angle in degrees
        
    Returns:
        SkyCoord: Corrected coordinates
    """
    try:
        # For each source, calculate position angle and separation from center
        separations = center_coords.separation(image_coords)
        position_angles = center_coords.position_angle(image_coords)
        
        # Apply rotation (derotator angle adjustment)
        corrected_angles = position_angles + rotation_angle * u.deg
        
        # Calculate new positions
        new_coords = []
        for i in range(len(image_coords)):
            new_coord = center_coords.directional_offset_by(
                corrected_angles[i], separations[i])
            new_coords.append(new_coord)
        
        # Create new SkyCoord from list of coordinates
        return SkyCoord(new_coords)
    except Exception as e:
        logger.warning(f"Rotation correction failed: {e}")
        return image_coords

def cross_match_gaia(fits_file: str, search_radius: float = 0.5, mag_limit: float = 15.0) -> Table | None:
    """
    Cross-match extracted sources with Gaia DR3 catalog.
    
    Args:
        fits_file: Path to the FITS file
        search_radius: Search radius in degrees
        mag_limit: Limiting magnitude for Gaia catalog query
        
    Returns:
        astropy.table.Table: Table of matched Gaia sources or None if query fails
    """
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            
            # Try to get coordinates from header
            ra_center = None
            dec_center = None
            
            # Try different keyword combinations
            if 'CRVAL1' in header and 'CRVAL2' in header:
                ra_center = header['CRVAL1']
                dec_center = header['CRVAL2']
            elif 'RA' in header and 'DEC' in header:
                ra_center = header['RA']
                dec_center = header['DEC']
            elif 'OBJCTRA' in header and 'OBJCTDEC' in header:
                # Convert from sexagesimal to decimal if needed
                ra_str = header['OBJCTRA']
                dec_str = header['OBJCTDEC']
                try:
                    coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                    ra_center = coord.ra.deg
                    dec_center = coord.dec.deg
                except:
                    logger.error("Could not parse RA/DEC strings")
                    return None
            
            if ra_center is None or dec_center is None:
                logger.error("Could not find RA/DEC information in FITS header")
                return None
                
            # Validate coordinates
            if not (-360 <= ra_center <= 360 and -90 <= dec_center <= 90):
                logger.error(f"Invalid coordinates: RA={ra_center}, Dec={dec_center}")
                return None
                
            # Log the field center for debugging
            logger.info(f"Field center: RA={ra_center:.6f}°, Dec={dec_center:.6f}°")
            
        from astroquery.gaia import Gaia
        Gaia.ROW_LIMIT = -1
        
        coord = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.deg, u.deg))
        radius = search_radius * u.degree
        
        # Query Gaia DR3 - use supplied magnitude limit
        query = f"""
        SELECT 
            source_id, ra, dec, 
            phot_g_mean_mag, 
            parallax, parallax_error,
            pmra, pmdec,
            phot_bp_mean_mag, phot_rp_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 
            1=CONTAINS(
                POINT(ra, dec), 
                CIRCLE({ra_center}, {dec_center}, {radius.value})
            )
            AND phot_g_mean_mag < {mag_limit}
        ORDER BY phot_g_mean_mag ASC  -- Sort by brightness
        """
        
        job = Gaia.launch_job(query)
        results = job.get_results()
        
        if len(results) == 0:
            logger.warning(f"No Gaia sources found within {search_radius}° of field center")
        else:
            logger.info(f"Retrieved {len(results)} Gaia sources")
            
        return results
        
    except Exception as e:
        logger.error(f"Gaia catalog query failed: {str(e)}")
        return None

def verify_pointing(fits_file: str, max_offset: float = 5.0, 
                   fwhm: float = 3.0, threshold_sigma: float = 5.0,
                   sharplo: float = 0.5, sharphi: float = 2.0,
                   roundlo: float = 0.5, roundhi: float = 1.5,
                   minsize: int = 2, min_flux_ratio: float = 0.1,
                   apply_rotation: bool = True) -> dict:
    """
    Verify telescope pointing accuracy by comparing extracted sources with Gaia catalog.
    
    Args:
        fits_file: Path to the FITS file
        max_offset: Maximum allowed offset in arcseconds for matching
        fwhm: Full width at half maximum of stars in pixels
        threshold_sigma: Detection threshold in sigma above background
        sharplo: Lower bound on sharpness for star detection
        sharphi: Upper bound on sharpness for star detection
        roundlo: Lower bound on roundness for star detection
        roundhi: Upper bound on roundness for star detection
        minsize: Minimum size of stars in pixels
        min_flux_ratio: Minimum flux ratio compared to brightest star
        apply_rotation: Whether to apply rotation correction
        
    Returns:
        dict: Dictionary containing analysis results
    """
    results = {
        'success': False,
        'n_extracted': 0,
        'n_catalog': 0,
        'n_matched': 0,
        'ra_offset_mean': None,
        'dec_offset_mean': None,
        'ra_offset_std': None,
        'dec_offset_std': None,
        'rotation_angle': None,
        'alt': None,
        'az': None
    }
    
    # Extract rotation angle and mount position from FITS header
    try:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            results['rotation_angle'] = header.get('DER', None)
            results['alt'] = header.get('ALT', None)
            results['az'] = header.get('AZ', None)
            
            # Get image center for rotation correction
            if 'CRVAL1' in header and 'CRVAL2' in header:
                center_ra = header['CRVAL1'] 
                center_dec = header['CRVAL2']
                center_coords = SkyCoord(ra=center_ra, dec=center_dec, unit=(u.deg, u.deg))
            else:
                center_coords = None
    except Exception as e:
        logger.warning(f"Could not extract header data: {e}")
        center_coords = None
    
    # Extract sources with improved filtering
    sources = extract_sources(
        fits_file,
        fwhm=fwhm,
        threshold_sigma=threshold_sigma,
        sharplo=sharplo,
        sharphi=sharphi,
        roundlo=roundlo,
        roundhi=roundhi,
        minsize=minsize,
        min_flux_ratio=min_flux_ratio
    )
    
    if sources is None:
        return results
    results['n_extracted'] = len(sources)
    
    # Get catalog sources
    catalog_sources = cross_match_gaia(fits_file)
    if catalog_sources is None:
        return results
    results['n_catalog'] = len(catalog_sources)
    
    if len(sources) == 0 or len(catalog_sources) == 0:
        logger.warning("No sources available for matching")
        return results
        
    # Perform cross-matching
    try:
        image_coords = SkyCoord(
            ra=sources['ALPHA_J2000'],
            dec=sources['DELTA_J2000'],
            unit=(u.deg, u.deg)
        )
        
        catalog_coords = SkyCoord(
            ra=catalog_sources['ra'],
            dec=catalog_sources['dec'],
            unit=(u.deg, u.deg)
        )
        
        # Get field rotation from results
        rotation_angle = results.get('rotation_angle')
        
        # If we have rotation information and center coordinates, apply correction
        original_image_coords = image_coords
        if apply_rotation and rotation_angle is not None and center_coords is not None:
            logger.info(f"Accounting for derotator angle: {rotation_angle:.2f}°")
            try:
                # Apply rotation correction
                image_coords = apply_rotation_correction(image_coords, center_coords, rotation_angle)
            except Exception as e:
                logger.warning(f"Rotation correction failed: {e}")
                # Fall back to original coords
                image_coords = original_image_coords
        
        # APPROACH 1: Try progressive matching tolerances
        matching_tolerances = [1.0, 2.0, 3.0, 5.0, 10.0]  # in arcseconds
        matched_sources = None
        matched_catalog = None
        used_tolerance = max_offset
        
        for tolerance in matching_tolerances:
            if tolerance > max_offset:
                break
                
            # Match sources
            idx, d2d, _ = image_coords.match_to_catalog_sky(catalog_coords)
            match_mask = d2d < tolerance * u.arcsec
            
            temp_matched_sources = sources[match_mask]
            temp_matched_catalog = catalog_sources[idx[match_mask]]
            
            if len(temp_matched_sources) >= 3:  # We found enough matches
                matched_sources = temp_matched_sources
                matched_catalog = temp_matched_catalog
                used_tolerance = tolerance
                logger.info(f"Found {len(matched_sources)} matches using {tolerance}\" tolerance")
                break
                
        # If we didn't find matches with the progressive approach, use the max tolerance
        if matched_sources is None:
            idx, d2d, _ = image_coords.match_to_catalog_sky(catalog_coords)
            match_mask = d2d < max_offset * u.arcsec
            
            matched_sources = sources[match_mask]
            matched_catalog = catalog_sources[idx[match_mask]]
        
        results['n_matched'] = len(matched_sources)
        
        if len(matched_sources) > 0:
            # Calculate offsets
            ra_offsets = (matched_sources['ALPHA_J2000'] - matched_catalog['ra']) * 3600  # Convert to arcsec
            dec_offsets = (matched_sources['DELTA_J2000'] - matched_catalog['dec']) * 3600
            
            # Log individual matches for debugging
            if len(matched_sources) <= 10:  # Only log details for a small number of matches
                logger.info("Match details:")
                for i in range(len(matched_sources)):
                    logger.info(f"  Match {i+1}: ΔRA={ra_offsets[i]:.2f}\", ΔDec={dec_offsets[i]:.2f}\" "
                               f"(Distance: {d2d[match_mask][i].arcsec:.2f}\")")
            
            # Calculate total pointing error (magnitude)
            total_offsets = np.sqrt(ra_offsets**2 + dec_offsets**2)
            
            # Calculate angular direction of the offset (in degrees)
            # 0° = North, 90° = East, etc.
            offset_angles = np.degrees(np.arctan2(ra_offsets, dec_offsets))
            # Convert to 0-360 range
            offset_angles = np.where(offset_angles < 0, offset_angles + 360, offset_angles)
            
            # If we have rotation information, include it in analysis
            if rotation_angle is not None:
                # Analyze if the offsets correlate with field rotation
                # A simple correlation metric
                try:
                    rot_correlation = np.corrcoef(np.array([offset_angles, np.ones_like(offset_angles) * rotation_angle]))[0, 1]
                except:
                    rot_correlation = np.nan
            else:
                rot_correlation = np.nan
            
            # Update results with detailed pointing analysis
            results.update({
                'success': True,
                'ra_offset_mean': float(np.mean(ra_offsets)),
                'dec_offset_mean': float(np.mean(dec_offsets)),
                'ra_offset_std': float(np.std(ra_offsets)) if len(matched_sources) > 1 else 0.0,
                'dec_offset_std': float(np.std(dec_offsets)) if len(matched_sources) > 1 else 0.0,
                'match_tolerance': used_tolerance,
                'pointing_analysis': {
                    'total_offset_mean': float(np.mean(total_offsets)),
                    'total_offset_std': float(np.std(total_offsets)) if len(matched_sources) > 1 else 0.0,
                    'angle_mean': float(np.mean(offset_angles)),
                    'angle_std': float(np.std(offset_angles)) if len(matched_sources) > 1 else 0.0,
                    'rotation_correlation': rot_correlation
                }
            })
            
            logger.info(f"Matched {len(matched_sources)} sources with {used_tolerance}\" tolerance")
            logger.info(f"Mean offset: RA={results['ra_offset_mean']:.2f}\", "
                       f"Dec={results['dec_offset_mean']:.2f}\"")
            
        else:
            logger.warning("No matches found within tolerance")
            
    except Exception as e:
        logger.error(f"Error during cross-matching: {str(e)}")
        
    return results

def process_directory(directory: str, max_offset: float = 5.0,
                     fwhm: float = 3.0, threshold_sigma: float = 5.0,
                     sharplo: float = 0.5, sharphi: float = 2.0,
                     roundlo: float = 0.5, roundhi: float = 1.5,
                     minsize: int = 2, min_flux_ratio: float = 0.1,
                     apply_rotation: bool = True) -> list:
    """
    Process all FITS files in a directory and its subdirectories.
    """
    all_results = []
    
    try:
        fits_files = list(Path(directory).rglob('*.fits'))
        if not fits_files:
            logger.warning(f"No FITS files found in {directory}")
            return all_results
            
        logger.info(f"Found {len(fits_files)} FITS files to process")
        
        for fits_path in fits_files:
            logger.info(f"\nProcessing {fits_path.name}")
            results = verify_pointing(
                str(fits_path),
                max_offset=max_offset,
                fwhm=fwhm,
                threshold_sigma=threshold_sigma,
                sharplo=sharplo,
                sharphi=sharphi,
                roundlo=roundlo,
                roundhi=roundhi,
                minsize=minsize,
                min_flux_ratio=min_flux_ratio,
                apply_rotation=apply_rotation
            )
            
            if results['success']:
                logger.info("Analysis completed successfully")
                logger.info(f"RA offset: {results['ra_offset_mean']:.2f}\" ± {results['ra_offset_std']:.2f}\"")
                logger.info(f"Dec offset: {results['dec_offset_mean']:.2f}\" ± {results['dec_offset_std']:.2f}\"")
                
                # Log additional alt-az specific information
                if 'pointing_analysis' in results:
                    pa = results['pointing_analysis']
                    logger.info(f"Total offset: {pa['total_offset_mean']:.2f}\" ± {pa['total_offset_std']:.2f}\"")
                    logger.info(f"Offset angle: {pa['angle_mean']:.1f}° ± {pa['angle_std']:.1f}°")
                    
                if results['rotation_angle'] is not None:
                    logger.info(f"Derotator angle: {results['rotation_angle']:.2f}°")
                    if 'pointing_analysis' in results and results['pointing_analysis']['rotation_correlation'] is not None:
                        corr = results['pointing_analysis']['rotation_correlation']
                        logger.info(f"Rotation-offset correlation: {corr:.2f}")
                
                # Add filename to results and save
                results['filename'] = fits_path.name
                all_results.append(results)
            else:
                logger.warning("Analysis failed or produced no matches")
                
        # After processing all files, generate a summary
        if all_results:
            logger.info("\n===== ANALYSIS SUMMARY =====")
            logger.info(f"Successfully analyzed {len(all_results)} out of {len(fits_files)} files")
            
            # Calculate average offsets
            ra_offsets = [r['ra_offset_mean'] for r in all_results]
            dec_offsets = [r['dec_offset_mean'] for r in all_results]
            
            logger.info(f"Average RA offset: {np.mean(ra_offsets):.2f}\" ± {np.std(ra_offsets):.2f}\"")
            logger.info(f"Average Dec offset: {np.mean(dec_offsets):.2f}\" ± {np.std(dec_offsets):.2f}\"")
            
            # Report on rotation angles if available
            rot_angles = [r['rotation_angle'] for r in all_results if r['rotation_angle'] is not None]
            if rot_angles:
                logger.info(f"Derotator angles: {np.mean(rot_angles):.2f}° ± {np.std(rot_angles):.2f}°")
        
        return all_results
                
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        return all_results

def analyze_alt_az_results(results_list):
    """
    Perform specialized analysis for alt-azimuth mount data.
    """
    if not results_list:
        logger.warning("No results to analyze")
        return
    
    logger.info("\n===== ALT-AZ MOUNT ANALYSIS =====")
    
    # Extract pointing errors, rotation angles, and mount positions
    ra_offsets = []
    dec_offsets = []
    total_offsets = []
    angles = []
    rot_angles = []
    alts = []
    azs = []
    
    # Collect data from all successful results
    for result in results_list:
        if not result['success']:
            continue
            
        ra_offsets.append(result['ra_offset_mean'])
        dec_offsets.append(result['dec_offset_mean'])
        
        if 'pointing_analysis' in result:
            total_offsets.append(result['pointing_analysis']['total_offset_mean'])
            angles.append(result['pointing_analysis']['angle_mean'])
        
        if result['rotation_angle'] is not None:
            rot_angles.append(result['rotation_angle'])
            
        if result['alt'] is not None:
            alts.append(result['alt'])
            
        if result['az'] is not None:
            azs.append(result['az'])
    
    # Calculate vector statistics
    ra_mean = np.mean(ra_offsets) if ra_offsets else None
    dec_mean = np.mean(dec_offsets) if dec_offsets else None
    total_mean = np.mean(total_offsets) if total_offsets else None
    
    # Analyze correlation with rotation if we have rotation data
    if rot_angles and angles:
        try:
            # Simple correlation
            correlation = np.corrcoef(rot_angles, angles)[0, 1]
            logger.info(f"Correlation between derotator angle and offset direction: {correlation:.2f}")
        except:
            logger.info("Could not calculate correlation between derotator angle and offset direction")
        
        # Group by rotation angle ranges
        rot_groups = {
            "0-90°": [],
            "90-180°": [],
            "180-270°": [],
            "270-360°": []
        }
        
        for i, rot in enumerate(rot_angles):
            if i >= len(total_offsets):
                continue
                
            if 0 <= rot < 90:
                rot_groups["0-90°"].append(total_offsets[i])
            elif 90 <= rot < 180:
                rot_groups["90-180°"].append(total_offsets[i])
            elif 180 <= rot < 270:
                rot_groups["180-270°"].append(total_offsets[i])
            else:
                rot_groups["270-360°"].append(total_offsets[i])
        
        # Calculate average error by rotation quadrant
        logger.info("Average pointing error by derotator angle quadrant:")
        for quadrant, errors in rot_groups.items():
            if errors:
                logger.info(f"  {quadrant}: {np.mean(errors):.2f}\" (n={len(errors)})")
    
    # Output overall conclusion
    logger.info("\nCONCLUSION:")
    if total_mean:
        logger.info(f"Average pointing error: {total_mean:.2f} arcseconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify telescope pointing using Gaia catalog')
    parser.add_argument('directory', help='Directory containing FITS files')
    parser.add_argument('--radius', type=float, default=0.5, help='Search radius in degrees (default: 0.5)')
    parser.add_argument('--match-tolerance', type=float, default=5.0, help='Match tolerance in arcseconds (default: 5.0)')
    parser.add_argument('--alt-az', action='store_true', help='Perform specialized analysis for alt-azimuth mounts')
    parser.add_argument('--fwhm', type=float, default=3.0, help='Full width at half maximum of stars in pixels (default: 3.0)')
    parser.add_argument('--threshold', type=float, default=5.0, help='Detection threshold in sigma above background (default: 5.0)')
    parser.add_argument('--sharplo', type=float, default=0.5, help='Lower bound on sharpness for star detection (default: 0.5)')
    parser.add_argument('--sharphi', type=float, default=2.0, help='Upper bound on sharpness for star detection (default: 2.0)')
    parser.add_argument('--roundlo', type=float, default=0.5, help='Lower bound on roundness for star detection (default: 0.5)')
    parser.add_argument('--roundhi', type=float, default=1.5, help='Upper bound on roundness for star detection (default: 1.5)')
    parser.add_argument('--minsize', type=int, default=2, help='Minimum size of stars in pixels (default: 2)')
    parser.add_argument('--min-flux-ratio', type=float, default=0.1, help='Minimum flux ratio compared to brightest star (default: 0.1)')
    parser.add_argument('--no-rotation', action='store_true', help='Disable rotation correction')
    parser.add_argument('--output', help='Output file for results (CSV format)')
    parser.add_argument('--mag-limit', type=float, default=15.0, help='Limiting magnitude for Gaia catalog query (default: 15.0)')
    
    args = parser.parse_args()
    
    results = process_directory(
        args.directory,
        max_offset=args.match_tolerance,
        fwhm=args.fwhm,
        threshold_sigma=args.threshold,
        sharplo=args.sharplo,
        sharphi=args.sharphi,
        roundlo=args.roundlo,
        roundhi=args.roundhi,
        minsize=args.minsize,
        min_flux_ratio=args.min_flux_ratio,
        apply_rotation=not args.no_rotation
    )
    
    # If we found results and alt-az flag is set, perform specialized analysis
    if results and args.alt_az:
        analyze_alt_az_results(results)
        
    # If output file is specified, save results to CSV
    if args.output and results:
        try:
            import csv
            with open(args.output, 'w', newline='') as csvfile:
                fieldnames = ['filename', 'success', 'n_extracted', 'n_catalog', 'n_matched', 
                             'ra_offset_mean', 'dec_offset_mean', 'ra_offset_std', 'dec_offset_std',
                             'rotation_angle', 'alt', 'az']
                             
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    # Create a simplified version of the result dictionary
                    row = {field: result.get(field, None) for field in fieldnames}
                    writer.writerow(row)
                    
                logger.info(f"Results saved to {args.output}")
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")