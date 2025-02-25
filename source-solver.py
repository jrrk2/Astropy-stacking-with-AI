                       sources: Table, 
                       ra_center: float, 
                       dec_center: float, 
                       max_offset: float = 3.0,
                       mag_limit: float = 15.0,
                       search_radius: float = 0.5) -> Optional[Dict]:
        """
        Match extracted sources to a catalog (like Gaia).
        
        Args:
            sources: Table of extracted sources
            ra_center: Right Ascension of field center in degrees
            dec_center: Declination of field center in degrees
            max_offset: Maximum matching tolerance in arcseconds
            mag_limit: Magnitude limit for catalog sources
            search_radius: Search radius in degrees
        
        Returns:
            Dictionary of matching results or None if matching fails
        """
        try:
            # Try to use astroquery for Gaia catalog
            try:
                from astroquery.gaia import Gaia
                Gaia.ROW_LIMIT = -1  # No row limit
                
                # Create coordinate object
                radius = search_radius * u.degree
                
                # Gaia DR3 catalog query
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
                ORDER BY phot_g_mean_mag ASC
                """
                
                self.logger.info("Querying Gaia DR3 catalog...")
                job = Gaia.launch_job(query)
                catalog_sources = job.get_results()
                
                if len(catalog_sources) == 0:
                    self.logger.warning("No catalog sources found")
                    return None
                
                # Create SkyCoord objects for matching
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
                
                # Match sources
                idx, d2d, _ = image_coords.match_to_catalog_sky(catalog_coords)
                match_mask = d2d < max_offset * u.arcsec
                
                # Apply matching mask
                matched_sources = sources[match_mask]
                matched_catalog = catalog_sources[idx[match_mask]]
                
                if len(matched_sources) == 0:
                    self.logger.warning("No sources matched within tolerance")
                    return None
                
                # Calculate offsets
                ra_offsets = (matched_sources['ALPHA_J2000'] - matched_catalog['ra']) * 3600  # to arcsec
                dec_offsets = (matched_sources['DELTA_J2000'] - matched_catalog['dec']) * 3600
                
                # Return matching results
                return {
                    'n_matched': len(matched_sources),
                    'ra_offset_mean': float(np.mean(ra_offsets)),
                    'dec_offset_mean': float(np.mean(dec_offsets)),
                    'ra_offset_std': float(np.std(ra_offsets)),
                    'dec_offset_std': float(np.std(dec_offsets)),
                    'total_offset_mean': float(np.mean(np.sqrt(ra_offsets**2 + dec_offsets**2))),
                    'total_offset_std': float(np.std(np.sqrt(ra_offsets**2 + dec_offsets**2)))
                }
                
            except ImportError:
                self.logger.error("Astroquery not available. Cannot query Gaia catalog.")
                return None
            except Exception as e:
                self.logger.error(f"Catalog matching error: {str(e)}")
                return None
        
        except Exception as e:
            self.logger.error(f"Unexpected error in source matching: {str(e)}")
            return None

def process_directory(
    directory: str, 
    target: str,
    output_file: Optional[str] = None,
    **kwargs
) -> List[Dict]:
    """
    Process all FITS files in a directory using the Solver.
    
    Args:
        directory: Path to directory containing FITS files
        target: Target name to resolve via SIMBAD
        output_file: Optional CSV file to save results
        **kwargs: Additional parameters to pass to Solver.solve()
    
    Returns:
        List of solving results for each file
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Find FITS files
    fits_files = list(Path(directory).rglob('*.fits'))
    
    if not fits_files:
        logging.warning(f"No FITS files found in {directory}")
        return []
    
    # Query target coordinates
    with Solver() as solver:
        target_info = solver.query_target(target)
        
        if target_info is None:
            logging.error(f"Could not resolve target: {target}")
            return []
        
        # Prepare position hint
        position_hint = PositionHint(
            ra_deg=target_info.ra_deg, 
            dec_deg=target_info.dec_deg
        )
        
        # Process files
        results = []
        for fits_file in fits_files:
            try:
                solution = solver.solve(
                    str(fits_file), 
                    position_hint=position_hint,
                    **kwargs
                )
                solution['filename'] = fits_file.name
                solution['target'] = target_info.identifier
                results.append(solution)
            except Exception as e:
                logging.error(f"Error solving {fits_file}: {e}")
        
        # Optionally save to CSV
        if output_file and results:
            try:
                import csv
                with open(output_file, 'w', newline='') as csvfile:
                    # Determine fieldnames dynamically
                    fieldnames = set()
                    for result in results:
                        fieldnames.update(result.keys())
                    
                    writer = csv.DictWriter(csvfile, fieldnames=sorted(fieldnames))
                    writer.writeheader()
                    writer.writerows(results)
                logging.info(f"Results saved to {output_file}")
            except Exception as e:
                logging.error(f"Error saving results to CSV: {e}")
        
        return results

# Example usage script
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Astronomical Source Solver')
    parser.add_argument('directory', help='Directory containing FITS files')
    parser.add_argument('--target', required=True, help='Target name (e.g., "M51")')
    parser.add_argument('--output', help='Output CSV file for results')
    parser.add_argument('--max-offset', type=float, default=3.0, 
                        help='Maximum matching offset in arcseconds')
    
    args = parser.parse_args()
    
    # Process directory and solve astrometry
    results = process_directory(
        args.directory, 
        target=args.target, 
        output_file=args.output,
        solution_parameters={'max_offset': args.max_offset}
    )
    
    # Print summary
    if results:
        print("\nProcessing Summary:")
        print(f"Total files processed: {len(results)}")
        print(f"Successful solutions: {sum(1 for r in results if r['sources_matched'] > 0)}")
        
        # Calculate overall statistics
        ra_offsets = [r['ra_offset_mean'] for r in results if r['ra_offset_mean'] is not None]
        dec_offsets = [r['dec_offset_mean'] for r in results if r['dec_offset_mean'] is not None]
        
        if ra_offsets and dec_offsets:
            print("\nOverall Pointing Statistics:")
            print(f"Mean RA Offset: {np.mean(ra_offsets):.2f} arcsec")
            print(f"Mean Dec Offset: {np.mean(dec_offsets):.2f} arcsec")
