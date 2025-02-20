if catalog not in catalog_params:
        print(f"Unknown catalog {catalog}")
        return None
    
    params = catalog_params[catalog]
    vizier.column_filters[params['mag_limit_column']] = f"<{mag_limit}"
    
    print(f"Querying {catalog} catalog for stars brighter than magnitude {mag_limit}")
    
    try:
        catalog_query = vizier.query_region(
            center, 
            radius=radius,
            catalog=params['catalog']
        )
        
        if len(catalog_query) > 0 and len(catalog_query[0]) > 0:
            stars = []
            for row in catalog_query[0]:
                try:
                    stars.append({
                        'ra': float(row[params['ra_column']]),
                        'dec': float(row[params['dec_column']]),
                        'magnitude_r': float(row[params['magnitude_column']])
                    })
                except (ValueError, KeyError) as e:
                    print(f"Warning: Could not parse star: {e}")
                    continue
            
            print(f"Found {len(stars)} stars in {catalog} catalog")
            for i, star in enumerate(stars[:5]):  # Print first 5 stars
                print(f"Star {i+1}: RA={star['ra']:.4f}, Dec={star['dec']:.4f}, "
                      f"mag={star['magnitude_r']:.2f}")
            return stars
        else:
            print(f"No stars found in {catalog} catalog")
            if len(catalog_query) > 0:
                print("Query response:")
                print(catalog_query[0])
    except Exception as e:
        print(f"Error querying {catalog} catalog: {str(e)}")
    
    return None

def solve_astrometry(image_path):
    """
    Run astrometry.net's solve-field on an image
    """
    import subprocess
    import os
    
    print(f"\nRunning astrometry.net on {image_path}")
    try:
        # Run solve-field with typical options
        cmd = [
            'solve-field',
            '--overwrite',
            '--no-plots',
            '--no-verify',
            '--scale-units', 'arcsecperpix',
            '--scale-low', '0.5',
            '--scale-high', '2.0',
            image_path
        ]
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Astrometric solution succeeded")
            
            # Get the new file path (.new)
            base = os.path.splitext(image_path)[0]
            new_path = base + '.new'
            
            # Check if solved file exists
            if os.path.exists(new_path):
                os.replace(new_path, image_path)
                print(f"Updated {image_path} with new astrometric solution")
                
                # Clean up temporary files
                for ext in ['.axy', '.corr', '.match', '.rdls', '.solved', '.wcs']:
                    temp_file = base + ext
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                return True
            else:
                print(f"Warning: Solved file not found at {new_path}")
                return False
        else:
            print("Astrometric solution failed")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running solve-field: {str(e)}")
        return False