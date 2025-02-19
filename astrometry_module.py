import subprocess
import os
import numpy as np

def solve_astrometry(image_path):
    """
    Run astrometry.net's solve-field on an image
    """
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