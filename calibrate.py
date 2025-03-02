from dark_manager import DarkFrameManager
from stacking_quality_assessment import ImageQualityAnalyzer
import os
import tempfile
import glob
import shutil
from astropy.io import fits
import numpy as np

def preprocess_and_analyze(input_dir, dark_base_dir, output_dir, pattern='*.fits'):
    """
    Preprocess FITS files with dark subtraction before quality analysis.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing light frames
    dark_base_dir : str
        Base directory with dark frames in temp_XXX subdirectories
    output_dir : str
        Directory for output results
    pattern : str, optional
        File pattern for FITS files
    """
    # Create temporary directory for calibrated files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize dark frame manager
        dark_manager = DarkFrameManager(dark_base_dir)
        
        # Get list of light frames
        light_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
        if not light_files:
            raise ValueError(f"No files found matching {pattern} in {input_dir}")
        
        print(f"Found {len(light_files)} light frames for calibration and analysis")
        
        # Create calibrated frames directory
        calibrated_dir = os.path.join(temp_dir, 'calibrated')
        os.makedirs(calibrated_dir, exist_ok=True)
        
        # Process each light frame
        for idx, light_file in enumerate(light_files):
            print(f"Calibrating frame {idx+1}/{len(light_files)}: {os.path.basename(light_file)}")
            
            try:
                # Open light frame
                with fits.open(light_file) as hdul:
                    light_data = hdul[0].data
                    header = hdul[0].header
                    
                    # Extract temperature if available
                    temp_c = header.get('CCD-TEMP', header.get('TEMP', 20.0))
                    if isinstance(temp_c, str):
                        temp_c = float(temp_c.split()[0])  # Handle '20.0 C' format
                    
                    # Get Bayer pattern if available
                    pattern = header.get('BAYERPAT', 'RGGB').strip()
                    
                    # Get appropriate dark frame
                    dark_data, msg = dark_manager.get_dark_frame(temp_c, pattern, light_data)
                    print(f"  {msg}")
                    
                    # Perform dark subtraction
                    calibrated_data = light_data - dark_data
                    
                    # Floor at zero to avoid negative values
                    calibrated_data = np.maximum(calibrated_data, 0)
                    
                    # Save calibrated frame
                    output_file = os.path.join(calibrated_dir, os.path.basename(light_file))
                    calibrated_hdu = fits.PrimaryHDU(calibrated_data, header=header)
                    calibrated_hdu.header['CALSTAT'] = 'D'  # Mark as dark-subtracted
                    calibrated_hdu.writeto(output_file, overwrite=True)
                    
            except Exception as e:
                print(f"Error calibrating {light_file}: {str(e)}")
                # Copy original file if calibration fails
                output_file = os.path.join(calibrated_dir, os.path.basename(light_file))
                shutil.copy(light_file, output_file)
        
        print("\nCalibration complete, starting quality analysis...")
        
        # Initialize image quality analyzer on calibrated files
        analyzer = ImageQualityAnalyzer(
            input_dir=calibrated_dir,
            output_dir=output_dir,
            file_pattern='*.fits'
        )
        
        # Run analysis
        analyzer.analyze_all_images()
        
        print("\nAnalysis completed!")
        print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    preprocess_and_analyze(
        input_dir='lights/temp_290',
        dark_base_dir='../dark_temps',
        output_dir='quality_results'
    )

