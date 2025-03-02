from dark_manager import DarkFrameManager
from stacking_quality_assessment import ImageQualityAnalyzer
import os
import tempfile
import glob
import shutil
import argparse
import sys
from astropy.io import fits
import numpy as np
from tqdm import tqdm

def preprocess_and_analyze(input_dir, dark_base_dir, output_dir, pattern='*.fits', skip_calibration=False):
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
    skip_calibration : bool, optional
        Skip dark subtraction if True (just run quality analysis)
    """
    # Initialize output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if skip_calibration:
        print(f"Skipping calibration, running direct analysis on {input_dir}")
        analyzer = ImageQualityAnalyzer(
            input_dir=input_dir,
            output_dir=output_dir,
            file_pattern=pattern
        )
        analyzer.analyze_all_images()
        return
    
    # Create temporary directory for calibrated files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize dark frame manager
        try:
            dark_manager = DarkFrameManager(dark_base_dir)
        except Exception as e:
            print(f"Error initializing dark frame manager: {str(e)}")
            print("Falling back to uncalibrated analysis.")
            analyzer = ImageQualityAnalyzer(
                input_dir=input_dir,
                output_dir=output_dir,
                file_pattern=pattern
            )
            analyzer.analyze_all_images()
            return
        
        # Get list of light frames
        light_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
        if not light_files:
            raise ValueError(f"No files found matching {pattern} in {input_dir}")
        
        print(f"Found {len(light_files)} light frames for calibration and analysis")
        
        # Create calibrated frames directory
        calibrated_dir = os.path.join(temp_dir, 'calibrated')
        os.makedirs(calibrated_dir, exist_ok=True)
        
        # Track calibration statistics
        successful_calibrations = 0
        failed_calibrations = 0
        
        # Process each light frame
        for idx, light_file in enumerate(tqdm(light_files, desc="Calibrating frames")):
            basename = os.path.basename(light_file)
            
            try:
                # Open light frame
                with fits.open(light_file) as hdul:
                    light_data = hdul[0].data
                    header = hdul[0].header
                    
                    # Extract temperature if available
                    temp_c = None
                    for temp_key in ['CCD-TEMP', 'TEMP', 'CCDTEMP', 'TEMPERAT']:
                        if temp_key in header:
                            temp_val = header[temp_key]
                            if isinstance(temp_val, str):
                                try:
                                    # Try to extract numeric part from string like "20.0 C"
                                    temp_c = float(temp_val.split()[0])
                                except:
                                    continue
                            else:
                                temp_c = float(temp_val)
                            break
                    
                    # If no temperature found, use directory name as fallback
                    if temp_c is None:
                        dir_name = os.path.basename(os.path.dirname(light_file))
                        if dir_name.startswith('temp_'):
                            try:
                                temp_k = int(dir_name.split('_')[1])
                                temp_c = temp_k - 273.15
                            except:
                                temp_c = 20.0  # Default temperature
                        else:
                            temp_c = 20.0  # Default temperature
                    
                    # Get Bayer pattern if available
                    pattern = header.get('BAYERPAT', 'RGGB').strip()
                    
                    # Get appropriate dark frame
                    dark_data, msg = dark_manager.get_dark_frame(temp_c, pattern, light_data)
                    
                    # Perform dark subtraction
                    calibrated_data = light_data - dark_data
                    
                    # Floor at zero to avoid negative values
                    calibrated_data = np.maximum(calibrated_data, 0)
                    
                    # Save calibrated frame
                    output_file = os.path.join(calibrated_dir, basename)
                    calibrated_hdu = fits.PrimaryHDU(calibrated_data, header=header)
                    calibrated_hdu.header['CALSTAT'] = 'D'  # Mark as dark-subtracted
                    calibrated_hdu.header['DARKSUB'] = True
                    calibrated_hdu.header['DARKTEMP'] = temp_c
                    calibrated_hdu.writeto(output_file, overwrite=True)
                    
                    successful_calibrations += 1
                    
            except Exception as e:
                print(f"\nError calibrating {basename}: {str(e)}")
                # Copy original file if calibration fails
                output_file = os.path.join(calibrated_dir, basename)
                shutil.copy(light_file, output_file)
                failed_calibrations += 1
        
        # Report calibration results
        print(f"\nCalibration summary:")
        print(f"  - Successfully calibrated: {successful_calibrations} frames")
        if failed_calibrations > 0:
            print(f"  - Failed calibration: {failed_calibrations} frames (using originals)")
        
        print("\nStarting quality analysis on calibrated frames...")
        
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

def main():
    parser = argparse.ArgumentParser(description='Preprocess astronomical images with dark subtraction and analyze for stacking quality')
    
    parser.add_argument('input_dir', help='Directory containing light frames')
    parser.add_argument('--dark-dir', '-d', help='Base directory with dark frames in temp_XXX subdirectories')
    parser.add_argument('--output', '-o', help='Directory for output results', default='quality_results')
    parser.add_argument('--pattern', '-p', help='File pattern for FITS files', default='*.fits')
    parser.add_argument('--skip-calibration', '-s', action='store_true', help='Skip dark subtraction')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return 1
    
    # Validate dark directory if calibration is needed
    if not args.skip_calibration and (args.dark_dir is None or not os.path.isdir(args.dark_dir)):
        print(f"Error: Dark directory not specified or not found.")
        print("Use --skip-calibration to analyze without dark subtraction.")
        return 1
    
    try:
        preprocess_and_analyze(
            input_dir=args.input_dir,
            dark_base_dir=args.dark_dir,
            output_dir=args.output,
            pattern=args.pattern,
            skip_calibration=args.skip_calibration
        )
        return 0
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
