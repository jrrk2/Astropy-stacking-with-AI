import sys
import os
from astro_stacking_script import AstroStackProcessor

def test_alignment_methods():
    """
    Run a test to compare star-based and FFT-based alignment methods
    """
    # Adjust these paths to match your setup
    csv_path = './quality_results/quality_metrics.csv'
    fits_directory = './lights/temp_290/'
    
    # Initialize processor
    print("Initializing Stacking Processor...")
    processor = AstroStackProcessor(csv_path, fits_directory)
    
    # Filter frames
    print("\nFiltering frames...")
    filtered_frames = processor.filter_frames()
    
    # Run alignment validation
    print("\nValidating alignment methods...")
    processor.validate_alignment(filtered_frames)
    
    print("\nAlignment validation complete!")

    # Validate WCS headers
    print("\nValidating WCS headers...")
    processor.validate_wcs_headers(filtered_frames)
    
    print("\nWCS header validation complete!")

if __name__ == "__main__":
    test_alignment_methods()
