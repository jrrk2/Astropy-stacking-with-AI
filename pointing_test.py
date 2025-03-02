import sys
import os
from astro_stacking_script import AstroStackProcessor

def test_pointing_accuracy():
    """
    Run pointing accuracy analysis on astronomical images
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
    
    # Analyze pointing accuracy
    print("\nAnalyzing pointing accuracy...")
    processor.analyze_pointing_accuracy(filtered_frames, output_csv='pointing_accuracy.csv')
    
    print("\nPointing accuracy analysis complete!")

if __name__ == "__main__":
    test_pointing_accuracy()
