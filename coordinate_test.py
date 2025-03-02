import sys
import os
import argparse
from enhanced_coordinate_conversion import analyze_coordinate_accuracy

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Analyze coordinate transformation accuracy')
    parser.add_argument('--directory', '-d', type=str, default='./lights/temp_290/',
                        help='Directory containing FITS files')
    parser.add_argument('--output', '-o', type=str, default='coordinate_analysis.csv',
                        help='Output CSV file')
    parser.add_argument('--lat', type=float, default=52.24509819278444,
                        help='Observer latitude in degrees')
    parser.add_argument('--long', type=float, default=0.0795527475486776,
                        help='Observer longitude in degrees')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run analysis
    print(f"Analyzing coordinate transformation accuracy for files in {args.directory}")
    print(f"Using location: LAT={args.lat}, LONG={args.long}")
    
    # Perform the analysis
    analyze_coordinate_accuracy(
        args.directory,
        args.output,
        args.lat,
        args.long
    )
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
