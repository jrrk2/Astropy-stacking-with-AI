import numpy as np
import pandas as pd
import os
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats
from skimage.transform import AffineTransform, warp
from photutils.detection import DAOStarFinder
from photutils.centroids import centroid_2dg

class AstroStackProcessor:
    def __init__(self, csv_path, fits_directory):
        """
        Initialize the stacking processor
        
        Parameters:
        -----------
        csv_path : str
            Path to the quality metrics CSV file
        fits_directory : str
            Directory containing FITS files
        """
        # Load quality metrics
        self.metrics_df = pd.read_csv(csv_path)
        
        # Compute dataset characteristics
        fwhm_max = self.metrics_df['mean_fwhm'].max()
        fwhm_median = self.metrics_df['mean_fwhm'].median()
        
        # Set quality thresholds based on dataset characteristics
        self.quality_thresholds = {
            'fwhm': min(10.73, fwhm_max),  # Use best FWHM or dataset max
            'ellipticity': 1.50,
            'snr': max(11.96, self.metrics_df['mean_snr'].quantile(0.25)),
            'quality_score': max(30.60, self.metrics_df['quality_score'].quantile(0.25))
        }
        
        # Additional configuration for flexible stacking
        self.stacking_config = {
            'fwhm_relaxation_factor': 1.2,  # Allow 20% relaxation
            'snr_relaxation_factor': 0.8,   # Minimum acceptable SNR
            'max_frames_to_stack': 50,      # Limit number of frames to prevent memory issues
            'normalize_by_exposure': True,  # Normalize frames by exposure time
            'use_sigma_clipping': True,     # Use sigma clipping during stacking
            'star_detection_threshold': 5   # Sigma threshold for star detection
        }
        
        # Sort frames by quality score in descending order
        self.metrics_df = self.metrics_df.sort_values('quality_score', ascending=False)
        
        # FITS directory
        self.fits_directory = fits_directory
        
        # Stacking parameters
        self.reference_frame = None
        self.stacked_image = None
        
        # Print initialization details
        print("Stacking Processor Initialized:")
        print(f"Total Frames: {len(self.metrics_df)}")
        print("Quality Thresholds:")
        for key, value in self.quality_thresholds.items():
            print(f"  {key}: {value}")

    def detect_alignment_stars(self, frame, n_stars=25):
        """
        Detect stars for alignment using DAOStarFinder
        
        Parameters:
        -----------
        frame : ndarray
            Input image frame
        n_stars : int, optional
            Target number of stars to detect
        
        Returns:
        --------
        stars : astropy.table.Table or None
            Detected star coordinates
        """
        # Calculate background statistics
        mean, median, std = sigma_clipped_stats(frame)
        
        # Use DAOStarFinder for star detection
        try:
            # Estimate FWHM from dataset metrics or use a default
            est_fwhm = self.quality_thresholds.get('fwhm', 3.0)
            
            # Create star finder
            daofind = DAOStarFinder(
                fwhm=est_fwhm, 
                threshold=self.stacking_config['star_detection_threshold'] * std
            )
            
            # Detect stars
            sources = daofind(frame - median)
            
            if sources is None or len(sources) == 0:
                print("Warning: No stars detected in frame.")
                return None
            
            # Refine star centroids
            for source in sources:
                x, y = centroid_2dg(frame, (source['ycentroid'], source['xcentroid']))
                source['xcentroid'] = x
                source['ycentroid'] = y
            
            # Sort by flux and select top n_stars
            sources.sort('flux', reverse=True)
            return sources[:n_stars]
        
        except Exception as e:
            print(f"Error in star detection: {e}")
            return None

    # ... [rest of the previous implementation remains the same]

# Example usage
if __name__ == "__main__":
    # Adjust these paths as needed
    csv_path = './quality_results/quality_metrics.csv'
    fits_directory = './fits_frames/'
    
    # Initialize and run stacking processor
    processor = AstroStackProcessor(csv_path, fits_directory)
    processor.process(output_path='stacked_astronomical_image.fits')
