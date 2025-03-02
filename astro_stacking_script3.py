import numpy as np
import pandas as pd
import os
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats
from skimage.transform import AffineTransform, warp
from photutils import StarFinder

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
            'use_sigma_clipping': True     # Use sigma clipping during stacking
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

    def filter_frames(self):
        """
        Adaptive frame filtering with progressive quality relaxation
        
        Returns:
        --------
        filtered_frames : list
            List of filenames meeting quality criteria
        """
        # Print initial dataset statistics
        print("Initial Dataset Quality Check:")
        print(f"Total Frames: {len(self.metrics_df)}")
        print(f"FWHM Range: {self.metrics_df['mean_fwhm'].min():.2f} - {self.metrics_df['mean_fwhm'].max():.2f}")
        print(f"SNR Range: {self.metrics_df['mean_snr'].min():.2f} - {self.metrics_df['mean_snr'].max():.2f}")
        print(f"Quality Score Range: {self.metrics_df['quality_score'].min():.2f} - {self.metrics_df['quality_score'].max():.2f}")
        
        # Progressive filtering stages
        filtering_stages = [
            {
                'name': 'Strict Filtering',
                'filters': {
                    'mean_fwhm': ('<=', self.quality_thresholds['fwhm']),
                    'mean_ellipticity': ('<=', self.quality_thresholds['ellipticity']),
                    'mean_snr': ('>=', self.quality_thresholds['snr']),
                    'quality_score': ('>=', self.quality_thresholds['quality_score'])
                }
            },
            {
                'name': 'Relaxed FWHM Filtering',
                'filters': {
                    'mean_fwhm': ('<=', self.quality_thresholds['fwhm'] * 1.5),
                    'mean_ellipticity': ('<=', self.quality_thresholds['ellipticity'] * 1.2),
                    'mean_snr': ('>=', self.quality_thresholds['snr'] * 0.9),
                    'quality_score': ('>=', self.quality_thresholds['quality_score'] * 0.9)
                }
            },
            {
                'name': 'Minimal Quality Filtering',
                'filters': {
                    'mean_snr': ('>=', self.quality_thresholds['snr'] * 0.7),
                    'quality_score': ('>=', self.quality_thresholds['quality_score'] * 0.7)
                }
            }
        ]
        
        # Try each filtering stage
        for stage in filtering_stages:
            # Construct the filter conditions
            filter_conditions = []
            for column, (comparison, threshold) in stage['filters'].items():
                if comparison == '<=':
                    condition = self.metrics_df[column] <= threshold
                elif comparison == '>=':
                    condition = self.metrics_df[column] >= threshold
                else:
                    raise ValueError(f"Unsupported comparison: {comparison}")
                filter_conditions.append(condition)
            
            # Combine conditions
            combined_condition = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_condition &= condition
            
            # Apply filtering
            filtered_df = self.metrics_df[combined_condition]
            
            # Print stage results
            print(f"\n{stage['name']}:")
            print(f"Frames meeting criteria: {len(filtered_df)}")
            
            # If frames found, return their filenames
            if len(filtered_df) > 0:
                # Sort by quality score and limit to max frames
                filtered_df = filtered_df.nlargest(
                    min(len(filtered_df), self.stacking_config['max_frames_to_stack']), 
                    'quality_score'
                )
                
                print("Selected frames:")
                for _, row in filtered_df.head().iterrows():
                    print(f"  {row['filename']} (FWHM: {row['mean_fwhm']:.2f}, SNR: {row['mean_snr']:.2f}, Quality: {row['quality_score']:.2f})")
                
                return filtered_df['filename'].tolist()
        
        # If no frames found in any stage
        print("\nWARNING: No frames meet even minimal quality criteria!")
        print("Selecting top frames by quality score")
        
        # Fallback: select top frames by quality score
        fallback_frames = self.metrics_df.nlargest(
            min(20, len(self.metrics_df)), 
            'quality_score'
        )
        
        print("Fallback frames:")
        for _, row in fallback_frames.iterrows():
            print(f"  {row['filename']} (FWHM: {row['mean_fwhm']:.2f}, SNR: {row['mean_snr']:.2f}, Quality: {row['quality_score']:.2f})")
        
        return fallback_frames['filename'].tolist()

    def select_reference_frame(self, filtered_frames):
        """
        Select the reference frame for alignment
        
        Parameters:
        -----------
        filtered_frames : list
            List of frames to choose from
        
        Returns:
        --------
        reference_frame : ndarray
            Reference frame data
        """
        # Select the highest quality frame as reference
        reference_filename = filtered_frames[0]
        reference_path = os.path.join(self.fits_directory, reference_filename)
        
        self.reference_frame = fits.getdata(reference_path)
        return self.reference_frame

    def detect_alignment_stars(self, frame, n_stars=25):
        """
        Detect stars for alignment
        
        Parameters:
        -----------
        frame : ndarray
            Input image frame
        n_stars : int, optional
            Target number of stars to detect
        
        Returns:
        --------
        stars : Table
            Detected star coordinates
        """
        # Calculate background statistics
        mean, median, std = sigma_clipped_stats(frame)
        
        # Use StarFinder to detect stars
        finder = StarFinder(
            fwhm=3.0,
            threshold=5*std
        )
        stars = finder(frame - median)
        
        # Sort stars by flux and select top n_stars
        if stars is not None and len(stars) > 0:
            stars_sorted = stars[np.argsort(stars['flux'])[::-1]]
            return stars_sorted[:n_stars]
        
        return None

    def align_frame(self, frame, ref_stars, target_stars):
        """
        Align a single frame to the reference frame
        
        Parameters:
        -----------
        frame : ndarray
            Input frame to align
        ref_stars : Table
            Reference stars
        target_stars : Table
            Target frame stars
        
        Returns:
        --------
        aligned_frame : ndarray
            Geometrically aligned frame
        """
        # Check for valid star detection
        if ref_stars is None or target_stars is None or len(ref_stars) == 0 or len(target_stars) == 0:
            print("Warning: Insufficient stars for alignment. Returning original frame.")
            return frame
        
        # Extract star coordinates
        ref_coords = np.column_stack((ref_stars['x'], ref_stars['y']))
        target_coords = np.column_stack((target_stars['x'], target_stars['y']))
        
        # Compute affine transformation
        tform = AffineTransform()
        tform.estimate(target_coords, ref_coords)
        
        # Apply transformation
        aligned_frame = warp(
            frame, 
            inverse_map=tform.inverse,
            output_shape=self.reference_frame.shape
        )
        
        return aligned_frame

    def progressive_stack(self, filtered_frames):
        """
        Progressively stack frames while monitoring quality
        
        Parameters:
        -----------
        filtered_frames : list
            List of frames to stack
        
        Returns:
        --------
        stacked_image : ndarray
            Final stacked image
        """
        # Select reference frame and detect its stars
        self.reference_frame = self.select_reference_frame(filtered_frames)
        ref_stars = self.detect_alignment_stars(self.reference_frame)
        
        # Initialize stacked image with reference frame
        stacked_image = self.reference_frame.astype(np.float32)
        frames_stacked = 1
        
        # Track quality metrics of reference frame
        current_snr = self.metrics_df.loc[self.metrics_df['filename'] == filtered_frames[0], 'mean_snr'].values[0]
        
        # Iterate through remaining frames
        for frame_name in filtered_frames[1:]:
            # Read frame
            frame_path = os.path.join(self.fits_directory, frame_name)
            current_frame = fits.getdata(frame_path)
            
            # Detect stars in current frame
            target_stars = self.detect_alignment_stars(current_frame)
            
            # Align frame
            aligned_frame = self.align_frame(current_frame, ref_stars, target_stars)
            
            # Check quality improvement
            new_snr = self.metrics_df.loc[self.metrics_df['filename'] == frame_name, 'mean_snr'].values[0]
            
            # Progressively stack if quality improves or remains stable
            if new_snr >= current_snr * 0.9:
                stacked_image += aligned_frame
                frames_stacked += 1
                current_snr = (current_snr * (frames_stacked - 1) + new_snr) / frames_stacked
            else:
                print(f"Stopping stacking: Quality degradation detected at {frame_name}")
                break
        
        # Normalize the stacked image
        self.stacked_image = stacked_image / frames_stacked
        return self.stacked_image

    def save_stacked_image(self, output_path):
        """
        Save the stacked image as a FITS file
        
        Parameters:
        -----------
        output_path : str
            Path to save the stacked image
        """
        if self.stacked_image is None:
            raise ValueError("No stacked image available. Run progressive_stack first.")
        
        # Create a new FITS HDU with the stacked image
        hdu = fits.PrimaryHDU(self.stacked_image)
        hdu.writeto(output_path, overwrite=True)
        print(f"Stacked image saved to {output_path}")

    def process(self, output_path='stacked_image.fits'):
        """
        Full processing pipeline
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the stacked image
        """
        # Filter frames based on quality
        filtered_frames = self.filter_frames()
        print(f"Frames to be stacked: {len(filtered_frames)}")
        
        # Perform progressive stacking
        self.progressive_stack(filtered_frames)
        
        # Save stacked image
        self.save_stacked_image(output_path)

# Example usage
if __name__ == "__main__":
    # Adjust these paths as needed
    csv_path = './quality_results/quality_metrics.csv'
    fits_directory = './fits_frames/'
    
    # Initialize and run stacking processor
    processor = AstroStackProcessor(csv_path, fits_directory)
    processor.process(output_path='stacked_astronomical_image.fits')
