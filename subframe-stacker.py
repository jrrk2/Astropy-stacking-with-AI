#!/usr/bin/env python3
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
from tqdm import tqdm

class SubframeStacker:
    def __init__(self, input_dir, output_file="stacked_image.fits"):
        """
        Initialize the subframe stacker.
        
        Parameters:
        -----------
        input_dir : str
            Directory containing the FITS files to stack
        output_file : str
            Name of the output stacked FITS file
        """
        self.input_dir = input_dir
        self.output_file = output_file
        self.fits_files = []
        self.frames = []
        self.aligned_frames = []
        self.reference_frame = None
        self.reference_stars = None
        
    def load_files(self):
        """
        Load all FITS files from the input directory.
        """
        self.fits_files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) 
                          if f.endswith(('.fits', '.fit', '.FITS', '.FIT'))]
        
        if not self.fits_files:
            raise ValueError(f"No FITS files found in {self.input_dir}")
        
        print(f"Found {len(self.fits_files)} FITS files to process")
        
    def load_frames(self):
        """
        Load the data from all FITS files into memory.
        """
        self.frames = []
        for file in tqdm(self.fits_files, desc="Loading frames"):
            with fits.open(file) as hdul:
                # Get the main image data
                data = hdul[0].data
                
                # Handle different dimensions
                if len(data.shape) == 3:
                    # RGB or multiple HDUs
                    if data.shape[0] == 3:
                        # Convert RGB to grayscale using luminance formula
                        data = 0.2989 * data[0] + 0.5870 * data[1] + 0.1140 * data[2]
                    else:
                        data = data[0]  # Just use the first frame
                
                # Store header for later use
                header = hdul[0].header
                
                self.frames.append({'data': data, 'header': header, 'file': file})
        
        if not self.frames:
            raise ValueError("No valid frames were loaded")
        
        # Select the frame with the highest mean value as the reference frame
        mean_values = [np.mean(frame['data']) for frame in self.frames]
        self.reference_frame = self.frames[np.argmax(mean_values)]
        print(f"Selected {os.path.basename(self.reference_frame['file'])} as reference frame")
    
    def detect_stars(self, image_data, threshold=5.0, fwhm=3.0):
        """
        Detect stars in an image using DAOStarFinder.
        
        Parameters:
        -----------
        image_data : numpy.ndarray
            Image data
        threshold : float
            Detection threshold in standard deviations
        fwhm : float
            Full-width half-maximum of the PSF
            
        Returns:
        --------
        stars : Table
            Table of detected stars
        """
        # Calculate background statistics using sigma clipping
        mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
        
        # Subtract background
        image_data_sub = image_data - median
        
        # Find stars
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
        stars = daofind(image_data_sub)
        
        if stars is None or len(stars) < 3:
            # If too few stars found, try lowering the threshold
            daofind = DAOStarFinder(fwhm=fwhm, threshold=3.0*std)
            stars = daofind(image_data_sub)
        
        return stars
    
    def align_frames_using_stars(self):
        """
        Align frames by matching star patterns.
        """
        print("Aligning frames using star detection...")
        
        # Detect stars in the reference frame
        self.reference_stars = self.detect_stars(self.reference_frame['data'])
        
        if self.reference_stars is None or len(self.reference_stars) < 3:
            print("Not enough stars detected in reference frame. Falling back to phase correlation.")
            return self.align_frames_using_phase_correlation()
        
        # Prepare the aligned frames list
        self.aligned_frames = []
        
        # The reference frame is already aligned
        self.aligned_frames.append(self.reference_frame['data'])
        
        # Process each frame and align it to the reference
        for i, frame in enumerate(tqdm(self.frames, desc="Aligning frames")):
            if frame['file'] == self.reference_frame['file']:
                continue  # Skip reference frame
            
            # Detect stars in this frame
            stars = self.detect_stars(frame['data'])
            
            if stars is None or len(stars) < 3:
                print(f"Not enough stars detected in frame {i}. Using phase correlation.")
                # If star detection fails, use phase correlation for this frame
                shift_vector = self.get_shift_using_phase_correlation(self.reference_frame['data'], frame['data'])
                aligned_data = shift(frame['data'], shift=shift_vector, mode='constant', cval=0)
                self.aligned_frames.append(aligned_data)
                continue
            
            # Find transformation between star sets
            # For simplicity, we'll just find the average shift between the brightest stars
            # A more robust method would use a matching algorithm like Triangle Matching
            
            # Sort stars by flux in both frames
            ref_brightest = self.reference_stars[np.argsort(self.reference_stars['flux'])[-10:]]
            frame_brightest = stars[np.argsort(stars['flux'])[-10:]]
            
            # Limit to the smaller number of stars
            min_stars = min(len(ref_brightest), len(frame_brightest))
            if min_stars < 3:
                min_stars = 3
            
            # Calculate average shift
            dx = np.mean(ref_brightest['xcentroid'][:min_stars]) - np.mean(frame_brightest['xcentroid'][:min_stars])
            dy = np.mean(ref_brightest['ycentroid'][:min_stars]) - np.mean(frame_brightest['ycentroid'][:min_stars])
            
            # Apply the shift to align the frame
            aligned_data = shift(frame['data'], shift=(dy, dx), mode='constant', cval=0)
            self.aligned_frames.append(aligned_data)
    
    def get_shift_using_phase_correlation(self, ref_image, image):
        """
        Calculate the shift between two images using phase cross-correlation.
        
        Parameters:
        -----------
        ref_image : numpy.ndarray
            Reference image
        image : numpy.ndarray
            Image to align
            
        Returns:
        --------
        shift_vector : tuple
            (y, x) shift vector
        """
        # Ensure images have same shape
        if ref_image.shape != image.shape:
            raise ValueError("Images must have the same shape for phase correlation")
            
        # Calculate shift using phase cross-correlation
        shift_vector, error, _ = phase_cross_correlation(ref_image, image, upsample_factor=10)
        
        return shift_vector
    
    def align_frames_using_phase_correlation(self):
        """
        Align frames using phase cross-correlation.
        """
        print("Aligning frames using phase correlation...")
        
        # Prepare the aligned frames list
        self.aligned_frames = []
        
        # The reference frame is already aligned
        self.aligned_frames.append(self.reference_frame['data'])
        
        # Process each frame and align it to the reference
        for frame in tqdm(self.frames, desc="Aligning frames"):
            if frame['file'] == self.reference_frame['file']:
                continue  # Skip reference frame
                
            # Calculate shift using phase cross-correlation
            shift_vector = self.get_shift_using_phase_correlation(self.reference_frame['data'], frame['data'])
            
            # Apply the shift to align the frame
            aligned_data = shift(frame['data'], shift=shift_vector, mode='constant', cval=0)
            self.aligned_frames.append(aligned_data)
    
    def stack_frames(self, method='median'):
        """
        Stack the aligned frames using the specified method.
        
        Parameters:
        -----------
        method : str
            Stacking method: 'mean', 'median', or 'sum'
        """
        if not self.aligned_frames:
            raise ValueError("No aligned frames available. Run align_frames first.")
        
        print(f"Stacking {len(self.aligned_frames)} frames using {method} method...")
        
        # Convert list to a 3D array
        stack = np.array(self.aligned_frames)
        
        # Apply stacking method
        if method == 'mean':
            stacked_image = np.mean(stack, axis=0)
        elif method == 'median':
            stacked_image = np.median(stack, axis=0)
        elif method == 'sum':
            stacked_image = np.sum(stack, axis=0)
        else:
            raise ValueError(f"Unknown stacking method: {method}")
        
        # Save result to FITS file
        hdu = fits.PrimaryHDU(stacked_image, header=self.reference_frame['header'])
        hdu.writeto(self.output_file, overwrite=True)
        
        print(f"Stacked image saved to {self.output_file}")
        
        return stacked_image
    
    def visualize_result(self, stacked_image):
        """
        Visualize the result by showing the first frame and the stacked image.
        
        Parameters:
        -----------
        stacked_image : numpy.ndarray
            Stacked image data
        """
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Normalize the images for better visualization
        norm1 = ImageNormalize(self.frames[0]['data'], stretch=SqrtStretch())
        norm2 = ImageNormalize(stacked_image, stretch=SqrtStretch())
        
        # Display the first frame
        ax1.imshow(self.frames[0]['data'], origin='lower', norm=norm1, cmap='viridis')
        ax1.set_title('First Frame')
        
        # Display the stacked image
        ax2.imshow(stacked_image, origin='lower', norm=norm2, cmap='viridis')
        ax2.set_title('Stacked Image')
        
        plt.tight_layout()
        plt.savefig('stacking_comparison.png')
        plt.show()
    
    def process(self, align_method='stars', stack_method='median', visualize=True):
        """
        Process all frames: load, align, stack, and optionally visualize.
        
        Parameters:
        -----------
        align_method : str
            Alignment method: 'stars' or 'phase'
        stack_method : str
            Stacking method: 'mean', 'median', or 'sum'
        visualize : bool
            Whether to visualize the results
        """
        # Load FITS files
        self.load_files()
        
        # Load frames
        self.load_frames()
        
        # Align frames
        if align_method == 'stars':
            self.align_frames_using_stars()
        elif align_method == 'phase':
            self.align_frames_using_phase_correlation()
        else:
            raise ValueError(f"Unknown alignment method: {align_method}")
        
        # Stack frames
        stacked_image = self.stack_frames(method=stack_method)
        
        # Visualize if requested
        if visualize:
            self.visualize_result(stacked_image)
        
        return stacked_image


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stack astronomical images.')
    parser.add_argument('input_dir', help='Directory containing FITS files to stack')
    parser.add_argument('--output', '-o', default='stacked_image.fits', help='Output FITS file')
    parser.add_argument('--align', '-a', choices=['stars', 'phase'], default='stars', 
                        help='Alignment method: stars or phase correlation')
    parser.add_argument('--stack', '-s', choices=['mean', 'median', 'sum'], default='median',
                        help='Stacking method: mean, median, or sum')
    parser.add_argument('--no-vis', dest='visualize', action='store_false',
                        help='Disable visualization')
    
    args = parser.parse_args()
    
    # Create and run the stacker
    stacker = SubframeStacker(args.input_dir, args.output)
    stacker.process(align_method=args.align, stack_method=args.stack, visualize=args.visualize)
