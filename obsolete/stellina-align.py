#!/usr/bin/env python3
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
import astroalign as aa
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from tqdm import tqdm
import matplotlib.pyplot as plt

class EnhancedStellinaAligner:
    def __init__(self, pattern='img-*.fits', max_stars=100, min_stars=30):
        """
        Enhanced alignment specifically for Stellina data
        
        Args:
            pattern: Glob pattern for input files
            max_stars: Maximum number of stars to use for alignment
            min_stars: Minimum stars required for reliable alignment
        """
        self.max_stars = max_stars
        self.min_stars = min_stars
        self.reference_frame = None
        self.star_positions = {}
        
    def detect_stars(self, data, detection_sigma=2.0):
        """Enhanced star detection with validation"""
        # Estimate background
        sigma_clip = SigmaClip(sigma=3.0)
        bkg = Background2D(
            data, 
            (50, 50),
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=MedianBackground()
        )
        
        data_sub = data - bkg.background
        mean, median, std = sigma_clipped_stats(data_sub, sigma=3.0)
        
        # Initial star detection
        daofind = DAOStarFinder(
            fwhm=3.0,
            threshold=detection_sigma*std,
            sharplo=0.2,  # More stringent shape criteria
            sharphi=0.9,
            roundlo=-0.5,
            roundhi=0.5
        )
        
        stars = daofind(data_sub)
        if stars is None:
            return None
            
        # Sort by flux and get brightest stars
        stars.sort('peak', reverse=True)
        if len(stars) > self.max_stars:
            stars = stars[:self.max_stars]
            
        # Create star map for visualization
        star_map = np.zeros_like(data, dtype=bool)
        star_map[stars['ycentroid'].astype(int), stars['xcentroid'].astype(int)] = True
        
        return stars, star_map
        
    def process_single_frame(self, filename):
        """Process a single frame with enhanced debayering"""
        with fits.open(filename) as hdul:
            data = hdul[0].data.astype(np.float32)
            if 'BZERO' in hdul[0].header:
                data = data + hdul[0].header['BZERO']
            
            # Get Bayer pattern
            pattern = hdul[0].header.get('BAYERPAT', 'RGGB').strip().replace("'", "")
            
            # Debayer
            rgb = demosaicing_CFA_Bayer_bilinear(data, pattern)
            
            # Convert to luminance for alignment
            lum = 0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]
            
            return lum, rgb
            
    def align_frames(self, files, output_base='aligned'):
        """
        Align frames with enhanced validation and visualization
        """
        # Process first frame as reference
        print("Processing reference frame...")
        ref_lum, ref_rgb = self.process_single_frame(files[0])
        ref_stars, ref_map = self.detect_stars(ref_lum)
        
        if ref_stars is None or len(ref_stars) < self.min_stars:
            raise ValueError("Could not detect enough stars in reference frame")
            
        # Save reference star map for validation
        plt.figure(figsize=(12, 8))
        plt.imshow(ref_lum, cmap='gray')
        plt.plot(ref_stars['xcentroid'], ref_stars['ycentroid'], 'r+')
        plt.title(f'Reference frame stars: {len(ref_stars)}')
        plt.savefig(f'{output_base}_refstars.png')
        plt.close()
        
        # Initialize accumulators for each channel
        aligned_r = []
        aligned_g = []
        aligned_b = []
        
        # Process remaining frames
        for idx, filename in enumerate(tqdm(files)):
            try:
                lum, rgb = self.process_single_frame(filename)
                stars, _ = self.detect_stars(lum)
                
                if stars is None or len(stars) < self.min_stars:
                    print(f"Warning: Frame {filename} has too few stars, skipping")
                    continue
                    
                # Compute transform with increased control points
                transform, (source_pts, target_pts) = aa.find_transform(
                    lum, 
                    ref_lum,
                    max_control_points=self.max_stars,
                )
                
                # Apply transform to each channel
                aligned_frame_r = aa.apply_transform(transform, rgb[:,:,0], ref_lum)[0]
                aligned_frame_g = aa.apply_transform(transform, rgb[:,:,1], ref_lum)[0]
                aligned_frame_b = aa.apply_transform(transform, rgb[:,:,2], ref_lum)[0]
                
                # Validate alignment quality
                if idx % 20 == 0:  # Periodically check alignment
                    aligned_stars, _ = self.detect_stars(aligned_frame_g)  # Use green channel
                    if aligned_stars is not None:
                        plt.figure(figsize=(12, 8))
                        plt.subplot(121)
                        plt.imshow(lum, cmap='gray')
                        plt.plot(stars['xcentroid'], stars['ycentroid'], 'r+')
                        plt.title('Before alignment')
                        
                        plt.subplot(122)
                        plt.imshow(aligned_frame_g, cmap='gray')
                        plt.plot(aligned_stars['xcentroid'], aligned_stars['ycentroid'], 'g+')
                        plt.title('After alignment')
                        
                        plt.savefig(f'{output_base}_check_{idx:04d}.png')
                        plt.close()
                
                aligned_r.append(aligned_frame_r)
                aligned_g.append(aligned_frame_g)
                aligned_b.append(aligned_frame_b)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        # Stack aligned frames
        print("\nStacking aligned frames...")
        stacked_r = np.median(aligned_r, axis=0)
        stacked_g = np.median(aligned_g, axis=0)
        stacked_b = np.median(aligned_b, axis=0)
        
        # Save results
        print("Saving stacked channels...")
        fits.writeto('stacked_r.fits', stacked_r, overwrite=True)
        fits.writeto('stacked_g.fits', stacked_g, overwrite=True)
        fits.writeto('stacked_b.fits', stacked_b, overwrite=True)
        
        # Create preview
        rgb_preview = np.stack([stacked_r, stacked_g, stacked_b], axis=2)
        rgb_preview = np.clip(rgb_preview / np.percentile(rgb_preview, 99), 0, 1)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(rgb_preview)
        plt.title('Stacked Result')
        plt.savefig(f'{output_base}_final.png')
        plt.close()

if __name__ == "__main__":
    import glob
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Stellina frame alignment')
    parser.add_argument('pattern', help='Input file pattern (e.g., "img-*.fits")')
    parser.add_argument('--max-stars', type=int, default=100,
                        help='Maximum number of stars to use for alignment')
    parser.add_argument('--min-stars', type=int, default=30,
                        help='Minimum stars required for reliable alignment')
    parser.add_argument('--output', default='aligned',
                        help='Base name for output files')
    
    args = parser.parse_args()
    
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files found matching pattern: {args.pattern}")
        exit(1)
        
    print(f"Found {len(files)} files to process")
    
    aligner = EnhancedStellinaAligner(
        max_stars=args.max_stars,
        min_stars=args.min_stars
    )
    
    aligner.align_frames(files, args.output)
