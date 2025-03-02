#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astronomical Image Quality Assessment Script for Stacking
--------------------------------------------------------
This script analyzes FITS files to extract quality metrics for optimal stacking.
It measures FWHM, SNR, star shape analysis, and can track quality across frames.

Usage:
    python stacking_quality_assessment.py /path/to/fits/files [options]

Examples:
    # Basic usage with default parameters
    python stacking_quality_assessment.py /path/to/fits/files

    # Specify an output directory and a specific file pattern
    python stacking_quality_assessment.py /path/to/fits/files -o /path/to/output -p "img_*.fits"

Dependencies:
    - astropy
    - numpy
    - pandas
    - matplotlib
    - photutils
    - scipy
    - scikit-learn (optional, for clustering)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import warnings
import time
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, gaussian_sigma_to_fwhm
from astropy.table import Table
from astropy.visualization import simple_norm
from astropy.modeling import models, fitting
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.spatial import KDTree
from scipy import ndimage
from datetime import datetime
import matplotlib.gridspec as gridspec

# Suppress some common warnings
warnings.filterwarnings("ignore", category=Warning, module="astropy.modeling.fitting")
warnings.filterwarnings("ignore", category=UserWarning, module="astropy.io.fits")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, gaussian_sigma_to_fwhm
from astropy.table import Table
from astropy.visualization import simple_norm
from astropy.modeling import models, fitting
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.spatial import KDTree
from scipy import ndimage
from datetime import datetime
import matplotlib.gridspec as gridspec

class ImageQualityAnalyzer:
    """Analyzes astronomical images for quality assessment prior to stacking.
    
    This class processes FITS files to extract various quality metrics including:
    - FWHM (Full Width at Half Maximum) of stars
    - Star shape (ellipticity)
    - Signal-to-Noise Ratio (SNR)
    - Background statistics
    - Number of detectable stars
    - Registration accuracy
    
    It calculates a comprehensive quality score and generates recommendations
    for optimal stacking.
    """
    
    def __init__(self, input_dir, output_dir=None, file_pattern='*.fits'):
        """
        Initialize the analyzer with input directory and file pattern.
        
        Parameters:
        -----------
        input_dir : str
            Directory containing FITS files
        output_dir : str, optional
            Directory for saving results (default is input_dir/quality_assessment)
        file_pattern : str, optional
            Glob pattern for finding FITS files (default is *.fits)
        """
        self.input_dir = input_dir
        if output_dir is None:
            self.output_dir = os.path.join(input_dir, 'quality_assessment')
        else:
            self.output_dir = output_dir
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
        self.num_files = len(self.files)
        if self.num_files == 0:
            raise ValueError(f"No files found matching {file_pattern} in {input_dir}")
            
        print(f"Found {self.num_files} files for analysis")
        
        # DataFrame to store quality metrics
        self.quality_df = pd.DataFrame(index=range(self.num_files),
                                      columns=['filename', 'alt', 'az', 'ra', 'dec', 
                                               'mean_fwhm', 'median_fwhm', 'min_fwhm', 'max_fwhm',
                                               'mean_ellipticity', 'median_ellipticity',
                                               'background_mean', 'background_std', 
                                               'num_stars', 'mean_snr', 'median_snr',
                                               'quality_score', 'datetime'])
        
        # For storing reference stars across frames
        self.reference_stars = None
        self.reference_frame_idx = None
        
    def extract_metadata(self, header):
        """
        Extract relevant metadata from FITS header.
        
        Parameters:
        -----------
        header : fits.Header
            FITS header to extract metadata from
            
        Returns:
        --------
        dict
            Dictionary with extracted metadata
        """
        metadata = {}
        
        # Try to extract coordinates (ALT/AZ, RA/DEC)
        try:
            metadata['alt'] = header.get('ALT', header.get('ALTITUDE', np.nan))
            metadata['az'] = header.get('AZ', header.get('AZIMUTH', np.nan))
            metadata['ra'] = header.get('RA', header.get('OBJCTRA', np.nan))
            metadata['dec'] = header.get('DEC', header.get('OBJCTDEC', np.nan))
        except:
            metadata['alt'] = metadata['az'] = metadata['ra'] = metadata['dec'] = np.nan
            
        # Try to extract datetime
        try:
            date_str = header.get('DATE-OBS', header.get('DATE', None))
            if date_str:
                metadata['datetime'] = date_str
            else:
                metadata['datetime'] = None
        except:
            metadata['datetime'] = None
            
        return metadata
    
    def detect_stars(self, data, fwhm=3.0, threshold=5.0, exclude_border=10):
        """
        Detect stars in an image.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Image data
        fwhm : float, optional
            Estimated FWHM of stars in pixels
        threshold : float, optional
            Detection threshold in sigma above background
        exclude_border : int, optional
            Exclude sources within this many pixels of the border
            
        Returns:
        --------
        photutils.detection.DAOStarFinder.starlist
            Table of detected stars
        """
        # Calculate background statistics with sigma clipping
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
        # Use DAOStarFinder for star detection
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
        sources = daofind(data - median)
        
        if sources is None:
            return None
        
        # Filter out sources near the border
        if exclude_border > 0:
            height, width = data.shape
            good_sources = ((sources['xcentroid'] >= exclude_border) & 
                           (sources['xcentroid'] <= width - exclude_border) &
                           (sources['ycentroid'] >= exclude_border) &
                           (sources['ycentroid'] <= height - exclude_border))
            sources = sources[good_sources]
        
        return sources
    
    def fit_2d_gaussian(self, data, star_x, star_y, box_size=15):
        """
        Fit a 2D Gaussian to a star to measure FWHM and ellipticity.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Image data
        star_x, star_y : float
            Center coordinates of the star
        box_size : int, optional
            Size of the box around the star for fitting
            
        Returns:
        --------
        dict
            Dictionary with FWHM (x,y) and ellipticity
        """
        x, y = int(star_x), int(star_y)
        
        # Extract a cutout around the star
        half_box = box_size // 2
        y_min = max(0, y - half_box)
        y_max = min(data.shape[0], y + half_box + 1)
        x_min = max(0, x - half_box)
        x_max = min(data.shape[1], x + half_box + 1)
        
        if (y_max - y_min) < 5 or (x_max - x_min) < 5:
            return {'fwhm_x': np.nan, 'fwhm_y': np.nan, 'ellipticity': np.nan}
        
        star_data = data[y_min:y_max, x_min:x_max].copy()
        
        # Create coordinate arrays
        y_coords, x_coords = np.mgrid[:star_data.shape[0], :star_data.shape[1]]
        
        # Initialize model
        amplitude_init = np.max(star_data) - np.min(star_data)
        x_mean_init = half_box
        y_mean_init = half_box
        x_stddev_init = 2.0
        y_stddev_init = 2.0

        # Try to fit a 2D Gaussian
        try:
            # Initial Gaussian2D model
            model_init = models.Gaussian2D(amplitude=amplitude_init,
                                          x_mean=x_mean_init,
                                          y_mean=y_mean_init,
                                          x_stddev=x_stddev_init,
                                          y_stddev=y_stddev_init)

            fit_p = fitting.LevMarLSQFitter()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = fit_p(model_init, x_coords, y_coords, star_data)

            # Check if the fit converged reasonably
            if not np.isfinite(model.x_stddev.value) or not np.isfinite(model.y_stddev.value):
                return {'fwhm_x': np.nan, 'fwhm_y': np.nan, 'ellipticity': np.nan}

            if model.x_stddev.value < 0.5 or model.y_stddev.value < 0.5:
                return {'fwhm_x': np.nan, 'fwhm_y': np.nan, 'ellipticity': np.nan}

            if model.x_stddev.value > box_size/2 or model.y_stddev.value > box_size/2:
                return {'fwhm_x': np.nan, 'fwhm_y': np.nan, 'ellipticity': np.nan}

            # Calculate FWHM and ellipticity
            fwhm_x = model.x_stddev.value * gaussian_sigma_to_fwhm
            fwhm_y = model.y_stddev.value * gaussian_sigma_to_fwhm
            ellipticity = max(fwhm_x, fwhm_y) / min(fwhm_x, fwhm_y)

            return {'fwhm_x': fwhm_x, 'fwhm_y': fwhm_y, 'ellipticity': ellipticity}

        except Exception as e:
            return {'fwhm_x': np.nan, 'fwhm_y': np.nan, 'ellipticity': np.nan}        
    
    def measure_star_properties(self, data, sources):
        """
        Measure properties of detected stars.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Image data
        sources : astropy.table.Table
            Table of detected sources
            
        Returns:
        --------
        dict
            Dictionary with star properties (FWHM, ellipticity, etc.)
        """
        if sources is None or len(sources) == 0:
            return {
                'mean_fwhm': np.nan,
                'median_fwhm': np.nan,
                'min_fwhm': np.nan,
                'max_fwhm': np.nan,
                'mean_ellipticity': np.nan,
                'median_ellipticity': np.nan,
                'fwhm_values': [],
                'ellipticity_values': []
            }
        
        fwhm_values = []
        ellipticity_values = []
        
        # Measure FWHM and ellipticity for each star
        for source in sources:
            x, y = source['xcentroid'], source['ycentroid']
            props = self.fit_2d_gaussian(data, x, y)
            
            # Only keep valid measurements
            if not np.isnan(props['fwhm_x']) and not np.isnan(props['fwhm_y']):
                # Use average of X and Y FWHM as the overall FWHM
                fwhm = (props['fwhm_x'] + props['fwhm_y']) / 2
                fwhm_values.append(fwhm)
                ellipticity_values.append(props['ellipticity'])
        
        # Compute statistics if we have valid measurements
        if len(fwhm_values) > 0:
            mean_fwhm = np.mean(fwhm_values)
            median_fwhm = np.median(fwhm_values)
            min_fwhm = np.min(fwhm_values)
            max_fwhm = np.max(fwhm_values)
            mean_ellipticity = np.mean(ellipticity_values)
            median_ellipticity = np.median(ellipticity_values)
        else:
            mean_fwhm = median_fwhm = min_fwhm = max_fwhm = np.nan
            mean_ellipticity = median_ellipticity = np.nan
        
        return {
            'mean_fwhm': mean_fwhm,
            'median_fwhm': median_fwhm,
            'min_fwhm': min_fwhm,
            'max_fwhm': max_fwhm,
            'mean_ellipticity': mean_ellipticity,
            'median_ellipticity': median_ellipticity,
            'fwhm_values': fwhm_values,
            'ellipticity_values': ellipticity_values
        }
    
    def estimate_background(self, data, box_size=64, filter_size=3):
        """
        Estimate background and its statistics.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Image data
        box_size : int, optional
            Size of the box for background estimation
        filter_size : int, optional
            Size of the filter for background smoothing
            
        Returns:
        --------
        dict
            Dictionary with background statistics
        """
        try:
            # Create a background estimator
            bkg_estimator = MedianBackground()
            bkg = Background2D(data, box_size, filter_size=filter_size, 
                              bkg_estimator=bkg_estimator)
            
            # Get background statistics
            bkg_mean = bkg.background_median
            bkg_std = bkg.background_rms_median
            
            return {
                'background': bkg.background,
                'background_mean': bkg_mean,
                'background_std': bkg_std
            }
        except Exception as e:
            print(f"Error estimating background: {e}")
            # Fallback to simple statistics
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            return {
                'background': np.ones_like(data) * median,
                'background_mean': median,
                'background_std': std
            }
    
    def measure_snr(self, data, sources, background, aperture_radius=5.0, annulus_radii=(8.0, 12.0)):
        """
        Measure Signal-to-Noise Ratio for detected stars.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Image data
        sources : astropy.table.Table
            Table of detected sources
        background : numpy.ndarray
            Background image
        aperture_radius : float, optional
            Radius of aperture for photometry
        annulus_radii : tuple of float, optional
            Inner and outer radii of annulus for background estimation
            
        Returns:
        --------
        dict
            Dictionary with SNR statistics
        """
        if sources is None or len(sources) == 0:
            return {
                'mean_snr': np.nan,
                'median_snr': np.nan,
                'snr_values': []
            }
        
        # Create apertures for photometry
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        apertures = CircularAperture(positions, r=aperture_radius)
        
        # Perform aperture photometry
        background_subtracted = data - background
        phot_table = aperture_photometry(background_subtracted, apertures)
        
        # Background noise estimate
        background_std = self.estimate_background(data)['background_std']
        
        # Calculate SNR for each star
        snr_values = []
        for i, source in enumerate(sources):
            try:
                # Get flux from photometry
                flux = phot_table['aperture_sum'][i]
                
                # Calculate noise (Poisson + background)
                n_pixels = np.pi * aperture_radius**2
                noise = np.sqrt(flux + n_pixels * background_std**2)
                
                # Calculate SNR
                if noise > 0:
                    snr = flux / noise
                    snr_values.append(snr)
            except Exception as e:
                print(f"Error calculating SNR for star {i}: {e}")
        
        # Compute statistics
        if len(snr_values) > 0:
            mean_snr = np.mean(snr_values)
            median_snr = np.median(snr_values)
        else:
            mean_snr = median_snr = np.nan
        
        return {
            'mean_snr': mean_snr,
            'median_snr': median_snr,
            'snr_values': snr_values
        }
    
    def select_reference_stars(self, sources, min_stars=15, max_stars=30):
        """
        Select stars to use as reference for alignment assessment.
        
        Parameters:
        -----------
        sources : astropy.table.Table
            Table of detected sources
        min_stars : int, optional
            Minimum number of reference stars to select
        max_stars : int, optional
            Maximum number of reference stars to select
            
        Returns:
        --------
        numpy.ndarray
            Array of reference star positions (x, y)
        """
        if sources is None or len(sources) < min_stars:
            return None
        
        # Sort sources by flux (brightest first)
        sorted_sources = sources.copy()
        sorted_sources.sort('peak')
        sorted_sources.reverse()
        
        # Take the brightest stars up to max_stars
        num_stars = min(len(sorted_sources), max_stars)
        
        # Extract positions
        positions = np.zeros((num_stars, 2))
        for i in range(num_stars):
            positions[i, 0] = sorted_sources['xcentroid'][i]
            positions[i, 1] = sorted_sources['ycentroid'][i]
        
        return positions
    
    def calculate_registration_error(self, ref_stars, frame_stars):
        """
        Calculate registration error between reference and frame stars.
        
        Parameters:
        -----------
        ref_stars : numpy.ndarray
            Reference star positions (x, y)
        frame_stars : numpy.ndarray
            Frame star positions (x, y)
            
        Returns:
        --------
        float
            Root Mean Square (RMS) registration error
        """
        if ref_stars is None or frame_stars is None:
            return np.nan
        
        # Use a KD-Tree to find nearest neighbors
        tree = KDTree(frame_stars)
        
        # Find the nearest neighbor for each reference star
        distances, indices = tree.query(ref_stars, k=1)
        
        # Calculate RMS error
        rms_error = np.sqrt(np.mean(distances**2))
        
        return rms_error
    
    def calculate_quality_score(self, metrics, weights=None):
        """
        Calculate an overall quality score based on multiple metrics.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary with metrics
        weights : dict, optional
            Dictionary with weights for each metric
            
        Returns:
        --------
        float
            Quality score (0-100)
        """
        if weights is None:
            weights = {
                'fwhm': 0.3,           # Lower is better
                'ellipticity': 0.2,     # Lower is better
                'snr': 0.25,            # Higher is better
                'num_stars': 0.15,      # Higher is better
                'background_std': 0.1   # Lower is better
            }
        
        # Initialize score
        score = 0.0
        
        # Normalize and weight each metric
        try:
            # FWHM (lower is better) - invert so higher is better
            if not np.isnan(metrics['mean_fwhm']):
                # Assuming good FWHM is between 2-5 pixels
                fwhm_score = max(0, min(100, 100 * (1 - (metrics['mean_fwhm'] - 2) / 3)))
                score += weights['fwhm'] * fwhm_score
            
            # Ellipticity (lower is better) - invert so higher is better
            if not np.isnan(metrics['mean_ellipticity']):
                # Good ellipticity is close to 1, bad is > 1.5
                ellip_score = max(0, min(100, 100 * (1 - (metrics['mean_ellipticity'] - 1) / 0.5)))
                score += weights['ellipticity'] * ellip_score
            
            # SNR (higher is better)
            if not np.isnan(metrics['mean_snr']):
                # Assuming good SNR is > 50
                snr_score = max(0, min(100, 100 * (metrics['mean_snr'] / 50)))
                score += weights['snr'] * snr_score
            
            # Number of stars (higher is better)
            if metrics['num_stars'] > 0:
                # Assuming good images have > 100 stars
                num_stars_score = max(0, min(100, 100 * (metrics['num_stars'] / 100)))
                score += weights['num_stars'] * num_stars_score
            
            # Background std (lower is better) - invert so higher is better
            if not np.isnan(metrics['background_std']):
                # This is highly dependent on the image, so we use a relative measure
                # We'll assume the median across all images is a reasonable reference
                if hasattr(self, 'median_background_std'):
                    bg_score = max(0, min(100, 100 * (self.median_background_std / max(metrics['background_std'], 0.001))))
                    score += weights['background_std'] * bg_score
            
        except Exception as e:
            print(f"Error calculating quality score: {e}")
            score = np.nan
        
        return score
    
    def analyze_image(self, file_idx):
        """
        Analyze a single image and extract quality metrics.
        
        Parameters:
        -----------
        file_idx : int
            Index of the file to analyze
            
        Returns:
        --------
        dict
            Dictionary with quality metrics
        """
        file_path = self.files[file_idx]
        filename = os.path.basename(file_path)
        print(f"Analyzing {filename} ({file_idx+1}/{self.num_files})")
        
        try:
            # Open the FITS file
            with fits.open(file_path) as hdul:
                # Get the primary HDU
                hdu = hdul[0]
                data = hdu.data
                
                # Extract metadata from header
                metadata = self.extract_metadata(hdu.header)
                
                # Ensure we have 2D data
                if data.ndim > 2:
                    # If 3D, assume first plane is the image
                    data = data[0]
                
                # Detect stars
                sources = self.detect_stars(data)
                num_stars = 0 if sources is None else len(sources)
                
                # Measure star properties (FWHM, ellipticity)
                star_props = self.measure_star_properties(data, sources)
                
                # Estimate background
                bkg_props = self.estimate_background(data)
                
                # Measure SNR
                snr_props = self.measure_snr(data, sources, bkg_props['background'])
                
                # Select reference stars if this is the first good frame
                if self.reference_stars is None and sources is not None and len(sources) > 10:
                    self.reference_stars = self.select_reference_stars(sources)
                    self.reference_frame_idx = file_idx
                
                # Calculate registration error if we have reference stars
                registration_error = np.nan
                if self.reference_stars is not None and sources is not None and file_idx != self.reference_frame_idx:
                    frame_stars = np.array([[s['xcentroid'], s['ycentroid']] for s in sources])
                    registration_error = self.calculate_registration_error(self.reference_stars, frame_stars)
                
                # Assemble metrics
                metrics = {
                    'filename': filename,
                    'alt': metadata.get('alt', np.nan),
                    'az': metadata.get('az', np.nan),
                    'ra': metadata.get('ra', np.nan),
                    'dec': metadata.get('dec', np.nan),
                    'datetime': metadata.get('datetime', None),
                    'mean_fwhm': star_props['mean_fwhm'],
                    'median_fwhm': star_props['median_fwhm'],
                    'min_fwhm': star_props['min_fwhm'],
                    'max_fwhm': star_props['max_fwhm'],
                    'mean_ellipticity': star_props['mean_ellipticity'],
                    'median_ellipticity': star_props['median_ellipticity'],
                    'background_mean': bkg_props['background_mean'],
                    'background_std': bkg_props['background_std'],
                    'num_stars': num_stars,
                    'mean_snr': snr_props['mean_snr'],
                    'median_snr': snr_props['median_snr'],
                    'registration_error': registration_error,
                    'fwhm_values': star_props['fwhm_values'],
                    'ellipticity_values': star_props['ellipticity_values'],
                    'snr_values': snr_props['snr_values']
                }
                
                return metrics
                
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            return {
                'filename': filename,
                'alt': np.nan, 'az': np.nan, 'ra': np.nan, 'dec': np.nan,
                'datetime': None,
                'mean_fwhm': np.nan, 'median_fwhm': np.nan, 'min_fwhm': np.nan, 'max_fwhm': np.nan,
                'mean_ellipticity': np.nan, 'median_ellipticity': np.nan,
                'background_mean': np.nan, 'background_std': np.nan,
                'num_stars': 0, 'mean_snr': np.nan, 'median_snr': np.nan,
                'registration_error': np.nan,
                'fwhm_values': [], 'ellipticity_values': [], 'snr_values': []
            }
    
    def analyze_all_images(self):
        """
        Analyze all images and calculate overall quality metrics.
        """
        print(f"Starting analysis of {self.num_files} images...")
        start_time = datetime.now()
        
        # Process each file
        all_metrics = []
        for i in range(self.num_files):
            metrics = self.analyze_image(i)
            all_metrics.append(metrics)
            
            # Update DataFrame
            for key in self.quality_df.columns:
                if key in metrics:
                    self.quality_df.at[i, key] = metrics[key]
        
        # Calculate median background_std for quality score normalization
        self.median_background_std = self.quality_df['background_std'].median()
        
        # Calculate quality scores
        for i, metrics in enumerate(all_metrics):
            quality_score = self.calculate_quality_score(metrics)
            self.quality_df.at[i, 'quality_score'] = quality_score
        
        # Sort by quality score
        self.quality_df = self.quality_df.sort_values('quality_score', ascending=False)

        # Ensure all numeric columns have the correct data type
        numeric_columns = ['mean_fwhm', 'median_fwhm', 'min_fwhm', 'max_fwhm', 
                         'mean_ellipticity', 'median_ellipticity',
                         'background_mean', 'background_std', 
                         'num_stars', 'mean_snr', 'median_snr',
                         'quality_score']

        for col in numeric_columns:
            if col in self.quality_df.columns:
                self.quality_df[col] = pd.to_numeric(self.quality_df[col], errors='coerce')
        
        # Save results
        self.save_results()
        
        # Generate summary plots
        self.generate_summary_plots()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Analysis completed in {duration:.1f} seconds")
        
    def save_results(self):
        """
        Save analysis results to CSV and Excel files.
        """
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'quality_metrics.csv')
        self.quality_df.to_csv(csv_path, index=False)
        print(f"Saved metrics to {csv_path}")
        
        # Save to Excel
        try:
            excel_path = os.path.join(self.output_dir, 'quality_metrics.xlsx')
            self.quality_df.to_excel(excel_path, index=False)
            print(f"Saved metrics to {excel_path}")
        except Exception as e:
            print(f"Could not save Excel file: {e}")
        
        # Generate a stacking recommendations file
        self.generate_stacking_recommendations()
    
    def generate_stacking_recommendations(self):
        """
        Generate stacking recommendations based on quality analysis.
        """
        # Define cutoffs based on the analysis
        good_fwhm_cutoff = min(5.0, self.quality_df['mean_fwhm'].median() * 1.5)
        good_ellip_cutoff = min(1.5, self.quality_df['mean_ellipticity'].median() * 1.3)
        good_snr_cutoff = self.quality_df['mean_snr'].median() * 0.7
        good_quality_cutoff = self.quality_df['quality_score'].median() * 0.8
        
        # Count frames meeting each criterion
        num_good_fwhm = sum(self.quality_df['mean_fwhm'] <= good_fwhm_cutoff)
        num_good_ellip = sum(self.quality_df['mean_ellipticity'] <= good_ellip_cutoff)
        num_good_snr = sum(self.quality_df['mean_snr'] >= good_snr_cutoff)
        num_good_quality = sum(self.quality_df['quality_score'] >= good_quality_cutoff)
        
        # Get overall good frames (meeting all criteria)
        good_frames = self.quality_df[
            (self.quality_df['mean_fwhm'] <= good_fwhm_cutoff) &
            (self.quality_df['mean_ellipticity'] <= good_ellip_cutoff) &
            (self.quality_df['mean_snr'] >= good_snr_cutoff) &
            (self.quality_df['quality_score'] >= good_quality_cutoff)
        ]
        
        num_good_frames = len(good_frames)
        
        # Look for coordinate shifts that might indicate different groups
        ra_groups = []
        dec_groups = []
        
        if not self.quality_df['ra'].isna().all() and not self.quality_df['dec'].isna().all():
            # Use DBSCAN clustering to identify groups
            from sklearn.cluster import DBSCAN
            
            coords = np.array([
                self.quality_df['ra'].dropna().values,
                self.quality_df['dec'].dropna().values
            ]).T
            
            if len(coords) > 5:  # Only if we have enough data points
                try:
                    # Try to identify clusters with DBSCAN
                    dbscan = DBSCAN(eps=0.05, min_samples=5)
                    clusters = dbscan.fit_predict(coords)
                    
                    # Count how many images in each cluster
                    unique_clusters = np.unique(clusters)
                    if len(unique_clusters) > 1:  # If we found multiple clusters
                        ra_groups = []
                        dec_groups = []
                        for cluster_id in unique_clusters:
                            if cluster_id >= 0:  # Skip noise points (-1)
                                cluster_coords = coords[clusters == cluster_id]
                                ra_mean = np.mean(cluster_coords[:, 0])
                                dec_mean = np.mean(cluster_coords[:, 1])
                                ra_groups.append(ra_mean)
                                dec_groups.append(dec_mean)
                                
                except Exception as e:
                    print(f"Error during clustering: {e}")
        
        # Create a recommendations file
        recommendations_path = os.path.join(self.output_dir, 'stacking_recommendations.txt')
        
        with open(recommendations_path, 'w') as f:
            f.write("===== STACKING RECOMMENDATIONS =====\n\n")
            
            f.write("## Image Quality Summary\n")
            f.write(f"Total images analyzed: {self.num_files}\n")
            f.write(f"Images with good FWHM (≤ {good_fwhm_cutoff:.2f} pixels): {num_good_fwhm} ({100*num_good_fwhm/self.num_files:.1f}%)\n")
            f.write(f"Images with good ellipticity (≤ {good_ellip_cutoff:.2f}): {num_good_ellip} ({100*num_good_ellip/self.num_files:.1f}%)\n")
            f.write(f"Images with good SNR (≥ {good_snr_cutoff:.2f}): {num_good_snr} ({100*num_good_snr/self.num_files:.1f}%)\n")
            f.write(f"Images with good overall quality (≥ {good_quality_cutoff:.2f}): {num_good_quality} ({100*num_good_quality/self.num_files:.1f}%)\n")
            f.write(f"Images meeting ALL quality criteria: {num_good_frames} ({100*num_good_frames/self.num_files:.1f}%)\n\n")
            
            # Stacking stratgies recommendations
            f.write("## Recommended Stacking Strategies\n")
            
            # Position grouping
            if len(ra_groups) > 0:
                f.write("\n### Position Groups Detected\n")
                f.write("Multiple position groups were detected. Consider stacking these groups separately:\n")
                
                for i, (ra, dec) in enumerate(zip(ra_groups, dec_groups)):
                    f.write(f"Group {i+1}: RA: {ra:.4f}, Dec: {dec:.4f}\n")
                
                f.write("\nPosition shifts might indicate significant telescope adjustments or "
                        "different target acquisitions during the session.\n")
            
            # Progressive stacking
            f.write("\n### Progressive Stacking Approach\n")
            f.write("1. Start with the top 25% highest quality frames.\n")
            if len(good_frames) > 0:
                top_25pct = self.quality_df.nlargest(int(self.num_files * 0.25), 'quality_score')
                f.write(f"   - This would include {len(top_25pct)} frames with quality scores ≥ {top_25pct['quality_score'].min():.2f}\n")
            
            f.write("2. Create an initial reference stack from these frames.\n")
            f.write("3. Progressively add frames in order of decreasing quality.\n")
            f.write("4. Check the stack quality (SNR, detail preservation) after each addition.\n")
            f.write("5. Stop adding frames when quality begins to degrade.\n\n")
            
            # Quality cutoffs
            f.write("### Recommended Quality Cutoffs\n")
            f.write("Based on the analysis, consider these cutoff values for inclusion in stacking:\n")
            f.write(f"- FWHM: ≤ {good_fwhm_cutoff:.2f} pixels\n")
            f.write(f"- Ellipticity: ≤ {good_ellip_cutoff:.2f}\n")
            f.write(f"- SNR: ≥ {good_snr_cutoff:.2f}\n")
            f.write(f"- Overall quality score: ≥ {good_quality_cutoff:.2f}\n\n")
            
            # Reference frames
            f.write("### Suggested Reference Frames\n")
            if len(self.quality_df) > 0:
                best_frame = self.quality_df.iloc[0]
                f.write(f"Best overall frame: {best_frame['filename']} (Quality: {best_frame['quality_score']:.2f})\n")
                
                best_fwhm_frame = self.quality_df.nsmallest(1, 'mean_fwhm').iloc[0]
                f.write(f"Best FWHM frame: {best_fwhm_frame['filename']} (FWHM: {best_fwhm_frame['mean_fwhm']:.2f})\n")
                
                best_snr_frame = self.quality_df.nlargest(1, 'mean_snr').iloc[0]
                f.write(f"Best SNR frame: {best_snr_frame['filename']} (SNR: {best_snr_frame['mean_snr']:.2f})\n\n")
            
            # Star alignment tips
            f.write("### Star Alignment Tips\n")
            f.write("- Use a star detection threshold that identifies 20-30 stars across the field.\n")
            f.write("- Choose alignment stars that are well-distributed across the image.\n")
            f.write("- Consider using 'asterism' matching for more robust alignment.\n")
            f.write("- Use a transformation model appropriate for the distortion level:\n")
            f.write("  * Linear/Affine for minimal field distortion\n")
            f.write("  * Polynomial/Thin-plate spline for significant field distortion\n\n")
            
            # Output format recommendations
            f.write("### Output Format Recommendations\n")
            f.write("- Consider drizzle integration at 0.7x to 0.5x sampling if seeing was good.\n")
            f.write("- Use 32-bit floating point for stacking to preserve dynamic range.\n")
            f.write("- Apply a final normalization based on exposure time for each frame.\n")
            f.write("- Consider a local normalization algorithm if there are significant gradients.\n\n")
            
            f.write("===== END OF RECOMMENDATIONS =====\n")
        
        print(f"Saved stacking recommendations to {recommendations_path}")
    
    def generate_summary_plots(self):
        """
        Generate summary plots of quality metrics.
        """
        output_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create a sorted index based on quality score
        sorted_idx = self.quality_df['quality_score'].sort_values(ascending=False).index
        
        # Plot FWHM distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.quality_df['mean_fwhm'].dropna(), bins=20, alpha=0.7)
        plt.axvline(self.quality_df['mean_fwhm'].median(), color='r', linestyle='--', 
                   label=f'Median: {self.quality_df["mean_fwhm"].median():.2f}')
        plt.xlabel('Mean FWHM (pixels)')
        plt.ylabel('Number of Frames')
        plt.title('Distribution of Mean FWHM')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'fwhm_distribution.png'), dpi=150)
        
        # Plot Ellipticity distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.quality_df['mean_ellipticity'].dropna(), bins=20, alpha=0.7)
        plt.axvline(self.quality_df['mean_ellipticity'].median(), color='r', linestyle='--', 
                   label=f'Median: {self.quality_df["mean_ellipticity"].median():.2f}')
        plt.xlabel('Mean Ellipticity')
        plt.ylabel('Number of Frames')
        plt.title('Distribution of Mean Ellipticity')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'ellipticity_distribution.png'), dpi=150)
        
        # Plot SNR distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.quality_df['mean_snr'].dropna(), bins=20, alpha=0.7)
        plt.axvline(self.quality_df['mean_snr'].median(), color='r', linestyle='--', 
                   label=f'Median: {self.quality_df["mean_snr"].median():.2f}')
        plt.xlabel('Mean SNR')
        plt.ylabel('Number of Frames')
        plt.title('Distribution of Mean SNR')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'snr_distribution.png'), dpi=150)
        
        # Plot number of detected stars
        plt.figure(figsize=(10, 6))
        plt.hist(self.quality_df['num_stars'].dropna(), bins=20, alpha=0.7)
        plt.axvline(self.quality_df['num_stars'].median(), color='r', linestyle='--', 
                   label=f'Median: {self.quality_df["num_stars"].median():.0f}')
        plt.xlabel('Number of Detected Stars')
        plt.ylabel('Number of Frames')
        plt.title('Distribution of Detected Stars')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'stars_distribution.png'), dpi=150)
        
        # Plot quality score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.quality_df['quality_score'].dropna(), bins=20, alpha=0.7)
        plt.axvline(self.quality_df['quality_score'].median(), color='r', linestyle='--', 
                   label=f'Median: {self.quality_df["quality_score"].median():.2f}')
        plt.xlabel('Quality Score')
        plt.ylabel('Number of Frames')
        plt.title('Distribution of Quality Scores')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'quality_score_distribution.png'), dpi=150)
        
        # Plot metrics across all frames
        plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
        
        # FWHM and Ellipticity plot
        ax1 = plt.subplot(gs[0])
        ax1.plot(sorted_idx, self.quality_df.loc[sorted_idx, 'mean_fwhm'], 'bo-', alpha=0.7, label='FWHM')
        ax1.set_ylabel('FWHM (pixels)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title('Quality Metrics Across Frames (Ordered by Quality Score)')
        ax1.grid(alpha=0.3)
        
        ax1b = ax1.twinx()
        ax1b.plot(sorted_idx, self.quality_df.loc[sorted_idx, 'mean_ellipticity'], 'ro-', alpha=0.7, label='Ellipticity')
        ax1b.set_ylabel('Ellipticity', color='r')
        ax1b.tick_params(axis='y', labelcolor='r')
        
        # SNR and Number of Stars plot
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(sorted_idx, self.quality_df.loc[sorted_idx, 'mean_snr'], 'go-', alpha=0.7, label='SNR')
        ax2.set_ylabel('SNR', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.grid(alpha=0.3)
        
        ax2b = ax2.twinx()
        ax2b.plot(sorted_idx, self.quality_df.loc[sorted_idx, 'num_stars'], 'mo-', alpha=0.7, label='# Stars')
        ax2b.set_ylabel('# Stars', color='m')
        ax2b.tick_params(axis='y', labelcolor='m')
        
        # Quality Score plot
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(sorted_idx, self.quality_df.loc[sorted_idx, 'quality_score'], 'ko-', alpha=0.7)
        ax3.set_ylabel('Quality Score')
        ax3.set_xlabel('Frame Index (Sorted by Quality)')
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_overview.png'), dpi=150)
        
        # If we have altitude and azimuth data, plot them
        if not self.quality_df['alt'].isna().all() and not self.quality_df['az'].isna().all():
            plt.figure(figsize=(10, 6))
            plt.scatter(self.quality_df['az'], self.quality_df['alt'], 
                       c=self.quality_df['quality_score'], cmap='viridis', 
                       alpha=0.7, s=50)
            plt.colorbar(label='Quality Score')
            plt.xlabel('Azimuth (degrees)')
            plt.ylabel('Altitude (degrees)')
            plt.title('Quality Score by Telescope Position')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'quality_by_position.png'), dpi=150)
            
        # Plot histograms of key metrics with suggested cutoffs
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # FWHM histogram with suggested cutoff
        good_fwhm_cutoff = self.quality_df['mean_fwhm'].median() * 1.5
        axs[0, 0].hist(self.quality_df['mean_fwhm'].dropna(), bins=20, alpha=0.7)
        axs[0, 0].axvline(good_fwhm_cutoff, color='r', linestyle='--', 
                        label=f'Suggested Cutoff: {good_fwhm_cutoff:.2f}')
        axs[0, 0].set_xlabel('Mean FWHM (pixels)')
        axs[0, 0].set_ylabel('Number of Frames')
        axs[0, 0].set_title('FWHM Distribution with Recommended Cutoff')
        axs[0, 0].legend()
        axs[0, 0].grid(alpha=0.3)
        
        # Ellipticity histogram with suggested cutoff
        good_ellip_cutoff = min(1.5, self.quality_df['mean_ellipticity'].median() * 1.3)
        axs[0, 1].hist(self.quality_df['mean_ellipticity'].dropna(), bins=20, alpha=0.7)
        axs[0, 1].axvline(good_ellip_cutoff, color='r', linestyle='--', 
                        label=f'Suggested Cutoff: {good_ellip_cutoff:.2f}')
        axs[0, 1].set_xlabel('Mean Ellipticity')
        axs[0, 1].set_ylabel('Number of Frames')
        axs[0, 1].set_title('Ellipticity Distribution with Recommended Cutoff')
        axs[0, 1].legend()
        axs[0, 1].grid(alpha=0.3)
        
        # SNR histogram with suggested cutoff
        good_snr_cutoff = self.quality_df['mean_snr'].median() * 0.7
        axs[1, 0].hist(self.quality_df['mean_snr'].dropna(), bins=20, alpha=0.7)
        axs[1, 0].axvline(good_snr_cutoff, color='r', linestyle='--', 
                        label=f'Suggested Cutoff: {good_snr_cutoff:.2f}')
        axs[1, 0].set_xlabel('Mean SNR')
        axs[1, 0].set_ylabel('Number of Frames')
        axs[1, 0].set_title('SNR Distribution with Recommended Cutoff')
        axs[1, 0].legend()
        axs[1, 0].grid(alpha=0.3)
        
        # Quality score histogram with suggested cutoff
        good_quality_cutoff = self.quality_df['quality_score'].median() * 0.8
        axs[1, 1].hist(self.quality_df['quality_score'].dropna(), bins=20, alpha=0.7)
        axs[1, 1].axvline(good_quality_cutoff, color='r', linestyle='--', 
                        label=f'Suggested Cutoff: {good_quality_cutoff:.2f}')
        axs[1, 1].set_xlabel('Quality Score')
        axs[1, 1].set_ylabel('Number of Frames')
        axs[1, 1].set_title('Quality Score Distribution with Recommended Cutoff')
        axs[1, 1].legend()
        axs[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'recommended_cutoffs.png'), dpi=150)
        
        # Close all plots to free memory
        plt.close('all')

def main():
    """Parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze astronomical images for stacking quality assessment')
    
    parser.add_argument('input_directory', type=str, help='Directory containing FITS files')
    parser.add_argument('-o', '--output', type=str, help='Output directory for results', default=None)
    parser.add_argument('-p', '--pattern', type=str, help='File pattern for FITS files', default='*.fits')
    parser.add_argument('--fwhm', type=float, help='Initial FWHM estimate in pixels', default=3.0)
    parser.add_argument('--threshold', type=float, help='Star detection threshold (sigma)', default=5.0)
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Validate the input directory
    if not os.path.isdir(args.input_directory):
        print(f"Error: Input directory '{args.input_directory}' not found.")
        return 1
    
    # Set up output directory
    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.join(args.input_directory, 'quality_assessment')
    
    # Initialize the analyzer
    try:
        analyzer = ImageQualityAnalyzer(
            input_dir=args.input_directory,
            output_dir=output_dir,
            file_pattern=args.pattern
        )
        
        # Run the analysis
        print(f"Starting analysis of files in {args.input_directory} matching {args.pattern}")
        analyzer.analyze_all_images()
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved to {output_dir}")
        print("To assess your images for stacking, check the following files:")
        print(f"  - {os.path.join(output_dir, 'quality_metrics.csv')}")
        print(f"  - {os.path.join(output_dir, 'stacking_recommendations.txt')}")
        print(f"  - Plots in {os.path.join(output_dir, 'plots')}")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
