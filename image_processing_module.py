import os
import cv2
import numpy as np
from numpy import ma

def custom_stretch(image, black_point=0.1, white_point=0.999, gamma=0.5):
    """
    Apply a custom stretch with histogram equalization using masked arrays
    """
    masked_image = ma.masked_invalid(image)
    valid_data = masked_image.compressed()
    black = np.percentile(valid_data, black_point * 100)
    white = np.percentile(valid_data, white_point * 100)
    stretched = ma.clip((masked_image - black) / (white - black), 0, 1)
    stretched = ma.power(stretched, gamma)
    return stretched.filled(0)  # Fill masked values with 0

def raised_cosine_taper_2d(shape, taper_width_percent=10):
    """
    Create a 2D raised cosine (Hann) window taper
    """
    height, width = shape
    h_taper = int(height * taper_width_percent / 100)
    w_taper = int(width * taper_width_percent / 100)
    
    def raised_cosine_window(length):
        return 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, length)))
    
    h_window = np.ones(width)
    v_window = np.ones(height)
    
    if w_taper > 0:
        h_window[:w_taper] = raised_cosine_window(w_taper)
        h_window[-w_taper:] = raised_cosine_window(w_taper)[::-1]
    if h_taper > 0:
        v_window[:h_taper] = raised_cosine_window(h_taper)
        v_window[-h_taper:] = raised_cosine_window(h_taper)[::-1]
    
    return v_window[:, np.newaxis] * h_window[np.newaxis, :]

def eliminate_background_wavelet(image, levels=3, taper_width_percent=10, debug_plot=False):
    """
    Remove background using wavelet transform with raised cosine tapering
    """
    try:
        import pywt
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("PyWavelets or Matplotlib not installed. Falling back to local background elimination.")
        return eliminate_background_local(image)
    
    # Create raised cosine taper mask
    taper_mask = raised_cosine_taper_2d(image.shape, taper_width_percent)
    
    # Perform wavelet decomposition
    coeffs = list(pywt.wavedec2(image, 'db4', level=levels))  # Convert to list
    
    # Store original approximation coefficients
    original_approx = coeffs[0].copy()
    
    # Modify detailed coefficients to reduce noise while preserving structure
    for i in range(1, len(coeffs)):
        coeffs[i] = list(coeffs[i])  # Convert tuple to list
        for j in range(len(coeffs[i])):
            coeffs[i][j] *= 0.5
        coeffs[i] = tuple(coeffs[i])  # Convert back to tuple
    
    # Reconstruct image with modified coefficients
    background = pywt.waverec2(coeffs, 'db4')
    
    # Ensure background has same shape as original
    background = background[:image.shape[0], :image.shape[1]]
    
    # Apply raised cosine tapering to background subtraction
    tapered_background = background * taper_mask
    
    return image - tapered_background

def save_16bit_png(data, filename, max_size_mb=25, max_trim_percent=0.10):
    """
    Save a normalized 16-bit PNG, automatically reducing size if needed
    """
    def try_save_with_trim(data, trim_percent):
        data_transposed = data.transpose(1, 2, 0)
        height, width, _ = data_transposed.shape
        
        trim_height = int(height * trim_percent)
        trim_width = int(width * trim_percent)
        
        trimmed_data = data_transposed[
            trim_height:height-trim_height, 
            trim_width:width-trim_width
        ]
        
        # Handle invalid values before scaling
        valid_mask = ~np.isnan(trimmed_data) & ~np.isinf(trimmed_data)
        cleaned_data = np.zeros_like(trimmed_data)
        cleaned_data[valid_mask] = np.clip(trimmed_data[valid_mask], 0, 1)
        
        # Scale to 16-bit range
        scaled_data = (cleaned_data * 65535).astype(np.uint16)
        temp_filename = f"{filename}.temp.png"
        
        cv2.imwrite(temp_filename, scaled_data, [
            cv2.IMWRITE_PNG_COMPRESSION, 9,
            cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT
        ])
        
        file_size_mb = os.path.getsize(temp_filename) / (1024 * 1024)
        os.remove(temp_filename)
        return scaled_data, file_size_mb
    
    trim_steps = np.linspace(0, max_trim_percent, num=5)
    for trim_percent in trim_steps:
        scaled_data, file_size_mb = try_save_with_trim(data, trim_percent)
        print(f"Trying {trim_percent*100:.1f}% trim: {file_size_mb:.1f}MB")
        
        if file_size_mb <= max_size_mb:
            print(f"Found acceptable trim: {trim_percent*100:.1f}%")
            cv2.imwrite(filename, scaled_data, [
                cv2.IMWRITE_PNG_COMPRESSION, 9,
                cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT
            ])
            print(f"Final PNG size: {os.path.getsize(filename) / (1024 * 1024):.2f}MB")
            return
    
    print("Warning: Could not reduce file size below target even with maximum trimming")
