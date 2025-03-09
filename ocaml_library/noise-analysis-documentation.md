# Astronomical Noise Analysis Guide

This guide describes how to use the noise analysis capabilities added to the unified astronomy toolkit.

## Overview

The noise analysis tool helps you assess the quality of astronomical images by:
- Measuring background noise levels in different color channels
- Detecting stars automatically and reporting star coverage
- Estimating signal-to-noise ratio
- Providing image quality metrics

## Command Line Usage

### Analyzing a Single Image

To analyze noise in a single FITS file:

```bash
unified_cli -file "path/to/image.fits" analyze-noise
```

Optional parameters:
- `-threshold 3.0`: Set star detection sensitivity (default is 5.0, lower values detect fainter stars)
- `-v`: Enable verbose output

### Analyzing Multiple Images

To analyze noise in an entire directory of FITS files:

```bash
unified_cli -fits "path/to/directory" analyze-noise
```

The tool will provide:
- Individual analysis for each file
- Summary statistics for the entire dataset
- Range of values across all images

## Understanding the Results

### YCbCr Color Space

The analysis converts images to YCbCr color space:
- Y: Luminance (brightness) channel
- Cb and Cr: Color difference channels (blue-difference and red-difference)

For astronomical images, the Y channel is most important and represents overall signal quality.

### Key Metrics

- **Background Mean**: Average pixel value in background regions
- **Background StdDev**: Standard deviation of background pixels (noise level)
- **Star Fraction**: Percentage of image occupied by detected stars
- **S/N Ratio**: Signal-to-noise ratio (higher is better)

## Interpretation

- **High background mean with low star fraction**: Possible light pollution or sky glow
- **High background noise (stddev)**: Possible thermal noise, sensor issues, or insufficient exposure
- **Very low star fraction (<1%)**: Possible focus issues or incorrect exposure
- **Very high star fraction (>30%)**: Dense star field or potential false detections

## Examples

### Good Quality Image
```
Background Statistics (YCbCr color space):
  Y channel:  Mean=0.1430, StdDev=0.0087
  Cb channel: Mean=0.0012, StdDev=0.0021
  Cr channel: Mean=0.0008, StdDev=0.0018
Star Fraction: 4.25% of image contains stars
Estimated S/N ratio: 16.44
```

### Noisy Image
```
Background Statistics (YCbCr color space):
  Y channel:  Mean=0.1850, StdDev=0.0412
  Cb channel: Mean=0.0085, StdDev=0.0076
  Cr channel: Mean=0.0104, StdDev=0.0082
Star Fraction: 2.12% of image contains stars
Estimated S/N ratio: 4.49
```

## Integration with Other Tools

The noise analysis can be used in conjunction with other astronomy tools:

- Analyze noise before and after applying dark frame calibration
- Check noise levels in images before attempting plate solving
- Evaluate multiple images before stacking

## Technical Details

The noise analysis uses sigma-clipping to identify and separate background pixels from star pixels, then calculates statistics on the background regions. Star detection uses a threshold-based approach, examining local brightness compared to the background.
