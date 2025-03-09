# Astronomical Noise Analysis with Bayer Pattern Support

This guide explains how to use the enhanced noise analysis tool with Bayer pattern debayering support.

## Integration Overview

The integration adds the following capabilities:

1. **Automatic Bayer pattern detection** - Detects patterns like RGGB, BGGR from filenames or FITS headers
2. **Debayering** - Converts raw sensor data to proper RGB values for accurate color analysis
3. **Enhanced noise analysis** - Provides accurate statistics in both luminance and color channels

## Files Added/Modified

1. **debayer.ml** - New module handling Bayer pattern detection and debayering
2. **unified_interface.ml** - Modified to incorporate debayering in noise analysis
3. **unified_cli.ml** - Already updated to expose noise analysis functionality

## Installation

Add these files to your project and ensure they're correctly referenced in your build system:

```
# Example for OCaml build system (adjust as needed)
# Add debayer.ml to your sources
ocamlc -c debayer.ml
ocamlc -c unified_interface.ml
ocamlc -c unified_cli.ml
ocamlc -o unified_cli debayer.cmo unified_interface.cmo unified_cli.cmo ...
```

## Usage

### Analyzing Raw Files

When analyzing raw files with Bayer patterns, simply use the standard command:

```bash
unified_cli -file "path/to/raw.fits" analyze-noise
```

The system will:
1. Automatically detect the Bayer pattern
2. Apply appropriate debayering
3. Provide noise analysis on the properly processed image

### Batch Analysis

For directories containing both raw and processed files:

```bash
unified_cli -fits "path/to/directory" analyze-noise
```

The tool provides a summary that includes:
- Statistics for all files
- Information about how many files had Bayer patterns
- Distribution of different Bayer patterns in the dataset

### Adjusting Star Detection

For images with faint stars, you can adjust the star detection threshold:

```bash
unified_cli -file "path/to/raw.fits" -threshold 3.0 analyze-noise
```

Lower values detect more stars but may include false positives.

## Example Output

Example output for a raw BGGR file:

```
Reading image data from lights/temp_288/light_20250217_201554_BGGR.fits...
Image dimensions: 3072x2080
Detected BGGR Bayer pattern
Reading image data...
Applying 2x2 binning with BGGR pattern...
Processed dimensions: 1536x1040
Analyzing noise with star detection threshold of 5.0...

Noise Analysis Results for light_20250217_201554_BGGR.fits:
=============================================================
Background Statistics (YCbCr color space):
  Y channel:  Mean=0.6053, StdDev=0.0067
  Cb channel: Mean=-0.0023, StdDev=0.0042
  Cr channel: Mean=0.0015, StdDev=0.0038
Star Fraction: 1.24% of image contains stars
Estimated S/N ratio: 90.11

Note: Analysis performed on debayered RGB data
```

## Troubleshooting

1. **Missing Bayer Pattern Detection**: Ensure filenames contain 'r' for RGGB or 'b' for BGGR, or check that the BAYERPAT keyword is in the FITS header

2. **Very Low Color Channel Values**: If Cb/Cr values are close to zero after debayering, there might be an issue with the Bayer pattern detection or the image might be effectively monochrome

3. **Memory Issues with Large Files**: For very large files, the binning process helps reduce memory usage while preserving image quality for analysis

## Advanced Notes for Developers

The debayering process uses 2x2 binning, which:
- Reduces the image to 1/4 its original size (width/2 × height/2)
- Naturally follows the Bayer pattern structure
- Improves S/N ratio through averaging
- Requires no interpolation of missing color data

Custom Bayer patterns can be added by extending the `detect_bayer_pattern` and adding corresponding binning functions in the `debayer.ml` module.
