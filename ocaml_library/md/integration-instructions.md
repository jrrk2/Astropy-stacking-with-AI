# Hybrid Stacking WCS Alignment Fix

This patch fixes the alignment issues in the hybrid stacking implementation by using the same WCS-based transformation approach as the astrometric stacking code.

## Explanation of the Problem

The issue with the current hybrid stacking implementation is that it uses a different mathematical approach for transforming images compared to the astrometric stacking code:

1. The **astrometric stacking** uses the WCS (World Coordinate System) to directly transform between coordinate systems by:
   - Converting pixel coordinates to sky coordinates (RA/Dec)
   - Converting sky coordinates back to pixel coordinates in the target frame

2. The **hybrid stacking** currently tries to:
   - Extract rotation, scale, and translation parameters 
   - Apply these transformations directly to pixel coordinates

This difference in approaches leads to misaligned results when the `-no-live-stacking` option is used, even though both should be using plate solving data.

## Integration Steps

To fix this issue, please follow these steps:

1. **Add new helper functions**
   
   Add the `apply_wcs_transform` and `extract_and_align_rgb_plane` functions from the first code artifact to your `hybrid_stacking.ml` file. Place them after the existing `apply_transform` function.

2. **Update the monochrome image handling**
   
   Find the section in `hybrid_stack` function that handles monochrome images. Look for the pattern:
   ```ocaml
   match transform_opt with
   | Some transform ->
       ...
   ```
   
   Replace it with the corresponding code from the second artifact. This will modify the monochrome processing to use direct WCS transformations.

3. **Update the RGB image handling**
   
   Find the section inside the `if is_rgb then begin ... end` block that determines which transformation to use for RGB images. Replace the transformation code with the updated version from the second artifact.

## Testing the Fix

After making these changes, run the hybrid stacking with the `-no-live-stacking` flag and compare the results with the astrometric stacking output. The images should now be properly aligned and produce equivalent results.

## What This Fix Accomplishes

This patch ensures that:

1. When using `-no-live-stacking`, the hybrid stacking uses the exact same mathematical approach for transformations as the astrometric stacking
2. RGB images are properly aligned using WCS parameters
3. The hybrid mode still works, falling back to live stacking coordinates when WCS data isn't available

The key improvement is that we now use the `create_wcs_transform` function from the `Astrometric_alignment` module, which ensures mathematical consistency between the two stacking methods.
