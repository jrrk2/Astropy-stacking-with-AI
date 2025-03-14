# Implementation Guide: Delegating Hybrid Stacking to Astrometric Stacking

This guide outlines the simplified approach to hybrid stacking by leveraging the existing astrometric stacking functionality rather than duplicating its logic.

## Overview of the Approach

Instead of maintaining two separate implementations of WCS-based image alignment and stacking, we'll:

1. Use the astrometric stacking function directly when plate solving is enabled
2. Only implement the hybrid/live-stacking specific logic when needed
3. Avoid duplicating any complex alignment and stacking code

## Implementation Steps

### 1. Update Hybrid Stacking CLI

The CLI interface should be updated to detect when only plate solving is used and directly delegate to astrometric stacking:

```ocaml
(* In hybrid_stacking_cli.ml's main function *)

(* Handle special case: if no-live-stacking is specified and plate-solving is enabled,
   directly use astrometric stacking *)
if !use_plate_solving && not !use_live_stacking then begin
  Printf.printf "Using plate solving only: delegating to astrometric stacking\n";
  let success = Astrometric_alignment.stack_astrometric 
                   input_files !reference_index stacking_method !output_file in
  exit (if success then 0 else 1)
end
else begin
  (* Run the hybrid stacking process *)
  let success = hybrid_stack input_files !output_file ~config ~reference_idx:!reference_index () in
  exit (if success then 0 else 1)
end
```

### 2. Simplify Hybrid Stacking Implementation

Update the `hybrid_stack` function to handle three main cases:

1. **Plate solving only**: Delegate directly to astrometric stacking
2. **Live stacking only**: Implement minimal live stacking logic
3. **Hybrid mode**: Process each category of files separately

```ocaml
let hybrid_stack files output_path ?(config=default_config) ?(reference_idx=0) () =
  (* If plate solving only, delegate to astrometric stacking *)
  if config.use_plate_solving && not config.use_live_stacking then
    Astrometric_alignment.stack_astrometric files reference_idx config.stacking_method output_path
  
  (* Otherwise handle live stacking or hybrid mode *)
  else begin
    (* ... implement live stacking or hybrid logic ... *)
  end
```

### 3. Add Support for "Live Stacking Only" Mode (Optional)

This would require implementing transformation based on the live stacking coordinates. For now, you could either:

1. Leave this as "not implemented" with a warning message
2. Create a basic implementation that handles simple transformations

### 4. Future Hybrid Mode Development

For the full hybrid mode (combining both methods), you could:

1. Process plate-solved files first with astrometric stacking
2. Then handle live-stacking files separately
3. Combine the results if needed

## Benefits of This Approach

1. **Eliminates code duplication**: No need to maintain two implementations of the same WCS-based alignment
2. **Reduces debugging complexity**: When using `-no-live-stacking`, results will be guaranteed identical
3. **Modular approach**: Allows focusing development on the unique aspects of hybrid/live stacking
4. **Future flexibility**: Makes it easier to update either implementation independently

## Testing the Implementation

1. Test with `-no-live-stacking`: Should produce identical results to astrometric_stack_cli
2. Test with `-no-plate-solving`: Should use only live stacking coordinates
3. Test hybrid mode: Should handle both types of data

## Conclusion

This approach significantly simplifies the codebase by eliminating duplication and leveraging the well-tested astrometric stacking implementation. Focus your development efforts on the unique aspects of live stacking rather than reimplementing existing functionality.
