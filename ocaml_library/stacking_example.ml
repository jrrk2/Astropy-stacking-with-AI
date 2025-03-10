(* stacking_example.ml - Example usage of the image alignment and stacking module *)

open Printf
open Image_alignment

(* Example 1: Basic alignment and stacking *)
let example_basic_stacking input_dir output_file =
  printf "\nExample 1: Basic Alignment and Stacking\n";
  printf "=====================================\n";
  
  (* Find all FITS files in the directory *)
  let files = try
    Sys.readdir input_dir
    |> Array.to_list
    |> List.filter (fun f -> 
         Filename.check_suffix f ".fits" || 
         Filename.check_suffix f ".fit")
    |> List.map (fun f -> Filename.concat input_dir f)
    |> Array.of_list
  with _ -> 
    printf "Error reading directory %s\n" input_dir;
    [||]
  in
  
  if Array.length files = 0 then begin
    printf "No FITS files found in %s\n" input_dir;
    false
  end else begin
    printf "Found %d FITS files\n" (Array.length files);
    
    (* Auto-select reference frame *)
    let ref_idx = find_best_reference_image files in
    printf "Selected %s as reference frame\n" (Filename.basename files.(ref_idx));
    
    (* Parse pattern string *)
    let bayer_pattern = match pattern with
      | "RGGB" -> Some `RGGB
      | "BGGR" -> Some `BGGR
      | "GRBG" -> Some `GRBG
      | "GBRG" -> Some `GBRG
      | _ -> 
          printf "Unknown Bayer pattern '%s', using auto-detection\n" pattern;
          None
    in
    
    (* Perform stacking with specified Bayer pattern *)
    printf "Stacking Bayer pattern images with pattern %s...\n" 
      (match bayer_pattern with Some p -> Debayer_integration.describe_bayer_pattern p | None -> "auto-detect");
    
    let result = stack_bayer_images files ref_idx bayer_pattern Average output_file in
    
    match result with
    | Some r ->
        printf "Bayer stacking completed successfully!\n";
        printf "  Output file: %s\n" r.output_file;
        printf "  Aligned images: %d\n" (Array.length r.aligned_images);
        if Array.length r.failed_images > 0 then
          printf "  Failed to align: %d\n" (Array.length r.failed_images);
        true
    | None ->
        printf "Bayer stacking failed\n";
        false
  end

(* Example 4: RGB channel stacking *)
let example_rgb_stacking r_dir g_dir b_dir output_file =
  printf "\nExample 4: RGB Channel Stacking\n";
  printf "============================\n";
  
  (* Find all FITS files in each directory *)
  let load_files dir = 
    try
      Sys.readdir dir
      |> Array.to_list
      |> List.filter (fun f -> 
           Filename.check_suffix f ".fits" || 
           Filename.check_suffix f ".fit")
      |> List.map (fun f -> Filename.concat dir f)
      |> Array.of_list
    with _ -> 
      printf "Error reading directory %s\n" dir;
      [||]
  in
  
  let r_files = load_files r_dir in
  let g_files = load_files g_dir in
  let b_files = load_files b_dir in
  
  printf "Found %d red, %d green, and %d blue files\n" 
    (Array.length r_files) (Array.length g_files) (Array.length b_files);
  
  if Array.length r_files = 0 || Array.length g_files = 0 || Array.length b_files = 0 then begin
    printf "Missing files for one or more channels\n";
    false
  end else if Array.length r_files <> Array.length g_files || 
              Array.length r_files <> Array.length b_files then begin
    printf "Warning: Different number of files per channel\n";
    false
  end else begin
    (* Auto-select reference frame from red channel *)
    let ref_idx = find_best_reference_image r_files in
    printf "Selected %s as reference frame\n" (Filename.basename r_files.(ref_idx));
    
    (* Perform RGB stacking *)
    printf "Stacking separate RGB channels...\n";
    let result = stack_rgb_images r_files g_files b_files ref_idx Average output_file in
    
    match result with
    | Some r ->
        printf "RGB stacking completed successfully!\n";
        printf "  Output file: %s\n" r.output_file;
        printf "  Aligned images: %d\n" (Array.length r.aligned_images);
        if Array.length r.failed_images > 0 then
          printf "  Failed to align: %d\n" (Array.length r.failed_images);
        true
    | None ->
        printf "RGB stacking failed\n";
        false
  end

(* Main function to run all examples *)
let run_examples () =
  print_endline "Image Stacking Examples";
  print_endline "======================";
  
  (* Example paths - replace with actual paths *)
  let input_dir = "test_images" in
  let r_dir = "test_images/red" in
  let g_dir = "test_images/green" in
  let b_dir = "test_images/blue" in
  
  (* Run examples - comment out as needed *)
  ignore (example_basic_stacking input_dir "stacked_basic.fits");
  ignore (example_stacking_methods input_dir);
  ignore (example_bayer_stacking input_dir "stacked_bayer.fits" "RGGB");
  ignore (example_rgb_stacking r_dir g_dir b_dir "stacked_rgb.fits")

(* Run examples if executed directly *)
let () = 
  if not !Sys.interactive then
    run_examples()
printf "Selected %s as reference frame\n" (Filename.basename files.(ref_idx));
    
    (* Perform stacking with average method *)
    printf "Stacking images using average method...\n";
    let result = stack_auto files ref_idx Average output_file in
    
    match result with
    | Some r ->
        printf "Stacking completed successfully!\n";
        printf "  Output file: %s\n" r.output_file;
        printf "  Aligned images: %d\n" (Array.length r.aligned_images);
        if Array.length r.failed_images > 0 then
          printf "  Failed to align: %d\n" (Array.length r.failed_images);
        true
    | None ->
        printf "Stacking failed\n";
        false
  end

(* Example 2: Using different stacking methods *)
let example_stacking_methods input_dir =
  printf "\nExample 2: Different Stacking Methods\n";
  printf "==================================\n";
  
  (* Find all FITS files in the directory *)
  let files = try
    Sys.readdir input_dir
    |> Array.to_list
    |> List.filter (fun f -> 
         Filename.check_suffix f ".fits" || 
         Filename.check_suffix f ".fit")
    |> List.map (fun f -> Filename.concat input_dir f)
    |> Array.of_list
  with _ -> 
    printf "Error reading directory %s\n" input_dir;
    [||]
  in
  
  if Array.length files = 0 then begin
    printf "No FITS files found in %s\n" input_dir;
    false
  end else begin
    printf "Found %d FITS files\n" (Array.length files);
    
    (* Auto-select reference frame *)
    let ref_idx = find_best_reference_image files in
    printf "Selected %s as reference frame\n" (Filename.basename files.(ref_idx));
    
    (* Try different stacking methods *)
    let stack_methods = [
      "average", Average;
      "median", Median;
      "sigma_clip_3", SigmaClip 3.0;
      "kappa_2.5", Kappa 2.5;
    ] in
    
    let success_count = ref 0 in
    
    List.iter (fun (name, stack_method) ->
      let output_file = sprintf "stacked_%s.fits" name in
      printf "\nStacking with %s method...\n" name;
      
      match stack_auto files ref_idx stack_method output_file with
      | Some r ->
          printf "  Success! Output saved to %s\n" output_file;
          incr success_count
      | None ->
          printf "  Failed to stack with %s method\n" name
    ) stack_methods;
    
    printf "\nCompleted %d/%d stacking operations\n" !success_count (List.length methods);
    !success_count > 0
  end

(* Example 3: Handling Bayer pattern images *)
let example_bayer_stacking input_dir output_file pattern =
  printf "\nExample 3: Bayer Pattern Stacking\n";
  printf "==============================\n";
  
  (* Find all FITS files in the directory *)
  let files = try
    Sys.readdir input_dir
    |> Array.to_list
    |> List.filter (fun f -> 
         Filename.check_suffix f ".fits" || 
         Filename.check_suffix f ".fit")
    |> List.map (fun f -> Filename.concat input_dir f)
    |> Array.of_list
  with _ -> 
    printf "Error reading directory %s\n" input_dir;
    [||]
  in
  
  if Array.length files = 0 then begin
    printf "No FITS files found in %s\n" input_dir;
    false
  end else begin
    printf "Found %d FITS files\n" (Array.length files);
    
    (* Auto-select reference frame *)
    let ref_idx = find_best_reference_image files in
    
