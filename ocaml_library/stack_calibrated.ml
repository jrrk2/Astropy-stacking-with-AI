(* stack_calibrated.ml - Specialized script for stacking calibrated RGB stellina images *)

open Types
open Printf
open Image_alignment
open Fits
open Debayer_integration

(* Function to collect calibrated images from a directory *)
let collect_calibrated_images dir pat =
  printf "Searching for calibrated %s images in %s...\n" pat dir;
  let pattern = Str.regexp (".*"^pat^".*") in
  (* Find all calibrated FITS files matching pattern *)
  let files = 
    try
      Sys.readdir dir
      |> Array.to_list
      |> List.filter (fun f -> 
           Filename.check_suffix f ".fits" && 
           String.starts_with ~prefix:"cal_" f &&
           (pat = "" || Str.string_match pattern f 0))
      |> List.map (fun f -> Filename.concat dir f)
      |> List.sort compare
      |> Array.of_list
    with _ -> 
      printf "Error reading directory %s\n" dir;
      [||]
  in
  
  printf "Found %d calibrated files\n" (Array.length files);
  files

(* Main function to run the stacking process *)
let stack_calibrated_images input_dir output_file stacking_method =
  (* Collect all calibrated images *)
  let images = collect_calibrated_images input_dir "_rgb" in
  
  if Array.length images = 0 then begin
    printf "No calibrated images found in %s\n" input_dir;
    exit 1
  end else begin
    (* Print summary of files to be stacked *)
    printf "\nStacking summary:\n";
    printf "================\n";
    printf "Input directory: %s\n" input_dir;
    printf "Output file: %s\n" output_file;
    printf "Files to process: %d\n" (Array.length images);
    
    (* Examine the first file to determine format *)
    let first_img = read_image images.(0) in
    let hdrh, _ = find_header_end images.(0) first_img in
    
    (* Check if it's a Bayer pattern image *)
    let bayer_pattern = get_bayer_pattern hdrh in
    let pattern_str = match bayer_pattern with
      | Some pattern -> describe_bayer_pattern pattern
      | None -> "None"
    in
    printf "Bayer pattern: %s\n" pattern_str;
    
    (* Check image dimensions *)
    let width = parse_int hdrh "NAXIS1" in
    let height = parse_int hdrh "NAXIS2" in
    printf "Image dimensions: %dx%d\n" width height;
    
    (* Auto-select reference frame based on quality *)
    printf "\nSelecting best reference frame...\n";
    let ref_idx = find_best_reference_image images in
    printf "Selected %s as reference frame\n" (Filename.basename images.(ref_idx));
    
    (* Perform stacking *)
    printf "\nStarting stacking process with method: ";
    (match stacking_method with
     | Average -> printf "Average\n"
     | Median -> printf "Median\n"
     | SigmaClip sigma -> printf "Sigma clipping (%.1f)\n" sigma
     | Kappa k -> printf "Kappa-sigma (%.1f)\n" k
     | WeightedAverage -> printf "Weighted average\n");
    
    let result = match bayer_pattern with
      | Some bayer -> 
          printf "Using Bayer pattern stacking for %s pattern\n" (describe_bayer_pattern bayer);
          stack_bayer_images images ref_idx (Some bayer) stacking_method output_file
      | None ->
          printf "Using standard monochrome stacking\n";
          stack_images images ref_idx stacking_method output_file
    in
    
    (* Report results *)
    match result with
    | Some r ->
        printf "\nStacking completed successfully!\n";
        printf "  Output file: %s\n" r.output_file;
        printf "  Aligned and stacked: %d/%d images\n" 
          (Array.length r.aligned_images) (Array.length images);
        if Array.length r.failed_images > 0 then begin
          printf "  Failed to align: %d images\n" (Array.length r.failed_images);
          printf "  Failed images:\n";
          Array.iter (fun f -> printf "    %s\n" (Filename.basename f)) r.failed_images;
        end;
        exit 0
    | None ->
        printf "\nStacking failed\n";
        exit 1
  end

(* Command line parsing *)
let () =
  let input_dir = ref "calibrated" in
  let output_file = ref "stacked.fits" in
  let method_name = ref "average" in
  let sigma = ref 3.0 in
  let kappa = ref 2.5 in
  
  let specs = [
    ("-input", Arg.Set_string input_dir, "Directory containing calibrated FITS files");
    ("-output", Arg.Set_string output_file, "Output file (default: stacked.fits)");
    ("-method", Arg.Set_string method_name, "Stacking method: average, median, sigmaclip, kappa, weighted");
    ("-sigma", Arg.Set_float sigma, "Sigma value for sigma clipping (default: 3.0)");
    ("-kappa", Arg.Set_float kappa, "Kappa value for kappa-sigma clipping (default: 2.5)");
  ] in
  
  let usage = "Usage: stack_calibrated [options]" in
  Arg.parse specs (fun _ -> ()) usage;
  
  (* Convert method string to stacking method value *)
  let stacking_method = match String.lowercase_ascii !method_name with
    | "average" | "mean" -> Average
    | "median" -> Median
    | "sigmaclip" | "sigma" -> SigmaClip !sigma
    | "kappa" -> Kappa !kappa
    | "weighted" -> WeightedAverage
    | _ -> 
        printf "Warning: Unknown stacking method '%s', using average\n" !method_name;
        Average
  in
  
  stack_calibrated_images !input_dir !output_file stacking_method
