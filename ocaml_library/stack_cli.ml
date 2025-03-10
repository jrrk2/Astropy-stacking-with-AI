(* stack_cli.ml - Command line interface for image stacking *)

open Printf
open Image_alignment

(* Main CLI function *)
let main () =
  (* Define command line arguments *)
  let input_dir = ref "" in
  let output_file = ref "stacked.fits" in
  let reference_idx = ref (-1) in
  let auto_reference = ref true in
  let pattern = ref None in
  let stacking_method = ref Average in
  let sigma = ref 3.0 in
  let kappa = ref 2.5 in
  let detection_threshold = ref 5.0 in
  let verbose = ref false in
  
  (* Parse pattern string *)
  let parse_pattern str =
    match String.uppercase_ascii str with
    | "RGGB" -> pattern := Some `RGGB
    | "BGGR" -> pattern := Some `BGGR
    | "GRBG" -> pattern := Some `GRBG
    | "GBRG" -> pattern := Some `GBRG
    | _ -> printf "Warning: Unknown Bayer pattern '%s', using auto-detection\n" str
  in
  
  (* Parse stacking method *)
  let parse_method str =
    match String.uppercase_ascii str with
    | "AVERAGE" | "MEAN" -> stacking_method := Average
    | "MEDIAN" -> stacking_method := Median
    | "SIGMACLIP" | "SIGMA" -> stacking_method := SigmaClip !sigma
    | "KAPPA" -> stacking_method := Kappa !kappa
    | "WEIGHTED" -> stacking_method := WeightedAverage
    | _ -> printf "Warning: Unknown stacking method '%s', using average\n" str
  in
  
  let usage = "Usage: stack_cli [options] -input <directory>\n\nOptions:\n  -input <dir> - Directory containing FITS files to stack\n  -output <file> - Output file (default: stacked.fits)\n  -ref <index> - Index of reference image (default: auto-select)\n  -method <method> - Stacking method: average, median, sigmaclip, kappa, weighted\n  -sigma <value> - Sigma value for sigma clipping (default: 3.0)\n  -kappa <value> - Kappa value for kappa-sigma clipping (default: 2.5)\n  -pattern <pattern> - Bayer pattern: RGGB, BGGR, GRBG, GBRG\n  -threshold <value> - Star detection threshold (default: 5.0)\n  -verbose - Enable verbose output" in
  
  let specs = [
    ("-input", Arg.Set_string input_dir, "Directory containing FITS files to stack");
    ("-output", Arg.Set_string output_file, "Output file (default: stacked.fits)");
    ("-ref", Arg.Int (fun i -> reference_idx := i; auto_reference := false), "Index of reference image (0-based)");
    ("-method", Arg.String parse_method, "Stacking method: average, median, sigmaclip, kappa, weighted");
    ("-sigma", Arg.Set_float sigma, "Sigma value for sigma clipping (default: 3.0)");
    ("-kappa", Arg.Set_float kappa, "Kappa value for kappa-sigma clipping (default: 2.5)");
    ("-pattern", Arg.String parse_pattern, "Bayer pattern: RGGB, BGGR, GRBG, GBRG");
    ("-threshold", Arg.Set_float detection_threshold, "Star detection threshold (default: 5.0)");
    ("-verbose", Arg.Set verbose, "Enable verbose output");
  ] in
  
  (* Parse command line *)
  Arg.parse specs (fun _ -> ()) usage;
  
  (* Check required parameters *)
  if !input_dir = "" then begin
    printf "Error: Input directory must be specified\n";
    Arg.usage specs usage;
    exit 1
  end;
  
  (* Update stacking method with current sigma/kappa values *)
  (match !stacking_method with
   | SigmaClip _ -> stacking_method := SigmaClip !sigma
   | Kappa _ -> stacking_method := Kappa !kappa
   | _ -> ());
  
  (* Find FITS files in the directory *)
  let files = try
    Sys.readdir !input_dir
    |> Array.to_list
    |> List.filter (fun f -> 
         Filename.check_suffix f ".fits" || 
         Filename.check_suffix f ".fit" || 
         Filename.check_suffix f ".FITS" || 
         Filename.check_suffix f ".FIT")
    |> List.map (fun f -> Filename.concat !input_dir f)
    |> Array.of_list
  with _ -> 
    printf "Error reading directory %s\n" !input_dir;
    [||]
  in
  
  if Array.length files = 0 then begin
    printf "No FITS files found in %s\n" !input_dir;
    exit 1
  end;
  
  printf "Found %d FITS files\n" (Array.length files);
  
  (* Select reference frame *)
  let ref_idx = 
    if !auto_reference then begin
      printf "Auto-selecting best reference image...\n";
      find_best_reference_image files
    end else begin
      if !reference_idx < 0 || !reference_idx >= Array.length files then begin
        printf "Warning: Reference index %d out of range (0-%d), using auto-selection\n" 
          !reference_idx (Array.length files - 1);
        find_best_reference_image files
      end else
        !reference_idx
    end
  in
  
  printf "Using %s as reference image\n" (Filename.basename files.(ref_idx));
  
  (* Print stacking method *)
  let method_str = match !stacking_method with
    | Average -> "Average"
    | Median -> "Median"
    | SigmaClip sigma -> sprintf "Sigma-clip (%.1f)" sigma
    | Kappa k -> sprintf "Kappa-sigma (%.1f)" k
    | WeightedAverage -> "Weighted average"
  in
  printf "Stacking method: %s\n" method_str;
  
  (* Perform stacking *)
  printf "Starting stacking process...\n";
  
  let result = stack_auto files ref_idx !stacking_method !output_file in
  
  match result with
  | Some r ->
      printf "\nStacking completed successfully!\n";
      printf "  Output file: %s\n" r.output_file;
      printf "  Aligned images: %d\n" (Array.length r.aligned_images);
      if Array.length r.failed_images > 0 then
        printf "  Failed to align: %d\n" (Array.length r.failed_images);
      exit 0
  | None ->
      printf "\nStacking failed\n";
      exit 1

(* Program entry point *)
let () = main ()
