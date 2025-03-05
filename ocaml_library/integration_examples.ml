(* integration_examples.ml - Usage examples for the unified interface *)

open Printf
open Unified_interface

(* Example 1: Basic coordinate conversion *)
let example_coordinate_conversion () =
  printf "\nExample 1: Coordinate Conversion\n";
  printf "================================\n";
  
  (* Create a context for Cambridge, UK *)
  let cambridge = create_context 52.245091 0.079609 in
  
  (* Convert coordinates for M42 (Orion Nebula) *)
  let ra = 83.8221 in  (* RA in degrees *)
  let dec = -5.3911 in (* Dec in degrees *)
  
  let (alt, az, ha) = radec_to_altaz cambridge ra dec in
  
  printf "M42 (Orion Nebula) coordinates:\n";
  printf "  J2000:   RA=%s, Dec=%s\n" (Altaz.hms_of_float ra) (Altaz.dms_of_float dec);
  printf "  Current: Alt=%.2f°, Az=%.2f°\n" alt az;
  printf "  Hour Angle: %.2f hours\n" ha;
  
  (* Convert back to RA/Dec *)
  let (ra2, dec2, ha2) = altaz_to_radec cambridge alt az in
  
  printf "\nConverting back to equatorial coordinates:\n";
  printf "  RA=%s (%.4f°), Dec=%s (%.4f°)\n" 
    (Altaz.hms_of_float ra2) ra2 (Altaz.dms_of_float dec2) dec2;
  printf "  Round-trip error: RA=%.4f°, Dec=%.4f°\n" (ra2 -. ra) (dec2 -. dec)

(* Example 2: Building a pointing model *)
let example_build_pointing_model directory output_file =
  printf "\nExample 2: Building a Pointing Model\n";
  printf "===================================\n";
  
  (* Find all FITS files in the directory *)
  let files = try
    Array.map (fun f -> Filename.concat directory f)
             (Array.of_list (List.filter (fun f -> 
                Filename.check_suffix f ".fits" || 
                Filename.check_suffix f ".fit") 
              (Array.to_list (Sys.readdir directory))))
  with _ -> 
    printf "Error reading directory %s\n" directory;
    [||]
  in
  
  printf "Found %d FITS files in %s\n" (Array.length files) directory;
  
  (* Build the model *)
  if Array.length files > 0 then begin
    let model = build_model_from_fits files in
    
    printf "Built pointing model with %d reference points\n" 
      (List.length model.reference_points);
      
    (* Save the model *)
    save_model model output_file;
    printf "Model saved to %s\n" output_file
  end

(* Example 3: Applying pointing corrections *)
let example_apply_corrections model_file =
  printf "\nExample 3: Applying Pointing Corrections\n";
  printf "=======================================\n";
  
  match load_pointing_model model_file with
  | Some model -> 
      printf "Loaded pointing model with %d reference points\n" 
        (List.length model.reference_points);
      
      (* Create a context *)
      let context = create_context 52.2053 0.1218 in
      
      (* Test objects *)
      let test_objects = [
        ("M31 (Andromeda)", 10.6847, 41.2687);
        ("M42 (Orion Nebula)", 83.8221, -5.3911);
        ("M51 (Whirlpool)", 202.4696, 47.1953);
      ] in
      
      List.iter (fun (name, ra, dec) ->
        (* Convert to Alt/Az *)
        let (alt, az, _) = radec_to_altaz context ra dec in
        
        (* Apply the correction in RA/Dec space *)
        let (corr_ra, corr_dec) = correct_position model ra dec 0 in
        
        (* Calculate the corrected Alt/Az *)
        let (corr_alt, corr_az, _) = radec_to_altaz context corr_ra corr_dec in
        
        (* Or use the direct Alt/Az correction *)
        let (alt2, az2) = convert_correction_to_altaz context model alt az 0 in
        
        printf "\nObject: %s\n" name;
        printf "  Original coordinates: RA=%.4f°, Dec=%.4f°\n" ra dec;
        printf "  Corrected coordinates: RA=%.4f°, Dec=%.4f°\n" corr_ra corr_dec;
        printf "  Correction: RA=%.4f°, Dec=%.4f°\n" (corr_ra -. ra) (corr_dec -. dec);
        printf "  Alt/Az: %.2f°, %.2f° -> %.2f°, %.2f°\n" alt az corr_alt corr_az;
        printf "  Direct Alt/Az correction: %.2f°, %.2f°\n" alt2 az2
      ) test_objects
  | None ->
      printf "Failed to load pointing model from %s\n" model_file

(* Example 4: Analyzing pointing data *)
let example_analyze_pointing directory =
  printf "\nExample 4: Analyzing Pointing Data\n";
  printf "==================================\n";
  
  (* Find all FITS files in the directory *)
  let files = try
    Array.map (fun f -> Filename.concat directory f)
             (Array.of_list (List.filter (fun f -> 
                Filename.check_suffix f ".fits" || 
                Filename.check_suffix f ".fit") 
              (Array.to_list (Sys.readdir directory))))
  with _ -> 
    printf "Error reading directory %s\n" directory;
    [||]
  in
  
  printf "Found %d FITS files in %s\n" (Array.length files) directory;
  
  (* Extract pointing data *)
  let data = Array.map (fun file ->
    match extract_pointing_data file with
    | Some data -> data
    | None -> failwith ("Failed to extract data from " ^ file)
  ) files in
  
  (* Analyze the data *)
  analyze_pointing data true

(* Example 5: Looking up objects in SIMBAD *)
let example_simbad_lookup objects =
  printf "\nExample 5: SIMBAD Object Lookup\n";
  printf "==============================\n";
  
  List.iter (fun name ->
    printf "Looking up %s...\n" name;
    match QuerySimbad.get_object_coordinates name with
    | Some (ra, dec) ->
        let context = create_context 52.2053 0.1218 in
        let (alt, az, _) = radec_to_altaz context ra dec in
        print_coords name ra dec alt az
    | None ->
        printf "  Object not found in SIMBAD\n"
  ) objects

(* Main function to run all examples *)
let run_examples () =
  print_help();
  
  (* Example 1: Basic coordinate conversion *)
  example_coordinate_conversion();
  
  (* Example 2: Building a pointing model *)
  (* Uncomment and provide a valid directory to test
  example_build_pointing_model "/path/to/fits/files" "pointing_model.json";
  *)
  
  (* Example 3: Applying pointing corrections *)
  (* Uncomment and provide a valid model file to test
  example_apply_corrections "pointing_model.json";
  *)
  
  (* Example 4: Analyzing pointing data *)
  (* Uncomment and provide a valid directory to test
  example_analyze_pointing "/path/to/fits/files";
  *)
  
  (* Example 5: Looking up objects in SIMBAD *)
  example_simbad_lookup ["M42"; "M31"; "NGC 7293"; "Pleiades"]

(* Run the examples when the file is executed *)
let () = run_examples()
