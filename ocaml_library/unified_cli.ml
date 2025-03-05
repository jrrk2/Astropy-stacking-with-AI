(* unified_cli.ml - Command line interface for the unified astronomy tools *)

open Printf
open Unified_interface

(* Main CLI function *)
let main () =
  (* Define command line arguments *)
  let fits_dir = ref "" in
  let json_dir = ref "" in
  let model_file = ref None in
  let output_file = ref None in
  let latitude = ref 52.0 in  (* Default to Cambridge, UK *)
  let longitude = ref 0.0 in
  let action = ref "help" in
  let object_name = ref "" in
  let ra = ref 0.0 in
  let dec = ref 0.0 in
  let alt = ref 0.0 in
  let az = ref 0.0 in
  let focus = ref 0 in
  let verbose = ref false in
  
  let usage = "Usage: unified_cli [options] action\n\nActions:\n  help - Show this help message\n  convert - Convert coordinates\n  lookup - Look up object in SIMBAD\n  analyze - Analyze pointing data\n  build - Build pointing model\n  correct - Apply pointing correction" in
  
  let specs = [
    ("-fits", Arg.Set_string fits_dir, "Directory containing FITS files");
    ("-json", Arg.Set_string json_dir, "Directory containing JSON files");
    ("-model", Arg.String (fun f -> model_file := Some f), "Model file (load/save)");
    ("-output", Arg.String (fun f -> output_file := Some f), "Output file");
    ("-lat", Arg.Set_float latitude, "Observer latitude in degrees");
    ("-long", Arg.Set_float longitude, "Observer longitude in degrees");
    ("-object", Arg.Set_string object_name, "Object name for SIMBAD lookup");
    ("-ra", Arg.Set_float ra, "Right Ascension in degrees");
    ("-dec", Arg.Set_float dec, "Declination in degrees");
    ("-alt", Arg.Set_float alt, "Altitude in degrees");
    ("-az", Arg.Set_float az, "Azimuth in degrees");
    ("-focus", Arg.Set_int focus, "Focus position for model");
    ("-v", Arg.Set verbose, "Verbose output");
  ] in
  
  (* Parse command line *)
  let anon_fun param = action := param in
  Arg.parse specs anon_fun usage;
  
  (* Create a context for coordinate transformations *)
  let context = create_context !latitude !longitude in
  
  (* Execute the requested action *)
  match !action with
  | "help" ->
      print_endline usage;
      print_help ()
      
  | "convert" ->
      if !ra <> 0.0 || !dec <> 0.0 then begin
        (* Convert RA/Dec to Alt/Az *)
        let (alt', az', ha) = radec_to_altaz context !ra !dec in
        printf "Converting RA=%.4f°, Dec=%.4f° to Alt/Az:\n" !ra !dec;
        printf "  RA: %.4f° = %s\n" !ra (Altaz.hms_of_float !ra);
        printf "  Dec: %.4f° = %s\n" !dec (Altaz.dms_of_float !dec);
        printf "  Alt: %.4f°\n" alt';
        printf "  Az: %.4f°\n" az';
        printf "  Hour Angle: %.4f hours\n" ha;
      end else if !alt <> 0.0 || !az <> 0.0 then begin
        (* Convert Alt/Az to RA/Dec *)
        let (ra', dec', ha) = altaz_to_radec context !alt !az in
        printf "Converting Alt=%.4f°, Az=%.4f° to RA/Dec:\n" !alt !az;
        printf "  Alt: %.4f°\n" !alt;
        printf "  Az: %.4f°\n" !az;
        printf "  RA: %.4f° = %s\n" ra' (Altaz.hms_of_float ra');
        printf "  Dec: %.4f° = %s\n" dec' (Altaz.dms_of_float dec');
        printf "  Hour Angle: %.4f hours\n" ha;
      end else begin
        printf "Error: Must specify either RA/Dec or Alt/Az for conversion\n";
        printf "Example: unified_cli -ra 83.8 -dec -5.4 convert\n";
        printf "      or unified_cli -alt 45.0 -az 180.0 convert\n";
      end
      
  | "lookup" ->
      if !object_name = "" then begin
        printf "Error: Must specify object name with -object\n";
        printf "Example: unified_cli -object \"M42\" lookup\n";
      end else begin
        printf "Looking up %s in SIMBAD...\n" !object_name;
        match QuerySimbad.get_object_coordinates !object_name with
        | Some (ra, dec) ->
            let (alt, az, _) = radec_to_altaz context ra dec in
            print_coords !object_name ra dec alt az
        | None ->
            printf "Object not found in SIMBAD\n"
      end
      
  | "analyze" ->
      if !fits_dir = "" then begin
        printf "Error: Must specify FITS directory with -fits\n";
        printf "Example: unified_cli -fits \"/path/to/fits\" analyze\n";
      end else begin
        (* Find all FITS files in the directory *)
        let files = try
          Array.map (fun f -> Filename.concat !fits_dir f)
                   (Array.of_list (List.filter (fun f -> 
                      Filename.check_suffix f ".fits" || 
                      Filename.check_suffix f ".fit") 
                    (Array.to_list (Sys.readdir !fits_dir))))
        with _ -> 
          printf "Error reading directory %s\n" !fits_dir;
          [||]
        in
        
        printf "Found %d FITS files in %s\n" (Array.length files) !fits_dir;
        
        let data = Array.fold_left (fun acc file ->
          match extract_pointing_data file with
          | Some data -> Array.append acc [|data|]
          | None -> acc
        ) [||] files in
        
        printf "Successfully extracted data from %d files\n" (Array.length data);
        analyze_pointing data true
      end
      
  | "build" ->
      if !fits_dir = "" then begin
        printf "Error: Must specify FITS directory with -fits\n";
        printf "Example: unified_cli -fits \"/path/to/fits\" -output \"model.json\" build\n";
      end else if !output_file = None then begin
        printf "Error: Must specify output file with -output\n";
        printf "Example: unified_cli -fits \"/path/to/fits\" -output \"model.json\" build\n";
      end else begin
        (* Find all FITS files in the directory *)
        let files = try
          Array.map (fun f -> Filename.concat !fits_dir f)
                   (Array.of_list (List.filter (fun f -> 
                      Filename.check_suffix f ".fits" || 
                      Filename.check_suffix f ".fit") 
                    (Array.to_list (Sys.readdir !fits_dir))))
        with _ -> 
          printf "Error reading directory %s\n" !fits_dir;
          [||]
        in
        
        printf "Found %d FITS files in %s\n" (Array.length files) !fits_dir;
        
        if Array.length files > 0 then begin
          let model = build_model_from_fits files in
          
          printf "Built pointing model with %d reference points\n" 
            (List.length model.reference_points);
            
          (* Save the model *)
          save_model model (Option.get !output_file);
          printf "Model saved to %s\n" (Option.get !output_file)
        end
      end
      
  | "correct" ->
      if !model_file = None then begin
        printf "Error: Must specify model file with -model\n";
        printf "Example: unified_cli -model \"model.json\" -ra 83.8 -dec -5.4 correct\n";
      end else if !ra = 0.0 && !dec = 0.0 && !alt = 0.0 && !az = 0.0 then begin
        printf "Error: Must specify either RA/Dec or Alt/Az for correction\n";
        printf "Example: unified_cli -model \"model.json\" -ra 83.8 -dec -5.4 correct\n";
        printf "      or unified_cli -model \"model.json\" -alt 45.0 -az 180.0 correct\n";
      end else begin
        match load_pointing_model (Option.get !model_file) with
        | Some model -> 
            printf "Loaded pointing model with %d reference points\n" 
              (List.length model.reference_points);
              
            if !ra <> 0.0 || !dec <> 0.0 then begin
              (* Apply correction in RA/Dec space *)
              let (corr_ra, corr_dec) = correct_position model !ra !dec !focus in
              
              printf "Original coordinates: RA=%.4f°, Dec=%.4f°\n" !ra !dec;
              printf "Corrected coordinates: RA=%.4f°, Dec=%.4f°\n" corr_ra corr_dec;
              printf "Correction: RA=%.4f°, Dec=%.4f°\n" (corr_ra -. !ra) (corr_dec -. !dec);
              
              (* Also show Alt/Az *)
              let (alt1, az1, _) = radec_to_altaz context !ra !dec in
              let (alt2, az2, _) = radec_to_altaz context corr_ra corr_dec in
              
              printf "Original Alt/Az: %.4f°, %.4f°\n" alt1 az1;
              printf "Corrected Alt/Az: %.4f°, %.4f°\n" alt2 az2;
            end else begin
              (* Apply correction in Alt/Az space *)
              let (alt2, az2) = convert_correction_to_altaz context model !alt !az !focus in
              
              printf "Original Alt/Az: %.4f°, %.4f°\n" !alt !az;
              printf "Corrected Alt/Az: %.4f°, %.4f°\n" alt2 az2;
              printf "Correction: Alt=%.4f°, Az=%.4f°\n" (alt2 -. !alt) (az2 -. !az);
              
              (* Also show RA/Dec *)
              let (ra1, dec1, _) = altaz_to_radec context !alt !az in
              let (ra2, dec2, _) = altaz_to_radec context alt2 az2 in
              
              printf "Original RA/Dec: %.4f°, %.4f°\n" ra1 dec1;
              printf "Corrected RA/Dec: %.4f°, %.4f°\n" ra2 dec2;
            end
        | None ->
            printf "Failed to load pointing model from %s\n" (Option.get !model_file)
      end
      
  | _ ->
      printf "Unknown action: %s\n" !action;
      print_endline usage

(* Program entry point *)
let () = main ()
