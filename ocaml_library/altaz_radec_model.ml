(* Interface between Alt/Az conversion and quaternion pointing model *)
open Types
open Util
open Fits
open Printf
open Altaz  (* Import your Alt/Az conversion module *)
open Altaz_to_radec
open PointingModel

module AltAzIntegration = struct
  type altaz_reference_point = {
    mount_alt: float;
    mount_az: float;
    mount_ra: float;
    mount_dec: float;
    solved_ra: float;
    solved_dec: float;
    temperature: float;
    focus_position: int;
    correction: qt;
    timestamp: float;
    src_file: string;
  }

  (* Convert an altitude/azimuth reference point to RA/Dec format for the model *)
  let convert_to_radec_reference_point point latitude longitude =
    let timestamp = point.timestamp in
    let tm = Unix.gmtime timestamp in
    let year = tm.Unix.tm_year + 1900 in
    let month = tm.Unix.tm_mon + 1 in
    let day = tm.Unix.tm_mday in
    let hour = tm.Unix.tm_hour in
    let minute = tm.Unix.tm_min in
    let second = tm.Unix.tm_sec in
    
    (* Convert Alt/Az to RA/Dec *)
    let jd_calc, ra2000, dec2000, ra_now, dec_now, lst, ha_now = 
      altaz_to_j2000_time year month day hour minute second 
        point.mount_alt point.mount_az latitude longitude in
    
    (* Create a reference point for the pointing model *)
    {
      mount_ra = ra_now;
      mount_dec = dec_now;
      solved_ra = point.solved_ra;
      solved_dec = point.solved_dec;
      temperature = point.temperature;
      focus_position = point.focus_position;
      correction = point.correction;
      timestamp = point.timestamp;
      src_file = point.src_file;
    }

  (* Create an Alt/Az reference point *)
  let create_altaz_reference_point alt az solved_ra solved_dec temp focus timestamp src_file =
    (* Initially set the correction quaternion to identity - it will be calculated later *)
    {
      mount_alt = alt;
      mount_az = az;
      mount_ra = 0.0;  (* To be calculated *)
      mount_dec = 0.0; (* To be calculated *)
      solved_ra;
      solved_dec;
      temperature = temp;
      focus_position = focus;
      correction = Quaternion.identity;
      timestamp;
      src_file;
    }

  (* Add an Alt/Az reference point to the model *)
  let add_altaz_reference_point model alt az solved_ra solved_dec temp focus timestamp src_file latitude longitude =
    (* Create Alt/Az reference point *)
    let altaz_point = create_altaz_reference_point alt az solved_ra solved_dec temp focus timestamp src_file in
    
    (* Convert to RA/Dec for the model *)
    let radec_point = convert_to_radec_reference_point altaz_point latitude longitude in
    
    (* Calculate correction quaternion *)
    let correction = create_correction_quaternion 
      radec_point.mount_ra radec_point.mount_dec 
      radec_point.solved_ra radec_point.solved_dec in
    
    (* Create the final reference point with the calculated correction *)
    let final_point = {
      mount_ra = radec_point.mount_ra;
      mount_dec = radec_point.mount_dec;
      solved_ra = radec_point.solved_ra;
      solved_dec = radec_point.solved_dec;
      temperature = radec_point.temperature;
      focus_position = radec_point.focus_position;
      correction = correction;
      timestamp = radec_point.timestamp;
      src_file = radec_point.src_file;
    } in
    
    (* Add to the model *)
    { model with reference_points = final_point :: model.reference_points }

  (* Process JSON files for Alt/Az telescope pointing model *)
  let process_altaz_json_files json_dir latitude longitude =
    printf "Processing Alt/Az JSON files in %s\n" json_dir;
    
    (* Create empty pointing model *)
    let model = create_empty_model () in
    
    (* Find all files in the base directory and subdirectories *)
    let rec find_json_files dir =
      try
        let entries = Sys.readdir dir in
        Array.fold_left (fun acc entry ->
          let path = Filename.concat dir entry in
          if Sys.is_directory path then
            (* Recursively process subdirectory *)
            acc @ (find_json_files path)
          else if Filename.check_suffix entry ".json" then
            (* Add JSON file to list *)
            path :: acc
          else
            acc
        ) [] entries
      with _ -> 
        printf "Warning: Could not read directory %s\n" dir;
        []
    in
    
    let json_files = find_json_files json_dir in
    printf "Found %d JSON files\n" (List.length json_files);
    
    (* Map to store target positions to avoid duplicates *)
    let target_map = Hashtbl.create 101 in
    
    (* Process pointing files first to find target info *)
    List.iter (fun file ->
      if Filename.basename file = "point-deep-sky.json" then
        match JsonParser.parse_pointing_json file with
        | Some data ->
            (* Store the target RA/DEC with the file directory *)
            let dir = Filename.dirname file in
            Hashtbl.add target_map dir (data.solved_ra, data.solved_dec);
            printf "Found target for %s: RA=%.4f, DEC=%.4f\n" 
              (Filename.basename dir) data.solved_ra data.solved_dec
        | None -> ()
    ) json_files;
    
    (* Process astrometry files *)
    let model_with_points = List.fold_left (fun m file ->
      if Filename.basename file = "astrometry.json" then
        match JsonParser.parse_astrometry_json file with
        | Some data ->
            (* Try to find target info from the directory *)
            let dir = Filename.dirname (Filename.dirname file) in
            (match Hashtbl.find_opt target_map dir with
            | Some (target_ra, target_dec) ->
                (* Add reference point with Alt/Az values *)
                add_altaz_reference_point m data.alt data.az 
                  target_ra target_dec 20.0 data.map
                  data.timestamp file latitude longitude
            | None -> m)
        | None -> m
      else
        m
    ) model json_files in
    
    (* Calculate temperature coefficients *)
    let final_model = calculate_temp_coefficients model_with_points in
    
    (* Evaluate model accuracy *)
    let mean_error = evaluate_model final_model in
    printf "Alt/Az pointing model built with %d reference points\n" 
      (List.length final_model.reference_points);
    printf "Mean prediction error: %.4f degrees\n" mean_error;
    
    (* Return the model *)
    final_model

  (* Function to correct Alt/Az position to properly point at a target RA/Dec *)
  let correct_altaz_position model alt az temp focus latitude longitude timestamp =
    if List.length model.reference_points = 0 then
      (alt, az)  (* No correction if no reference points *)
    else
      (* Convert current Alt/Az to RA/Dec *)
      let tm = Unix.gmtime timestamp in
      let year = tm.Unix.tm_year + 1900 in
      let month = tm.Unix.tm_mon + 1 in
      let day = tm.Unix.tm_mday in
      let hour = tm.Unix.tm_hour in
      let minute = tm.Unix.tm_min in
      let second = tm.Unix.tm_sec in
      
      let jd_calc, ra2000, dec2000, ra_now, dec_now, lst, ha_now = 
        altaz_to_j2000_time year month day hour minute second 
          alt az latitude longitude in
      
      (* Apply pointing model correction in RA/Dec space *)
      let (corrected_ra, corrected_dec) = 
        correct_position model ra_now dec_now temp focus in
      
      (* Convert corrected RA/Dec back to Alt/Az *)
      let jd = computeTheJulianDay true year month day +. 
                float_of_int(hour*3600+minute*60+second) /. 86400.0 in
      let lst_calc = local_siderial_time' longitude (jd -. jd_2000) in
      
      (* Apply j2000_to_jnow transform for current epoch coordinates *)
      let ra_epoch, dec_epoch = j2000_to_jnow corrected_ra corrected_dec in
      
      (* Convert to Alt/Az *)
      let alt_calc, az_calc, hour_calc = raDectoAltAz ra_epoch dec_epoch latitude longitude lst_calc in
      
      (alt_calc, az_calc)

  (* Function to estimate pointing errors for different Alt/Az positions *)
  let analyze_pointing_errors model latitude longitude =
    let alt_step = 10.0 in
    let az_step = 15.0 in
    
    let errors = ref [] in
    
    (* Sample points across the sky *)
    for alt_i = 1 to 8 do
      let alt = float_of_int alt_i *. alt_step in
      for az_i = 0 to 23 do
        let az = float_of_int az_i *. az_step in
        
        (* Current timestamp *)
        let timestamp = Unix.gettimeofday() in
        let tm = Unix.gmtime timestamp in
        let year = tm.Unix.tm_year + 1900 in
        let month = tm.Unix.tm_mon + 1 in
        let day = tm.Unix.tm_mday in
        let hour = tm.Unix.tm_hour in
        let minute = tm.Unix.tm_min in
        let second = tm.Unix.tm_sec in
        
        (* Convert to RA/Dec *)
        let jd_calc, ra2000, dec2000, ra_now, dec_now, lst, ha_now = 
          altaz_to_j2000_time year month day hour minute second 
            alt az latitude longitude in
        
        (* Apply correction *)
        let (corrected_ra, corrected_dec) = 
          correct_position model ra_now dec_now 20.0 0 in
        
        (* Calculate difference *)
        let ra_diff = corrected_ra -. ra_now in
        let dec_diff = corrected_dec -. dec_now in
        let total_diff = sqrt (ra_diff *. ra_diff +. dec_diff *. dec_diff) in
        
        errors := (alt, az, total_diff) :: !errors;
      done;
    done;
    
    (* Return all errors *)
    !errors

  (* Plot the pointing error distribution by altitude/azimuth *)
  let plot_altaz_errors errors =
    let arr = Array.of_list errors in
    let alt_vals = Array.map (fun (alt, _, _) -> alt) arr in
    let az_vals = Array.map (fun (_, az, _) -> az) arr in
    let err_vals = Array.map (fun (_, _, err) -> err) arr in
    
    (* Find ranges *)
    let alt_min = Array.fold_left min alt_vals.(0) alt_vals in
    let alt_max = Array.fold_left max alt_vals.(0) alt_vals in
    let az_min = Array.fold_left min az_vals.(0) az_vals in
    let az_max = Array.fold_left max az_vals.(0) az_vals in
    let err_max = Array.fold_left max err_vals.(0) err_vals in
    
    (* Plot error as point size in Alt/Az coordinates *)
    Plplot.plsdev "pngcairo";
    Plplot.plsfnam "altaz_errors.png";
    Plplot.plinit ();
    Plplot.plenv az_min az_max alt_min alt_max 0 0;
    Plplot.pllab "Azimuth (degrees)" "Altitude (degrees)" "Pointing Errors by Alt/Az";
    
    (* Scale point sizes *)
    let scaled_errs = Array.map (fun err -> 1.0 +. 15.0 *. err /. err_max) err_vals in
    
    for i = 0 to Array.length alt_vals - 1 do
      Plplot.plssym 0.0 scaled_errs.(i);
      Plplot.plpoin [|az_vals.(i)|] [|alt_vals.(i)|] 4;
    done;
    
    Plplot.plend ();
    printf "Generated altaz_errors.png\n"

  (* Build a pointing model from a set of observation files *)
  let build_altaz_pointing_model fit_files json_dir latitude longitude =
    printf "Building Alt/Az pointing model...\n";
    
    (* Process FITS files with plate solve results *)
    let fits_points = Array.fold_left (fun acc filename ->
      try
        let img = read_image filename in
        let hdrh, _ = find_header_end filename img in
        
        (* Extract required information *)
        let mountalt = parse_float hdrh "MOUNTALT" in
        let mountaz = parse_float hdrh "MOUNTAZ" in
        let solvedra = parse_float hdrh "CRVAL1" in
        let solveddec = parse_float hdrh "CRVAL2" in
        let temp = get_temperature hdrh in
        let focus = try int_of_string (Hashtbl.find hdrh "FOCUSPOS") with _ -> 0 in
        let timestamp = get_timestamp hdrh in
        
        (* Add to model *)
        add_altaz_reference_point acc mountalt mountaz solvedra solveddec
          temp focus timestamp filename latitude longitude
      with e ->
        printf "Error processing %s: %s\n" filename (Printexc.to_string e);
        acc
    ) (create_empty_model ()) fit_files in
    
    (* Combine with JSON points if available *)
    let combined_model = 
      if json_dir <> "" then
        let json_model = process_altaz_json_files json_dir latitude longitude in
        (* Merge the two models *)
        { fits_points with 
          reference_points = 
            fits_points.reference_points @ 
            json_model.reference_points }
      else
        fits_points
    in
    
    (* Calculate temperature coefficients *)
    let final_model = calculate_temp_coefficients combined_model in
    
    (* Evaluate model accuracy *)
    let mean_error = evaluate_model final_model in
    printf "Alt/Az pointing model built with %d reference points\n" 
      (List.length final_model.reference_points);
    printf "Mean prediction error: %.4f degrees\n" mean_error;
    
    (* Return the model *)
    final_model

  (* Interactive testing of Alt/Az model *)
  let interactive_test_altaz_model model latitude longitude =
    printf "\nInteractive Alt/Az Model Testing\n";
    printf "===============================\n";
    printf "Enter alt, az, temperature and focus (or 'q' to quit):\n";
    
    let continue = ref true in
    while !continue do
      printf "> ";
      flush stdout;
      
      let line = input_line stdin in
      if line = "q" || line = "quit" || line = "exit" then
        continue := false
      else
        try
          let alt, az, temp, focus = Scanf.sscanf line "%f %f %f %d" (fun a b c d -> (a, b, c, d)) in
          let timestamp = Unix.gettimeofday() in
          let (corr_alt, corr_az) = correct_altaz_position model alt az temp focus latitude longitude timestamp in
          
          printf "Mount Alt/Az:      (%.4f, %.4f)\n" alt az;
          printf "Corrected Alt/Az:  (%.4f, %.4f)\n" corr_alt corr_az;
          printf "Difference:        (%.4f, %.4f)\n" (corr_alt -. alt) (corr_az -. az);
          
          (* Also show the RA/Dec conversions *)
          let tm = Unix.gmtime timestamp in
          let year = tm.Unix.tm_year + 1900 in
          let month = tm.Unix.tm_mon + 1 in
          let day = tm.Unix.tm_mday in
          let hour = tm.Unix.tm_hour in
          let minute = tm.Unix.tm_min in
          let second = tm.Unix.tm_sec in
          
          let _, ra2000, dec2000, ra_now, dec_now, _, _ = 
            altaz_to_j2000_time year month day hour minute second 
              alt az latitude longitude in
          
          let _, corr_ra2000, corr_dec2000, corr_ra_now, corr_dec_now, _, _ = 
            altaz_to_j2000_time year month day hour minute second 
              corr_alt corr_az latitude longitude in
          
          printf "Original RA/Dec:   (%.4f, %.4f)\n" ra_now dec_now;
          printf "Corrected RA/Dec:  (%.4f, %.4f)\n" corr_ra_now corr_dec_now;
          
        with
        | Scanf.Scan_failure _ -> 
            printf "Invalid input format. Expected: ALT AZ TEMP FOCUS\n"
        | End_of_file -> 
            continue := false
        | e -> 
            printf "Error: %s\n" (Printexc.to_string e)
    done

  (* Main function to run Alt/Az pointing model analysis *)
  let run_altaz_analysis files json_dir latitude longitude model_file save_file =
    match model_file with
    | Some filename ->
        (* Load existing model *)
        (match load_model_from_file filename with
        | Some model ->
            printf "Loaded Alt/Az pointing model with %d reference points\n" 
              (List.length model.reference_points);
            
            (* Run analysis on the model *)
            let errors = analyze_pointing_errors model latitude longitude in
            plot_altaz_errors errors;
            
            (* Interactive testing *)
            interactive_test_altaz_model model latitude longitude;
            model
        | None ->
            printf "Failed to load model from %s, building new model\n" filename;
            let model = build_altaz_pointing_model files json_dir latitude longitude in
            model)
    | None ->
        (* Build new model *)
        let model = build_altaz_pointing_model files json_dir latitude longitude in
        
        (* Save if requested *)
        (match save_file with
        | Some filename ->
            save_model_to_file model filename;
            printf "Model saved to %s\n" filename
        | None -> ());
        
        (* Run analysis on the model *)
        let errors = analyze_pointing_errors model latitude longitude in
        plot_altaz_errors errors;
        
        (* Interactive testing *)
        interactive_test_altaz_model model latitude longitude;
        model
end

(* Command line parsing for the Alt/Az interface *)
let parse_altaz_args () =
  let files = ref [] in
  let json_dir = ref "" in
  let latitude = ref 0.0 in
  let longitude = ref 0.0 in
  let model_file = ref None in
  let save_file = ref None in
  
  let specs = [
    ("-json", Arg.Set_string json_dir, "Directory containing JSON files for pointing model");
    ("-lat", Arg.Set_float latitude, "Observer latitude in degrees (required)");
    ("-long", Arg.Set_float longitude, "Observer longitude in degrees (required)");
    ("-load", Arg.String (fun file -> model_file := Some file), "Load model from file");
    ("-save", Arg.String (fun file -> save_file := Some file), "Save model to file");
  ] in
  
  let add_file f = files := f :: !files in
  let usage = "Usage: altaz_model [-json dir] [-lat value] [-long value] [-load file] [-save file] [fits_file1 fits_file2 ...]\n" in
  
  Arg.parse specs add_file usage;
  
  (* Check required parameters *)
  if !latitude = 0.0 && !longitude = 0.0 then
    printf "Warning: Using default latitude/longitude (0,0). Use -lat and -long options for your location.\n";
  
  (Array.of_list (List.rev !files), !json_dir, !latitude, !longitude, !model_file, !save_file)

(* Main entry point for Alt/Az pointing model *)
let run_altaz_model () =
  let files, json_dir, latitude, longitude, model_file, save_file = parse_altaz_args () in
  
  if Array.length files = 0 && json_dir = "" && model_file = None then
    printf "No input files, JSON directory, or model file specified. Nothing to do.\n"
  else
    ignore (AltAzIntegration.run_altaz_analysis files json_dir latitude longitude model_file save_file)

(* Uncomment to make this the main entry point *)
let () = run_altaz_model ()
