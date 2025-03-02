open Types
open Fits
open Util
open Printf
open Plplot
open Yojson.Basic.Util

module Array = struct
  include Array  (* Include all the standard Array module functions *)
  
  (* fold_left2: Fold over two arrays simultaneously *)
  let fold_left2 f init arr1 arr2 =
    if Array.length arr1 <> Array.length arr2 then
      invalid_arg "Array.fold_left2: arrays have different lengths"
    else
      let acc = ref init in
      for i = 0 to Array.length arr1 - 1 do
        acc := f !acc arr1.(i) arr2.(i)
      done;
      !acc
end

(* Create directory if it doesn't exist *)
let ensure_directory dir =
  if not (Sys.file_exists dir) then
    Unix.mkdir dir 0o755
  else if not (Sys.is_directory dir) then
    failwith (sprintf "%s exists but is not a directory" dir)

(* Quick scan of just the FITS header *)
let scan_fits_temperature filename =
  let fd = open_in_bin filename in
  try
    let header = read_header fd "" in
    let qhdr = Hashtbl.create 257 in
    let () = scan_header qhdr header 0 in
    close_in fd;
    Some { filename; qhdr; temp = get_temperature qhdr }
  with _ -> 
    close_in fd;
    printf "Warning: Could not read temperature from %s\n" filename;
    None

(* Format date from FITS header for filename *)
let format_date_from_header hdr =
  try
    let date_str = String.trim (Hashtbl.find hdr "DATE-OBS=") in
    try 
      Some (Scanf.sscanf date_str "'%d-%d-%dT%d:%d:%d'" 
        (fun yr mon day hr min sec ->
          sprintf "%04d%02d%02d_%02d%02d%02d" yr mon day hr min sec))
    with _ -> 
      printf "Warning: Could not parse DATE-OBS format: %s\n" date_str;
      None
  with _ -> 
    printf "Warning: Could not find DATE-OBS in header\n";
    None

(* Temperature plots *)
let plot_temp_vs_time (stats:temp_info array) =
  (* Get timestamps first *)
  let datum = ref (Unix.gettimeofday()) in
  let x' = Array.map (fun (s:temp_info) -> 
    let stamp = get_timestamp s.qhdr in if !datum > stamp then datum := stamp; stamp) stats in
  let x = Array.map (fun itm -> 
    itm -. !datum) x' in
  let y = Array.map (fun (s:temp_info) -> s.temp) stats in
  
  (* Check for valid range *)
  let x_min = Array.fold_left min x.(0) x in
  let x_max = Array.fold_left max x.(0) x in
  let y_min = Array.fold_left min y.(0) y in
  let y_max = Array.fold_left max y.(0) y in
  
  (* Ensure valid plotting range *)
  let x_range = x_max -. x_min in
  let y_range = y_max -. y_min in
  let x_min = if x_range = 0.0 then x_min -. 1.0 else x_min -. (x_range *. 0.05) in
  let x_max = if x_range = 0.0 then x_max +. 1.0 else x_max +. (x_range *. 0.05) in
  let y_min = if y_range = 0.0 then y_min -. 1.0 else y_min -. (y_range *. 0.05) in
  let y_max = if y_range = 0.0 then y_max +. 1.0 else y_max +. (y_range *. 0.05) in
  
  (* Generate plot *)
  plsdev "pngcairo";
  plsfnam "temperature.png";
  plinit ();
  plenv x_min x_max y_min y_max 0 0;
  pllab "Time (seconds from start)" "Temperature (°C)" "Temperature vs Time";
  plline x y;
  
  (* Add linear trend line *)
  if Array.length x > 1 then begin
    (* Calculate linear regression *)
    let n = float_of_int (Array.length x) in
    let sum_x = Array.fold_left (+.) 0.0 x in
    let sum_y = Array.fold_left (+.) 0.0 y in
    let sum_xy = Array.fold_left2 (fun acc xi yi -> acc +. xi *. yi) 0.0 x y in
    let sum_xx = Array.fold_left (fun acc xi -> acc +. xi *. xi) 0.0 x in
    
    let slope = ((n *. sum_xy) -. (sum_x *. sum_y)) /. 
                ((n *. sum_xx) -. (sum_x *. sum_x)) in
    let intercept = (sum_y -. (slope *. sum_x)) /. n in
    
    (* Draw trend line *)
    let trend_x = [| x_min; x_max |] in
    let trend_y = [| intercept +. slope *. x_min; intercept +. slope *. x_max |] in
    
    plcol0 3; (* Green *)
    plline trend_x trend_y;
    plcol0 1; (* Reset color *)
    
    (* Add annotation about cooling/warming rate *)
    let rate_text = Printf.sprintf "Temperature rate: %.3f °C/hour" (slope *. 3600.0) in
    plptex (x_min +. x_range *. 0.1) (y_max -. y_range *. 0.1) 0.0 0.0 0.0 rate_text;
  end;
  
  plend ();
  printf "Generated temperature.png\n"

(* Plot data about the quaternion model accuracy *)
let plot_model_accuracy model =
  (* Extract actual errors from reference points *)
  let points = model.reference_points in
  if List.length points = 0 then
    printf "No reference points in model to plot\n"
  else
    (* Calculate errors for each reference point *)
    let errors = List.map (fun p ->
      (* Remove this point from the model temporarily *)
      let temp_model = { model with reference_points = 
        List.filter (fun p' -> p != p') model.reference_points } in
      
      (* Predict position using the temporary model *)
      let (pred_ra, pred_dec) = PointingModel.correct_position temp_model p.mount_ra 
                                   p.mount_dec p.temperature 
                                   p.focus_position in
      
      (* Calculate error between prediction and actual solved position *)
      let ra_error = pred_ra -. p.solved_ra in
      let dec_error = pred_dec -. p.solved_dec in
      let total_error = sqrt (ra_error *. ra_error +. dec_error *. dec_error) in
      
      (* Return point with errors *)
      (p, ra_error, dec_error, total_error)
    ) points in
    
    (* Plot errors vs temperature *)
    let temps = List.map (fun ((p:reference_point), _, _, _) -> p.temperature) errors in
    let total_errs = List.map (fun (_, _, _, e) -> e) errors in
    
    (* Convert to arrays for plotting *)
    let temp_array = Array.of_list temps in
    let err_array = Array.of_list total_errs in
    
    (* Find min/max for ranges *)
    let t_min = List.fold_left min (List.hd temps) temps in
    let t_max = List.fold_left max (List.hd temps) temps in
    let e_min = List.fold_left min (List.hd total_errs) total_errs in
    let e_max = List.fold_left max (List.hd total_errs) total_errs in
    
    (* Add margins *)
    let t_range = t_max -. t_min in
    let e_range = e_max -. e_min in
    let t_min = if t_range = 0.0 then t_min -. 1.0 else t_min -. (t_range *. 0.1) in
    let t_max = if t_range = 0.0 then t_max +. 1.0 else t_max +. (t_range *. 0.1) in
    let e_min = if e_range = 0.0 then e_min -. 0.01 else e_min -. (e_range *. 0.1) in
    let e_max = if e_range = 0.0 then e_max +. 0.01 else e_max +. (e_range *. 0.1) in
    
    (* Plot errors vs temperature *)
    plsdev "pngcairo";
    plsfnam "model_accuracy.png";
    plinit ();
    plenv t_min t_max e_min e_max 0 0;
    pllab "Temperature (°C)" "Error (degrees)" "Model Accuracy vs Temperature";
    plpoin temp_array err_array 4;
    plend ();
    printf "Generated model_accuracy.png\n";
    
    (* Plot errors vs sky position *)
    let ra_array = Array.of_list (List.map (fun (p, _, _, _) -> p.mount_ra) errors) in
    let dec_array = Array.of_list (List.map (fun (p, _, _, _) -> p.mount_dec) errors) in
    let err_size_array = Array.of_list (List.map (fun (_, _, _, e) -> 1.0 +. e *. 20.0) errors) in
    
    (* Find min/max for RA/DEC ranges *)
    let ra_min = Array.fold_left min ra_array.(0) ra_array in
    let ra_max = Array.fold_left max ra_array.(0) ra_array in
    let dec_min = Array.fold_left min dec_array.(0) dec_array in
    let dec_max = Array.fold_left max dec_array.(0) dec_array in
    
    (* Add margins *)
    let ra_range = ra_max -. ra_min in
    let dec_range = dec_max -. dec_min in
    let ra_min = if ra_range = 0.0 then ra_min -. 1.0 else ra_min -. (ra_range *. 0.1) in
    let ra_max = if ra_range = 0.0 then ra_max +. 1.0 else ra_max +. (ra_range *. 0.1) in
    let dec_min = if dec_range = 0.0 then dec_min -. 1.0 else dec_min -. (dec_range *. 0.1) in
    let dec_max = if dec_range = 0.0 then dec_max +. 1.0 else dec_max +. (dec_range *. 0.1) in
    
    plsdev "pngcairo";
    plsfnam "model_sky_errors.png";
    plinit ();
    plenv ra_min ra_max dec_min dec_max 0 0;
    pllab "RA (degrees)" "DEC (degrees)" "Model Errors Across Sky";
    for i = 0 to Array.length ra_array - 1 do
      plssym 0.0 err_size_array.(i);
      plpoin [|ra_array.(i)|] [|dec_array.(i)|] 4;
      done;
    plpoin ra_array dec_array 4;
    plend ();
    printf "Generated model_sky_errors.png\n"

(* Function to test the model on real-world data *)
let test_model_on_data model stats =
  printf "\nTesting model on %d data points...\n" (Array.length stats);
  
  (* Calculate errors with and without model correction *)
  let uncorrected_errors = ref [] in
  let corrected_errors = ref [] in
  
  Array.iter (fun s ->
    (* Original error *)
    let ra_error = s.mountra -. s.solvedra in
    let dec_error = s.mountdec -. s.solveddec in
    let total_error = sqrt (ra_error *. ra_error +. dec_error *. dec_error) in
    uncorrected_errors := total_error :: !uncorrected_errors;
    
    (* Get model correction *)
    let (pred_ra, pred_dec) = PointingModel.correct_position model s.mountra s.mountdec 
                                s.temperature (try int_of_string (Hashtbl.find s.hdrh "MAP=") with _ -> 178128) in
    
    (* Corrected error *)
    let corr_ra_error = pred_ra -. s.solvedra in
    let corr_dec_error = pred_dec -. s.solveddec in
    let corr_total_error = sqrt (corr_ra_error *. corr_ra_error +. corr_dec_error *. corr_dec_error) in
    corrected_errors := corr_total_error :: !corrected_errors;
  ) stats;
  
  (* Calculate statistics *)
  let avg_uncorr = List.fold_left (+.) 0.0 !uncorrected_errors /. float_of_int (List.length !uncorrected_errors) in
  let avg_corr = List.fold_left (+.) 0.0 !corrected_errors /. float_of_int (List.length !corrected_errors) in
  
  let max_uncorr = List.fold_left max (List.hd !uncorrected_errors) !uncorrected_errors in
  let max_corr = List.fold_left max (List.hd !corrected_errors) !corrected_errors in
  
  (* Print results *)
  printf "Original average error: %.4f degrees (max: %.4f)\n" avg_uncorr max_uncorr;
  printf "Corrected average error: %.4f degrees (max: %.4f)\n" avg_corr max_corr;
  printf "Improvement: %.1f%%\n" ((avg_uncorr -. avg_corr) /. avg_uncorr *. 100.0);
  
  (* Plot comparison histogram *)
  let nbins = 20 in
  let max_error = max max_uncorr max_corr in
  let bin_width = max_error /. float_of_int nbins in
  
  let uncorr_bins = Array.make nbins 0 in
  let corr_bins = Array.make nbins 0 in
  
  List.iter (fun err ->
    let bin = int_of_float (err /. bin_width) in
    if bin >= 0 && bin < nbins then
      uncorr_bins.(bin) <- uncorr_bins.(bin) + 1
  ) !uncorrected_errors;
  
  List.iter (fun err ->
    let bin = int_of_float (err /. bin_width) in
    if bin >= 0 && bin < nbins then
      corr_bins.(bin) <- corr_bins.(bin) + 1
  ) !corrected_errors;
  
  (* Convert to float arrays for plotting *)
  let x = Array.init nbins (fun i -> (float_of_int i +. 0.5) *. bin_width) in
  let y1 = Array.map float_of_int uncorr_bins in
  let y2 = Array.map float_of_int corr_bins in
  
  (* Plot histogram *)
  plsdev "pngcairo";
  plsfnam "error_comparison.png";
  plinit ();
  
  (* Find y max *)
  let y_max = max (Array.fold_left max 0.0 y1) (Array.fold_left max 0.0 y2) in
  
  plenv 0.0 max_error 0.0 (y_max *. 1.1) 0 0;
  pllab "Error (degrees)" "Count" "Error Distribution With/Without Model Correction";
  
  (* Plot uncorrected histogram *)
  plcol0 1; (* Red *)
  plbin x y1 [PL_BIN_DEFAULT];
  
  (* Plot corrected histogram *)
  plcol0 3; (* Green *)
  plbin x y2 [PL_BIN_DEFAULT];
  
  (* Add legend *)
  plptex (max_error *. 0.7) (y_max *. 0.9) 0.0 0.0 0.0 "Without correction";
  plcol0 1; (* Red *)
  plpoin [|max_error *. 0.65|] [|y_max *. 0.9|] 4;
  
  plptex (max_error *. 0.7) (y_max *. 0.8) 0.0 0.0 0.0 "With correction";
  plcol0 3; (* Green *)
  plpoin [|max_error *. 0.65|] [|y_max *. 0.8|] 4;
  
  plend ();
  printf "Generated error_comparison.png\n"

(* Interactive testing of the model *)
let interactive_test_model model =
  printf "\nInteractive Model Testing\n";
  printf "=========================\n";
  printf "Enter mount RA, DEC, temperature and focus (or 'q' to quit):\n";
  
  let continue = ref true in
  while !continue do
    printf "> ";
    flush stdout;
    
    let line = input_line stdin in
    if line = "q" || line = "quit" || line = "exit" then
      continue := false
    else
      try
        let ra, dec, temp, focus = Scanf.sscanf line "%f %f %f %d" (fun a b c d -> (a, b, c, d)) in
        let (corr_ra, corr_dec) = PointingModel.correct_position model ra dec temp focus in
        
        printf "Mount:      (%.4f, %.4f)\n" ra dec;
        printf "Corrected:  (%.4f, %.4f)\n" corr_ra corr_dec;
        printf "Difference: (%.4f, %.4f)\n" (corr_ra -. ra) (corr_dec -. dec);
      with
      | Scanf.Scan_failure _ -> 
          printf "Invalid input format. Expected: RA DEC TEMP FOCUS\n"
      | End_of_file -> 
          continue := false
      | e -> 
          printf "Error: %s\n" (Printexc.to_string e)
  done
  
(* New main analysis function to include model functionality *)
let analyze_with_model files flags model_file =
  let model = match model_file with
    | Some file -> load_model_from_file file
    | None -> None
  in
  
  match model with
  | Some model ->
      (* Analyze files with the model *)
      printf "Loaded pointing model with %d reference points\n" 
        (List.length model.reference_points);
      
      (* Plot model accuracy *)
      plot_model_accuracy model;
      
      (* If we have files to analyze, test the model on them *)
      if Array.length files > 0 then
        let stats = Array.map analyze_frame files in
        test_model_on_data model stats;
        
      (* Interactive testing *)
      interactive_test_model model
      
  | None ->
      (* No model loaded, proceed with normal analysis *)
      printf "No pointing model loaded. Proceeding with standard analysis.\n";
      analyze_frames files flags

(* Parse command line arguments *)
let parse_args () =
  let flags = {
    show_temp_plot = false;
    show_dist_plot = false;
    show_error_plot = false;
    show_stats = false;
    show_coeff = false;
    temp_range = None;
    dry_run = false;
    base_dir = "frame_temps";
    json_dir = None;
    build_model = false;
  } in
  let files = ref [] in
  let model_file = ref None in
  let specs = [
    ("-temp", Arg.Unit (fun () -> flags.show_temp_plot <- true), 
     "Show temperature vs time plot");
    ("-dist", Arg.Unit (fun () -> flags.show_dist_plot <- true),
     "Show temperature distribution");
    ("-error", Arg.Unit (fun () -> flags.show_error_plot <- true),
     "Show pointing error plots");
    ("-stats", Arg.Unit (fun () -> flags.show_stats <- true),
     "Show RA/DEC analysis");
    ("-coeff", Arg.Unit (fun () -> flags.show_coeff <- true),
     "Show temperature coefficient");
    ("-json", Arg.String (fun dir -> 
                flags.json_dir <- Some dir; 
                flags.build_model <- true),
     "Directory containing JSON files for pointing model");
    ("-model", Arg.Unit (fun () -> flags.build_model <- true),
     "Build quaternion pointing model");
    ("-save", Arg.String (fun file -> model_file := Some (`Save file)),
     "Save model to file");
    ("-load", Arg.String (fun file -> model_file := Some (`Load file)),
     "Load model from file");
    ("-dry-run", Arg.Unit (fun () -> flags.dry_run <- true),
     "Show what would be done without actually moving files");
    ("-outdir", Arg.String (fun dir -> flags.base_dir <- dir),
     "Base directory for sorted files (default: frame_temps)");
    ("-min-temp", Arg.Float (fun min_t -> flags.temp_range <- Some(min_t, 0.0)),
     "Set minimum temperature for analysis");
    ("-max-temp", Arg.Float (fun max_t -> 
       flags.temp_range <- 
         match flags.temp_range with 
         | Some(min_t,_) -> Some(min_t, max_t)
         | None -> Some(0.0, max_t)),
     "Set maximum temperature for analysis")
  ] in
  let add_file f = files := f :: !files in
  let usage = sprintf "Usage: %s [-temp] [-dist] [-error] [-stats] [-coeff] [-json dir] [-model] [-save file] [-load file] [-dry-run] [-outdir dir] [-min-temp val] [-max-temp val] <fits_file1> [fits_file2 ...]\n" Sys.argv.(0) in
  Arg.parse specs add_file usage;
  (flags, Array.of_list (List.rev !files), !model_file)

(* Process JSON files to build pointing model *)
let process_json_files json_dir =
  printf "Processing JSON files in %s\n" json_dir;
  
  (* Create empty pointing model *)
  let model = PointingModel.create_empty_model () in
  
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
  
  (* Process pointing_deep_sky.json files first to find target info *)
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
  let astrometry_data = List.fold_left (fun acc file ->
    if Filename.basename file = "astrometry.json" then
      match JsonParser.parse_astrometry_json file with
      | Some data ->
          (* Try to find target info from the directory *)
          let dir = Filename.dirname (Filename.dirname file) in
          (match Hashtbl.find_opt target_map dir with
           | Some (target_ra, target_dec) ->
               (* Update with target info *)
               let updated_data = { data with 
                 solved_ra = target_ra; 
                 solved_dec = target_dec;
		 src_file = file
               } in
               updated_data :: acc
           | None -> data :: acc)
      | None -> acc
    else
      acc
  ) [] json_files in
  
  printf "Successfully parsed %d astrometry files\n" (List.length astrometry_data);
  
  (* Estimate temperature based on timestamp if necessary *)
  let estimate_temp timestamp =
    (* Simple temperature model - this can be improved with actual data *)
    let hour = mod_float (timestamp /. 3600.0) 24.0 in
    (* Temperature range from 15°C at night to 25°C during day *)
    if hour > 6.0 && hour < 18.0 then
      (* Daytime rising temperature *)
      15.0 +. 10.0 *. (sin ((hour -. 6.0) /. 12.0 *. 3.14159))
    else
      (* Nighttime falling temperature *)
      15.0
  in
  
  (* Add reference points to the model *)
  let model_with_points = List.fold_left (fun m (data:astrometry_data) ->
    (* Use the default temperature if not specified *)
    let temp = 20.0 in
    
    (* Add the reference point *)
    PointingModel.add_reference_point m data.ra data.dec 
      data.solved_ra data.solved_dec temp data.map
      data.timestamp data.src_file
  ) model astrometry_data in
  
  (* Calculate temperature coefficients *)
  let final_model = calculate_temp_coefficients model_with_points in
  
  (* Evaluate model accuracy *)
  let mean_error = PointingModel.evaluate_model final_model in
  printf "Pointing model built with %d reference points\n" 
    (List.length final_model.reference_points);
  printf "Mean prediction error: %.4f degrees\n" mean_error;
  
  (* Return the model *)
  final_model
  
(* Main entry point *)
let () =
  let flags, files, model_file_opt = parse_args () in
  Quaternion.test_rotation();
  
  (* Handle model file operations *)
  match model_file_opt with
  | Some (`Load filename) ->
      (* Load model and analyze with it *)
      analyze_with_model files flags (Some filename)
      
  | Some (`Save filename) ->
      if flags.build_model && flags.json_dir <> None then
        (* Build model from JSON files and save it *)
        let model = process_json_files (Option.get flags.json_dir) in
        save_model_to_file model filename;
        (* Optionally analyze files with the new model *)
        if Array.length files > 0 then
          analyze_with_model files flags (Some filename)
        else
          ()
      else
        failwith "Cannot save model: No JSON directory specified or build_model not enabled"
        
  | None ->
      (* Normal operation without loading/saving model *)
      if Array.length files = 0 && flags.json_dir = None then
        failwith "No input files or JSON directory specified"
      else if flags.show_temp_plot && not (flags.show_dist_plot || flags.show_stats || 
               flags.show_coeff || flags.show_error_plot || flags.build_model) then
        (* Fast path: only temperature scan needed *)
        let temp_infos = Array.to_list files 
          |> List.filter_map scan_fits_temperature 
          |> Array.of_list in
        if Array.length temp_infos > 0 then
          let stats = Array.map (fun (ti:temp_info) -> 
            {filename = ti.filename; 
             temp = ti.temp;
             qhdr = ti.qhdr})
            temp_infos in
          plot_temp_vs_time stats
        else
          failwith "No valid temperature data found in files"
      else if flags.build_model && flags.json_dir <> None && Array.length files = 0 then
        (* Build pointing model from JSON files only *)
        let model = process_json_files (Option.get flags.json_dir) in
        (* Plot model stats *)
        plot_model_accuracy model;
        (* Interactive testing *)
        interactive_test_model model
      else
        (* Full analysis with or without building model *)
        analyze_frames files flags
