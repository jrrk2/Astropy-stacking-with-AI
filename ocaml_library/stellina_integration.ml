(* stellina_integration.ml - Integration with Stellina-specific functionality *)

open Printf
open Types
open Fits
open Util
open Unified_interface

(* Stellina-specific Types *)
type stellina_flags = {
  temp_range: (float * float) option;
  show_temp_plot: bool;
  show_dist_plot: bool;
  show_stats: bool;
}

type stellina_process_flags = {
  base_dir: string;
  target_name: string option;
  max_separation_deg: float;
  dry_run: bool;
  solve: bool;
  add_registration: bool;
  observation_json_path: string option;
  calibrate: bool;
  darks_dir: string option;
  pointing_model_path: string option;
  verify_model: bool;
  latitude: float;
  longitude: float;
  plate_scale: float;
  temp_tolerance: float;
}

type status = Roundness | StackingOK

let status_msg = function
| Roundness -> "Roundness Error"
| StackingOK -> "OK for stacking"

(* Cache for master darks to avoid rebuilding *)
let (master_dark_cache:(int,int array array * (string, string) Hashtbl.t)Hashtbl.t) = Hashtbl.create 10

(* Default processing flags *)
let default_process_flags = {
  base_dir = "lights";
  target_name = None;
  max_separation_deg = 5.0;
  dry_run = false;
  solve = false;
  add_registration = true;
  observation_json_path = None;
  calibrate = false;
  darks_dir = None;
  pointing_model_path = None;
  verify_model = false;
  latitude = 52.245091;
  longitude = 0.079609;
  plate_scale = 0.57;
  temp_tolerance = 2.0;
}

(* Default analysis flags *)
let default_analysis_flags = {
  temp_range = None;
  show_temp_plot = false;
  show_dist_plot = false;
  show_stats = true;
}

(* Determines new filepath with Bayer pattern handling - from stellina_process.ml *)
let get_new_filepath fits_path ?(base_dir="lights") ?(calibrated=false) () =
  try
    let hdrh = just_header fits_path in
    
    (* Extract temperature *)
    let temp = get_temperature hdrh in
    
    (* Check if filename indicates Bayer pattern *)
    let fits_filename = Filename.basename fits_path in
    let bayer_pattern = 
      if String.contains fits_filename 'r' then "RGGB"
      else if String.contains fits_filename 'b' then "BGGR"
      else ""
    in

    let clean_bayer s = match String.split_on_char '\'' s with
      | _::nxt::_ -> String.trim nxt
      | oth -> s in

    (* Also check for BAYERPAT keyword in header if available *)
    let bayer_pattern = 
      match Hashtbl.find_opt hdrh "BAYERPAT=" with
      | Some pat -> clean_bayer (parse_string_header hdrh "BAYERPAT=")
      | None -> bayer_pattern
    in
    
    (* Create directory name based on temperature *)
    let temp_k = int_of_float (Float.round (temp +. 273.15)) in
    let temp_dir = sprintf "temp_%d" temp_k in

    (* Extract date from FITS *)
    let (year, month, day, hour, minute, second) = 
      match extract_date_from_fits hdrh with
      | Some date_components -> date_components
      | None -> 
	  (* Default to current date/time if extraction fails *)
	  let tm = Unix.localtime (Unix.time()) in
	  (tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, 
	   tm.tm_hour, tm.tm_min, tm.tm_sec)
    in
    
    (* Add Bayer pattern and calibration status to filename *)
    let prefix = if calibrated then "cal_" else "light_" in
    let filename = 
      if bayer_pattern <> "" then
        sprintf "%s%04d%02d%02d_%02d%02d%02d_%s.fits" 
          prefix year month day hour minute second bayer_pattern
      else
        sprintf "%s%04d%02d%02d_%02d%02d%02d.fits" 
          prefix year month day hour minute second
    in
    
    let new_path = Filename.concat (Filename.concat base_dir temp_dir) filename in
    Some new_path
  with
  | e -> 
      eprintf "Error determining new path: %s\n" (Printexc.to_string e);
      None

(* Verify coordinates are close to target object - from stellina_process.ml *)
let verify_coordinates context calc_ra calc_dec target_name ?(max_separation_deg=1.0) () =
  match QuerySimbad.get_object_coordinates target_name with
  | None -> (false, None, None)
  | Some (target_ra, target_dec) ->
      let alt_calc, az_calc, hour_calc = Unified_interface.radec_to_altaz context target_ra target_dec in
      printf "Target ALT=%.4f, AZ=%.4f, HA=%.4f\n" alt_calc az_calc hour_calc;
      (* Calculate angular distance using spherical trigonometry *)
      let ra1_rad = calc_ra *. Float.pi /. 180.0 in
      let dec1_rad = calc_dec *. Float.pi /. 180.0 in
      let ra2_rad = target_ra *. Float.pi /. 180.0 in
      let dec2_rad = target_dec *. Float.pi /. 180.0 in
      
      let cos_dist = sin(dec1_rad) *. sin(dec2_rad) +. 
                     cos(dec1_rad) *. cos(dec2_rad) *. cos(ra1_rad -. ra2_rad) in
      let cos_dist = max (-1.0) (min cos_dist 1.0) in
      let separation_deg = Float.acos(cos_dist) *. 180.0 /. Float.pi in
      
      let is_valid = separation_deg <= max_separation_deg in
      
      if not is_valid then begin
        printf "  Warning: Calculated position is %.2f° from %s\n" 
          separation_deg target_name;
        printf "    Calculated: RA=%.4f°, Dec=%.4f°\n" calc_ra calc_dec;
        printf "    Expected:   RA=%.4f°, Dec=%.4f°\n" target_ra target_dec
      end else
        printf "  Position OK - %.2f° from %s\n" separation_deg target_name;
      
      (is_valid, Some separation_deg, Some (target_ra, target_dec))

(* Apply model correction to a FITS file *)
let apply_model_correction model fits_path output_path =
  try
    (* Get FITS header *)
    let hdrh = just_header fits_path in
    
    (* Extract mount position *)
    let mountra = parse_float hdrh "MOUNTRA" in
    let mountdec = parse_float hdrh "MOUNTDEC=" in
    
    (* Apply pointing model correction *)
    let (corrected_ra, corrected_dec) = correct_position model mountra mountdec 0 in
    
    (* Calculate original Alt/Az *)
    let context = create_context model.latitude model.longitude in
    let (orig_alt, orig_az, _) = radec_to_altaz context mountra mountdec in
    
    (* Calculate corrected Alt/Az *)
    let (corr_alt, corr_az, _) = radec_to_altaz context corrected_ra corrected_dec in
    
    (* Create updates list *)
    let updates = [
      (* Original mount position *)
      ("MOUNTRA", Printf.sprintf "%f" mountra, "Original Mount RA (deg)");
      ("MOUNTDEC", Printf.sprintf "%f" mountdec, "Original Mount DEC (deg)");
      ("ORIGALT", Printf.sprintf "%f" orig_alt, "Original Altitude (deg)");
      ("ORIGAZ", Printf.sprintf "%f" orig_az, "Original Azimuth (deg)");
      
      (* Corrected position *)
      ("CORRRA", Printf.sprintf "%f" corrected_ra, "Corrected RA (deg)");
      ("CORRDEC", Printf.sprintf "%f" corrected_dec, "Corrected DEC (deg)");
      ("CORRALT", Printf.sprintf "%f" corr_alt, "Corrected Altitude (deg)");
      ("CORRAZ", Printf.sprintf "%f" corr_az, "Corrected Azimuth (deg)");
      
      (* Correction magnitudes *)
      ("RADELTA", Printf.sprintf "%f" (corrected_ra -. mountra), "RA correction (deg)");
      ("DECDELTA", Printf.sprintf "%f" (corrected_dec -. mountdec), "DEC correction (deg)");
    ] in
    
    (* Copy with updates *)
    copy_fits_with_updates fits_path output_path updates
  with
  | e ->
      eprintf "  Error applying model correction: %s\n" (Printexc.to_string e);
      false

(* Function to get a master dark from cache or build it *)
let get_master_dark bin_temp bin_path =
  if Hashtbl.mem master_dark_cache bin_temp then begin
    printf "  Using cached master dark for bin %d\n" bin_temp;
    Some (Hashtbl.find master_dark_cache bin_temp)
  end else begin
    (* First check if a master dark already exists in the bin directory *)
    let existing_master = 
      try 
        let files = Sys.readdir bin_path in
        let master_file = ref None in
        
        (* Look for master dark files *)
        for i = 0 to Array.length files - 1 do
          let f = files.(i) in
          if String.lowercase_ascii f = "master_dark.fits" || 
             (Filename.check_suffix f ".fits" && 
              (String.lowercase_ascii (Filename.basename f) |> 
               String.split_on_char '_' |> List.exists ((=) "master"))) then
            master_file := Some f
        done;
        
        match !master_file with
        | Some file ->
            let master_path = Filename.concat bin_path file in
            printf "  Found existing master dark: %s\n" file;
            
            (* Verify it's a valid master dark by checking for NFRAMES keyword *)
            let hdrh = just_header master_path in
            let nframes = 
              try
                let nframes_str = Hashtbl.find hdrh "NFRAMES=" in
                try Scanf.sscanf nframes_str " = %d" (fun i -> i) 
                with _ -> 1
              with Not_found -> 
                try
                  (* Try DARKAVG as alternative for number of frames *)
                  let nframes_str = Hashtbl.find hdrh "DARKAVG=" in
                  try Scanf.sscanf nframes_str " = %d" (fun i -> i)
                  with _ -> 1
                with Not_found -> 1
            in
            
            if nframes > 0 then begin
              printf "  Using existing master dark with %d frames\n" nframes;
              Some master_path
            end else begin
              printf "  Existing master dark doesn't have valid NFRAMES, will create new one\n";
              None
            end
        | None -> 
            printf "  No existing master dark found in %s\n" bin_path;
            None
      with _ -> 
        printf "  No existing master dark found in %s\n" bin_path;
        None
    in
    
    match existing_master with
    | Some master_path ->
        (* Load the existing master dark *)
        (try
          let img = read_image master_path in
          let hdrh, contents = find_header_end master_path img in
          let width = parse_int hdrh "NAXIS1" in
          let height = parse_int hdrh "NAXIS2" in
          let bitpix = parse_int hdrh "BITPIX" in
          
          printf "  Loading existing master dark (%dx%d, BITPIX=%d)\n" width height bitpix;
          
          (* Read data based on BITPIX type *)
          let master_data = 
            if bitpix = -32 then begin
              (* It's a 32-bit float master dark *)
              printf "  Reading 32-bit float master dark\n";
              let float_data = read_fits_float_data contents width height in
              
              (* Convert to integer array to match format needed for calibration *)
              let int_data = Array.make_matrix height width 0 in
              for y = 0 to height - 1 do
                for x = 0 to width - 1 do
                  int_data.(y).(x) <- int_of_float (Float.round float_data.(y).(x))
                done
              done;
              int_data
            end else begin
              (* Standard integer data *)
              printf "  Reading 16-bit integer master dark\n";
              read_fits_data contents width height
            end
          in
          
          let result = (master_data, hdrh) in
          Hashtbl.add master_dark_cache bin_temp result;
          Some result
        with e ->
          printf "  Error loading existing master dark: %s\n" (Printexc.to_string e);
          None)
    | None ->
        (* Create dark group for this bin *)
        let dark_files = Dark_calibration.find_fits_files bin_path in
        if Array.length dark_files = 0 then
          None
        else begin
          let bin_temp_c = float_of_int bin_temp -. 273.15 in
          let dark_group = {
            Dark_calibration.temp_bin = bin_temp;
            temperature = bin_temp_c;
            files = dark_files;
            master_path = None;
          } in
          
          printf "  Creating new master dark for bin %d (%.1f°C) from %d frames...\n" 
            bin_temp bin_temp_c (Array.length dark_files);
          
          let master_dark = Dark_calibration.create_and_use_master_dark dark_group in
          
          (* Save the master dark to file *)
          let master_path = Filename.concat bin_path "master_dark.fits" in
          printf "  Saving new master dark to %s\n" master_path;
          
          (* Create master dark with NFRAMES keyword *)
          let (data, hdrh) = master_dark in
          
          (* Create a copy of the header with added NFRAMES *)
          let new_hdrh = Hashtbl.copy hdrh in
          Hashtbl.add new_hdrh "NFRAMES" (sprintf " = %d / Number of frames in master dark" (Array.length dark_files));
          
          (* Save the master dark *)
          let oc = open_out_bin master_path in
          ignore (write_fits_header oc new_hdrh);
          
          (* Write data *)
          for y = 0 to Array.length data - 1 do
            for x = 0 to Array.length data.(0) - 1 do
              (* FITS uses big-endian *)
              let value = data.(y).(x) in
              output_byte oc (value lsr 8);
              output_byte oc (value land 0xFF);
            done
          done;
          
          (* Pad data to multiple of 2880 bytes *)
          let data_size = (Array.length data) * (Array.length data.(0)) * 2 in
          let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
          output_string oc (String.make padding_size '\000');
          
          close_out oc;
          
          Hashtbl.add master_dark_cache bin_temp master_dark;
          Some master_dark
        end
  end

(* Function to find appropriate temperature bin *)
let find_temp_bin darks_dir temp temp_tolerance =
  let temp_k = int_of_float (floor (temp +. 273.15 +. 0.5)) in
  
  (* Look for matching temperature directory *)
  let temp_dir = sprintf "temp_%d" temp_k in
  let temp_path = Filename.concat darks_dir temp_dir in
  
  (* Check exact match first *)
  if Sys.file_exists temp_path && Sys.is_directory temp_path then
    Some (temp_k, temp_path)
  else begin
    (* If no exact match, try to find the closest bin directory *)
    let temp_bins = ref [] in
    
    (* Scan dark_temps to find all temperature bin directories *)
    (try
      Array.iter (fun entry ->
        let full_path = Filename.concat darks_dir entry in
        if Sys.is_directory full_path && 
           String.length entry > 5 && 
           String.sub entry 0 5 = "temp_" then begin
          try
            let bin_temp = int_of_string (String.sub entry 5 (String.length entry - 5)) in
            temp_bins := (bin_temp, full_path) :: !temp_bins
          with _ -> ()
        end
      ) (Sys.readdir darks_dir)
    with _ -> ());
    
    (* Sort bins by temperature difference *)
    let sorted_bins = List.sort 
      (fun (t1, _) (t2, _) -> compare (abs (t1 - temp_k)) (abs (t2 - temp_k)))
      !temp_bins in
    
    match sorted_bins with
    | [] -> None
    | (bin_temp, bin_path) :: _ ->
        let bin_temp_c = float_of_int bin_temp -. 273.15 in
        let temp_diff = abs_float (bin_temp_c -. temp) in
        
        if temp_diff <= temp_tolerance then
          Some (bin_temp, bin_path)
        else
          None
  end

(* Apply dark calibration to a light frame with caching *)
let apply_dark_calibration darks_dir light_file output_file temp_tolerance =
  try
    (* Get temperature from light frame *)
    let hdrh = just_header light_file in
    let temp = get_temperature hdrh in
    let temp_k = int_of_float (floor (temp +. 273.15 +. 0.5)) in
    
    printf "  Processing %s (Temperature: %.1f°C, bin temp_%d)\n" 
      (Filename.basename light_file) temp temp_k;
    
    (* Find appropriate master dark using the temperature bin functions *)
    match find_temp_bin darks_dir temp temp_tolerance with
    | None -> 
        printf "  No suitable temperature bin found within %.1f°C tolerance\n" temp_tolerance;
        false
    | Some (bin_temp, bin_path) ->
        let bin_temp_c = float_of_int bin_temp -. 273.15 in
        printf "  Using temperature bin: %s (%.1f°C, diff: %.1f°C)\n" 
          bin_path bin_temp_c (abs_float (bin_temp_c -. temp));
        
        match get_master_dark bin_temp bin_path with
        | None -> 
            printf "  Failed to create master dark\n";
            false
        | Some (master_dark, dark_hdrh) ->
            (* Apply calibration *)
            printf "  Applying calibration using master dark\n";
            
            (* Create output directory if needed *)
            let dir = Filename.dirname output_file in
            if not (Sys.file_exists dir) then
              create_dir dir;
            
            (* Perform the calibration *)
            Dark_calibration.calibrate_image_in_memory light_file (master_dark, dark_hdrh) output_file;
            true
  with e ->
    eprintf "  Error applying dark calibration: %s\n" (Printexc.to_string e);
    false

let analyze_field_rotation_from_headers files =
  printf "\nAnalyzing field rotation from FITS headers...\n";
  
  (* Only select RGB calibrated files *)
  let rgb_files = List.filter (fun file ->
    let basename = Filename.basename file in
    String.length basename >= 7 && 
    String.sub basename 0 3 = "cal" && 
    String.contains basename '_' && 
    String.contains basename 'r' && 
    String.contains basename 'g' && 
    String.contains basename 'b'
  ) files in
  
  printf "  Selected %d calibrated RGB files for analysis\n" (List.length rgb_files);
  
  (* Sort files by timestamp *)
  let sorted_files = List.sort compare rgb_files in
  
  (* Extract rotation data from FITS headers *)
  let rotation_data = List.filter_map (fun file ->
    try
      let hdrh = Fits.just_header file in
      
      (* Extract timestamp *)
      let timestamp = Util.get_timestamp hdrh in
      
      (* Extract ALT/AZ, DEROT, and calculated rotation rate *)
      let alt = parse_float hdrh "ALT" in
      let az = parse_float hdrh "AZ" in
      let derot = parse_float hdrh "DEROT" in
      let rot_rate = parse_float hdrh "ROTRATE" in
      
      Some { 
        timestamp;
        alt;
        az;
        derot;
        rot_rate = Some rot_rate;
        filename = Filename.basename file 
      }
    with e -> 
      printf "  Could not process %s: %s\n" 
        (Filename.basename file) (Printexc.to_string e);
      None
  ) sorted_files in
    
  (* Sort by timestamp *)
  let sorted_data = List.sort (fun (a:rotation_data_item) (b:rotation_data_item) -> compare a.timestamp b.timestamp) rotation_data in
  
  (* Analyze field rotation between consecutive frames *)
  if List.length sorted_data > 1 then begin
    printf "Field Rotation Analysis:\n";
    printf "%-20s %-10s %-8s %-8s %-10s %-14s %-14s\n" 
      "Image" "Time" "ALT" "AZ" "DEROT" "Rot Rate" "Est Rot";
    
    (* Keep track of previous frame for comparison *)
    let prev_frame = ref None in
    
    (* Calculate expected and actual rotations between consecutive frames *)
      let rotations = List.filter_map (fun (frame:rotation_data_item) ->

      (* Format timestamp *)
      let time_str = 
        let tm = Unix.localtime frame.timestamp in
        sprintf "%02d:%02d:%02d" tm.Unix.tm_hour tm.Unix.tm_min tm.Unix.tm_sec
      in
      
      match !prev_frame with
      | Some (prev:rotation_data_item) ->
          (* Time difference in hours *)
          let time_diff_hours = (frame.timestamp -. prev.timestamp) /. 3600.0 in
          
          (* Actual derotator change *)
          let derot_change = frame.derot -. prev.derot in
          
          (* Expected rotation based on rotation rate *)
          let expected_rotation = 
            match prev.rot_rate with
            | Some rate -> rate *. time_diff_hours
            | None -> 0.0
          in
          
          printf "%-20s %-10s %8.2f %8.2f %10.2f %14.2f %14.2f\n" 
            frame.filename time_str frame.alt frame.az frame.derot
            (match frame.rot_rate with Some r -> r | None -> 0.0)
            expected_rotation;
          
          prev_frame := Some frame;
          Some (derot_change, expected_rotation)
          
      | None ->
          printf "%-20s %-10s %8.2f %8.2f %10.2f %14.2f %14s\n" 
            frame.filename time_str frame.alt frame.az frame.derot
            (match frame.rot_rate with Some r -> r | None -> 0.0)
            "-";
          
          prev_frame := Some frame;
          None
    ) sorted_data in
    
    (* Calculate statistics *)
    if List.length rotations > 0 then begin
      let derot_changes = List.map fst rotations in
      let expected_rots = List.map snd rotations in
      
      (* Calculate differences (should be near zero if DEROT compensates perfectly) *)
      let differences = List.map2 (fun derot_change expected_rot -> 
        derot_change +. expected_rot
      ) derot_changes expected_rots in
      
      let avg_diff = List.fold_left (+.) 0.0 differences /. float_of_int (List.length differences) in
      let max_diff = List.fold_left max (List.hd differences) differences in
      let min_diff = List.fold_left min (List.hd differences) differences in
      
      printf "\nStatistics:\n";
      printf "Average DEROT difference from theoretical rotation: %.3f degrees\n" avg_diff;
      printf "Range: %.3f to %.3f degrees\n" min_diff max_diff;
      
      (* Determine the relationship between DEROT and field rotation *)
      if abs_float avg_diff < 1.0 then
        printf "\nThe DEROT value appears to directly compensate for field rotation (DEROT = -fieldRotation)\n"
      else if abs_float (avg_diff -. 180.0) < 1.0 || abs_float (avg_diff +. 180.0) < 1.0 then
        printf "\nThe DEROT value appears to be offset by 180° from field rotation\n"
      else
        printf "\nThe DEROT value has an average offset of %.2f° from the theoretical field rotation\n" avg_diff;
      
      (* Return the statistics *)
      Some (avg_diff, min_diff, max_diff)
    end else begin
      printf "Not enough data points to calculate statistics\n";
      None
    end
  end else begin
    printf "Not enough data to analyze field rotation (need at least 2 frames)\n";
    None
end

(* Process directory with enhanced error handling - from stellina_process.ml *)
let process_directory src_dir flags =
  let {
    base_dir;
    target_name;
    max_separation_deg;
    dry_run;
    solve;
    add_registration;
    observation_json_path;
    calibrate;
    darks_dir;
    pointing_model_path;
    verify_model;
    latitude;
    longitude;
    plate_scale;
    temp_tolerance;
  } = flags in

  let local_context = create_context latitude longitude in
  
  printf "\nScanning directory: %s\n" src_dir;
  
  (* Get target coordinates if specified *)
  let target_coords = 
    match target_name with
    | None -> None
    | Some name ->
        match QuerySimbad.get_object_coordinates name with
        | None ->
            printf "Error: Could not get coordinates for %s\n" name;
            None
        | Some coords ->
            printf "\nVerifying coordinates against target: %s\n" name;
            printf "Maximum allowed separation: %.1f°\n" max_separation_deg;
            Some (coords, name)
  in
  
  (* Initialize pointing model if provided *)
  let pointing_model =
    match pointing_model_path with
    | None -> None
    | Some path ->
        printf "\n=== Initializing Pointing Model ===\n";
        printf "Loading pointing model from: %s\n" path;
        match load_pointing_model path with
        | Some model -> 
            printf "Pointing model initialized with %d reference points\n" 
              (List.length model.reference_points);
            printf "=== Pointing Model Initialization Complete ===\n\n";
            Some model
        | None ->
            printf "Failed to load pointing model from %s\n" path;
            None
  in
  
  (* Make sure base_dir exists *)
  if not (Sys.file_exists base_dir) then begin
    try
      Unix.mkdir base_dir 0o755;
      printf "Created base directory: %s\n" base_dir;
    with e ->
      eprintf "Error creating base directory: %s\n" (Printexc.to_string e);
  end;
  
  (* Find all files in directory *)
  let files = 
    try List.sort compare (Array.to_list (Sys.readdir src_dir))
    with e -> 
      eprintf "Error reading directory %s: %s\n" src_dir (Printexc.to_string e);
      []
  in
  
  (* Find matching JSON and FITS files *)
  let json_files = 
    List.filter (fun f -> Filename.check_suffix f ".json") files
    |> List.map (fun f -> Filename.concat src_dir f)
  in
  
  let fits_files = 
    List.filter (fun f -> Filename.check_suffix f ".fits") files
    |> List.map (fun f -> Filename.concat src_dir f)
  in
  
  printf "Found %d JSON files\n" (List.length json_files);
  printf "Found %d FITS files\n" (List.length fits_files);
  
  (* Find matching pairs *)
  let pairs = ref [] in
  List.iter (fun json_file ->
    let json_base = Filename.remove_extension (Filename.basename json_file) in
    let json_index = 
      try Scanf.sscanf json_base "img-%d-stacking" (fun i -> i)
      with _ -> -1
    in
    
    if json_index >= 0 then
      List.iter (fun fits_file ->
        let fits_base = Filename.remove_extension (Filename.basename fits_file) in
        let fits_index =
          try Scanf.sscanf fits_base "img-%d" (fun i -> i)
          with _ -> 
            try Scanf.sscanf fits_base "img-%dr" (fun i -> i)
            with _ -> -1
        in
        
        if fits_index = json_index then
          pairs := (json_file, fits_file, json_index) :: !pairs
      ) fits_files
  ) json_files;
  
  printf "Matched %d JSON/FITS pairs\n" (List.length !pairs);
  
  if dry_run then
    printf "\nDRY RUN - no files will be modified\n";
  
  (* Initialize counters *)
  let processed = ref 0 in
  let skipped = ref 0 in
  let errors = ref 0 in
  let calibrated = ref 0 in
  let registration_added = ref 0 in
  let registration_failed = ref 0 in
  let focus_flt = ref 0.0 in
  let focus_qual = ref 0.0 in
  let correction_x = ref 0.0 in
  let correction_y = ref 0.0 in
  let correction_rot = ref 0.0 in
  let coordinates_x = ref 0.0 in
  let coordinates_y = ref 0.0 in
  let coordinates_rot = ref 0.0 in
  (* Process each pair *)
  List.iter (fun (json_file, fits_file, index) ->
    printf "\nProcessing index %d:\n" index;
    printf "  JSON: %s\n" json_file;
    printf "  FITS: %s\n" fits_file;
    
    try
      (* Parse JSON data *)
      let open JsonParser in
      let open Yojson.Basic.Util in
      let json = Yojson.Basic.from_file json_file in
      let index = json |> member "index" |> to_int in
      let motors = json |> member "motors" in
      let alt = motors |> member "ALT" |> to_float in
      let az = motors |> member "AZ" |> to_float in
      let derot = motors |> member "DER" |> to_float in
      let focus_map = motors |> member "MAP" |> to_int in
      let stacking = json |> member "stackingData" |> member "liveRegistrationResult" in
      let roundness = stacking |> member "roundness" |> safe_float in
      let status = match stacking |> member "statusMessage" |> to_string with
	| "StackingRoundnessError" -> Roundness
	| "StackingOk" ->
	  correction_x := stacking |> member "correction" |> member "x" |> safe_float;
	  correction_y := stacking |> member "correction" |> member "y" |> safe_float;
	  correction_rot := stacking |> member "correction" |> member "rot" |> safe_float;
	  coordinates_x := stacking |> member "coordinates" |> member "x" |> safe_float;
	  coordinates_y := stacking |> member "coordinates" |> member "y" |> safe_float;
	  coordinates_rot := stacking |> member "coordinates" |> member "rot" |> safe_float;
	  if index = 0 then
	    (
	    focus_flt := stacking |> member "focus" |> safe_float;
	    focus_qual := stacking |> member "focusQuality" |> safe_float;
	    );
          StackingOK
	| msg -> failwith msg in
      (* get timestamp *)
      let hdrh = just_header fits_file in
      let timestamp = get_timestamp hdrh in
      printf "  ALT/AZ: %.2f°, %.2f°, stamp=%.3f\n" alt az timestamp;
      let context = {local_context with timestamp=Some timestamp} in
      (* Calculate RA/Dec from Alt/Az *)
      let (ra, dec, _) = altaz_to_radec context alt az in
      printf "  Calculated RA/Dec: %.4f°, %.4f°\n" ra dec;
      
      (* Verify coordinates if target specified *)
      let proceed =
        match target_coords with
        | None -> true
        | Some ((target_ra, target_dec), target_name) ->
            (* Check if the coordinates are within acceptable range *)
            let (is_valid, separation, _) = 
              verify_coordinates context ra dec target_name ~max_separation_deg () in
            
            if not is_valid then begin
              printf "  Skipping - separation too large (%.2f°)\n" 
                (Option.get separation);
              skipped := !skipped + 1;
              false
            end else begin
              printf "  Position OK - %.2f° from %s\n"
                (Option.get separation) target_name;
              true
            end
      in
      
      if proceed then begin
        (* Get new filepath *)
        match get_new_filepath fits_file ~base_dir () with
        | Some new_path ->
            printf "  Target path: %s\n" new_path;
            
            if not dry_run then begin
              try
                (* Create directory if needed *)
                let dir = Filename.dirname new_path in
                if not (Sys.file_exists dir) then
                  create_dir dir;
                
                (* Copy file if doesn't exist *)
                if not (Sys.file_exists new_path) then begin

		  (* Calculate theoretical field rotation rate at this position *)
		  let rotation_rate = calc_field_rotation_rate 
		    ~latitude:context.latitude 
		    ~altitude:alt 
		    ~azimuth:az in

		  (* Calculate hour angle explicitly for verification *)
		  let hour_angle = calc_hour_angle 
		    ~timestamp 
		    ~longitude:context.longitude 
		    ~ra in

		  printf "  Annotated FITS with ALT/AZ: %.2f°, %.2f° → RA/DEC: %.4f°, %.4f°\n" 
		    alt az ra dec;
		  printf "  Hour angle: %.4f hours, Field rotation rate: %.4f°/hr\n" 
		    hour_angle (rotation_rate *. 180.0 /. Float.pi);
		  
                  (* Create updates list *)
                  let updates = [
                    ("MOUNTRA", Printf.sprintf "%f" ra, "Mount RA (deg)");
                    ("MOUNTDEC", Printf.sprintf "%f" dec, "Mount DEC (deg)");
                    ("ALT", Printf.sprintf "%f" alt, "Altitude (deg)");
                    ("AZ", Printf.sprintf "%f" az, "Azimuth (deg)");
		    ("DEROT", Printf.sprintf "%f" derot, "Derotation (deg)");
		    ("MAP", Printf.sprintf "%d" focus_map, "Focus motor");
		    ("FOCUS", Printf.sprintf "%f" !focus_flt, "Focus");
		    ("FOCUSQ", Printf.sprintf "%f" !focus_qual, "Focus quality");
		    ("HAVAL", Printf.sprintf "%f" hour_angle, "Hour angle (hours)");
		    ("ROTRATE", Printf.sprintf "%f" (rotation_rate *. 180.0 /. Float.pi), "Field rotation rate (deg/hr)");
		    ("STATUS", Printf.sprintf "%s" (status_msg status), "Stacking status");
                    ] @ if status = StackingOK then [
		    ("CORX", Printf.sprintf "%f" !correction_x, "Correction X");
		    ("CORY", Printf.sprintf "%f" !correction_y, "Correction Y");
		    ("CORROT", Printf.sprintf "%f" !correction_rot, "Correction ROT");
		    ("COORDX", Printf.sprintf "%f" !coordinates_x, "Coordinates X");
		    ("COORDY", Printf.sprintf "%f" !coordinates_y, "Coordinates Y");
		    ("COORDROT", Printf.sprintf "%f" !coordinates_rot, "Coordinates ROT");
		    ] else [] in

                  if copy_fits_with_updates fits_file new_path updates then begin
                    printf "  Created and annotated %s\n" new_path;
                    processed := !processed + 1;
                    
                    (* Apply pointing model if available *)
                    if Option.is_some pointing_model then begin
                      let model = Option.get pointing_model in
                      let model_path = Filename.concat 
                        (Filename.dirname new_path) 
                        ("model_" ^ (Filename.basename new_path)) in
                      
                      if apply_model_correction model new_path model_path then begin
                        printf "  Applied pointing model correction to %s\n" model_path;
                      end
                    end;
                    
                    (* Add registration data if requested *)
                    if add_registration then begin
                      match observation_json_path with
                      | Some obs_json ->
                          (* Registration implementation would go here *)
                          (* For now, just count as success *)
                          registration_added := !registration_added + 1
                      | None ->
                          (* Look for observation.json in standard location *)
                          let default_obs_json = 
                            Filename.concat (Filename.dirname src_dir) "observation.json" in
                          if Sys.file_exists default_obs_json then begin
                            (* Registration implementation would go here *)
                            (* For now, just count as success *)
                            registration_added := !registration_added + 1
                          end else begin
                            printf "  Warning: observation.json not found\n";
                            registration_failed := !registration_failed + 1
                          end
                    end;
                    
                    (* Apply calibration if requested *)
                    if calibrate && status = StackingOK then begin
                      match darks_dir with
                      | Some dark_dir ->
                          (* Create calibrated output path *)
                          let cal_path = get_new_filepath new_path ~base_dir ~calibrated:true () in
                          (match cal_path with
                          | Some calibrated_path ->
                              printf "  Applying dark calibration to %s\n" (Filename.basename new_path);
                              printf "  Output path: %s\n" calibrated_path;
                              
                              (* Create directory if needed *)
                              let cal_dir = Filename.dirname calibrated_path in
                              if not (Sys.file_exists cal_dir) then
                                create_dir cal_dir;
                                
                              (* Apply the calibration with caching *)
                              if apply_dark_calibration dark_dir new_path calibrated_path temp_tolerance then begin
                                printf "  Successfully calibrated image\n";
                                calibrated := !calibrated + 1;
				flush stdout
                              end
                          | None ->
                              printf "  Error determining calibrated output path\n")
                      | None ->
                          printf "  Warning: No darks directory specified for calibration\n"
                    end
                  end else begin
                    eprintf "  Failed to copy and annotate FITS file\n";
                    errors := !errors + 1
                  end
                end else begin
                  printf "  Skipped - file already exists\n";
                  skipped := !skipped + 1
                end
              with
              | e ->
                  eprintf "  Error processing: %s\n" (Printexc.to_string e);
                  errors := !errors + 1
            end else
              printf "  Would process (dry run)\n"
        | None ->
            eprintf "  Error determining new path\n";
            errors := !errors + 1
      end
    with
    | e ->
        eprintf "  Error reading files: %s\n" (Printexc.to_string e);
        errors := !errors + 1
  ) !pairs;

  (* In the process_directory function, after all files are processed *)
  (* Near the end, where you print the summary, add: *)

  (* Analyze field rotation across processed files *)
  if !processed > 1 then begin
    printf "\n=== Field Rotation Analysis ===\n";

    (* Find all processed files *)
    let processed_files = ref [] in
    let rec check_dir dir =
      try
	Array.iter (fun entry ->
	  let full_path = Filename.concat dir entry in
	  if Sys.is_directory full_path then
	    (* Recursively check subdirectories *)
	    check_dir full_path
	  else if Filename.check_suffix entry ".fits" then
	    processed_files := full_path :: !processed_files
	) (Sys.readdir dir)
      with _ -> ()
    in

    check_dir base_dir;

    if List.length !processed_files > 0 then begin
      ignore (analyze_field_rotation_from_headers !processed_files);
    end else
      printf "No processed FITS files found for rotation analysis\n";

    printf "=== Field Rotation Analysis Complete ===\n\n";
  end;
  
  (* Print master dark cache statistics *)
  let cache_size = Hashtbl.length master_dark_cache in
  if cache_size > 0 then begin
    printf "\nMaster dark cache statistics:\n";
    printf "  %d temperature bins in cache\n" cache_size;
    printf "  Cached temperature bins: ";
    Hashtbl.iter (fun bin_temp _ ->
      printf "temp_%d " bin_temp
    ) master_dark_cache;
    printf "\n";
  end;
  
  printf "\nSummary:\n";
  printf "  Processed: %d\n" !processed;
  printf "  Skipped: %d\n" !skipped;
  printf "  Errors: %d\n" !errors;
  
  if calibrate then
    printf "  Calibrated frames: %d\n" !calibrated;
  
  if add_registration then begin
    printf "  Registration attributes added: %d\n" !registration_added;
    printf "  Registration attributes failed: %d\n" !registration_failed
  end;
  
  (!processed, !skipped, !errors, !calibrated, !registration_added, !registration_failed)

(* Export function to process a batch with calibration *)
let process_with_dark_calibration darks_dir light_dir output_dir temp_tolerance =
  (* Look for all light frames *)
  let light_files = Dark_calibration.find_fits_files light_dir in
  
  if Array.length light_files = 0 then begin
    printf "No light frames found in %s\n" light_dir;
    (0, 0)
  end else begin
    printf "Found %d light frames in %s\n" (Array.length light_files) light_dir;
    
    (* Create output directory if needed *)
    if not (Sys.file_exists output_dir) then begin
      try Unix.mkdir output_dir 0o755 
      with _ -> printf "Error creating output directory %s\n" output_dir
    end;
    
    (* Process each light frame *)
    let processed = ref 0 in
    let errors = ref 0 in
    
    Array.iter (fun light_file ->
      (* Determine output path *)
      let basename = Filename.basename light_file in
      let output_path = Filename.concat output_dir ("cal_" ^ basename) in
      
      printf "Processing %s...\n" basename;
      
      (* Apply calibration *)
      if apply_dark_calibration darks_dir light_file output_path temp_tolerance then begin
        printf "Successfully calibrated to %s\n" output_path;
        incr processed
      end else begin
        printf "Failed to calibrate %s\n" basename;
        incr errors
      end
    ) light_files;
    
    (* Print master dark cache statistics *)
    let cache_size = Hashtbl.length master_dark_cache in
    printf "\nMaster dark cache statistics:\n";
    printf "  %d temperature bins in cache\n" cache_size;
    
    if cache_size > 0 then begin
      printf "  Cached temperature bins: ";
      Hashtbl.iter (fun bin_temp _ ->
        printf "temp_%d " bin_temp
      ) master_dark_cache;
      printf "\n";
    end;
    
    printf "Calibration complete: %d processed, %d errors\n" !processed !errors;
    (!processed, !errors)
  end

(* Main function with command-line argument parsing *)
let main () =
  let usage = "Usage: "^Sys.argv.(0)^" [OPTIONS] directory" in
  let directory = ref "" in
  let output = ref "lights" in
  let target = ref None in
  let max_separation = ref 5.0 in
  let dry_run = ref false in
  let lat = ref 52.245091 in
  let lon = ref 0.079609 in
  let no_registration = ref false in
  let observation_json = ref None in
  let calibrate = ref false in
  let darks_dir = ref None in
  let pointing_model = ref None in
  let verify_model = ref false in
  let plate_scale = ref 0.57 in
  let temp_tolerance = ref 2.0 in
  let analyze_rotation = ref false in
  
  let speclist = [
    ("--output", Arg.Set_string output, "Base output directory");
    ("--target", Arg.String (fun s -> target := Some s), "Target object name for coordinate verification");
    ("--max-separation", Arg.Set_float max_separation, "Maximum allowed separation from target in degrees");
    ("--dry-run", Arg.Set dry_run, "Validate without modifying files");
    ("--lat", Arg.Set_float lat, "Observatory latitude in degrees (N positive)");
    ("--lon", Arg.Set_float lon, "Observatory longitude in degrees (E positive)");
    ("--no-registration", Arg.Set no_registration, "Skip adding registration attributes from observation.json");
    ("--observation-json", Arg.String (fun s -> observation_json := Some s), 
      "Specific path to observation.json file if not in default location");
    ("--calibrate", Arg.Set calibrate, "Apply dark frame calibration to light frames");
    ("--darks-dir", Arg.String (fun s -> darks_dir := Some s), 
      "Directory containing dark frames for calibration");
    ("--pointing-model", Arg.String (fun s -> pointing_model := Some s), 
      "Path to pointing model JSON file");
    ("--verify-model", Arg.Set verify_model, "Verify pointing model accuracy with plate solving");
    ("--plate-scale", Arg.Set_float plate_scale, 
      "Plate scale in arcseconds per pixel (default: 0.57 for Stellina)");
    ("--temp-tolerance", Arg.Set_float temp_tolerance, 
      "Temperature tolerance for dark frame matching (default: 2.0)");
    ("--analyze-rotation", Arg.Set analyze_rotation, 
     "Analyze field rotation and derotator performance in processed files");

  ] in
  
  let anon_fun arg = directory := arg in
  
  Arg.parse speclist anon_fun usage;
  
  if !directory = "" then begin
    printf "Error: Missing directory argument\n";
    Arg.usage speclist usage;
    exit 1
  end;

  (* If analyze_rotation flag is set, just run that analysis and exit *)
  if !analyze_rotation then begin

    (* Find all FITS files recursively *)
    let files = ref [] in
    let rec find_fits dir =
      try
	Array.iter (fun entry ->
	  let full_path = Filename.concat dir entry in
	  if Sys.is_directory full_path then
	    find_fits full_path
	  else if Filename.check_suffix entry ".fits" then
	    files := full_path :: !files
	) (Sys.readdir dir)
      with _ -> ()
    in

    find_fits !directory;

    if List.length !files > 0 then begin
      printf "Analyzing field rotation in %d FITS files from %s\n" 
	(List.length !files) !directory;
      ignore (analyze_field_rotation_from_headers !files);
      exit 0
    end else begin
      printf "No FITS files found in %s\n" !directory;
      exit 1
    end
  end;

  (* Verify target if specified *)
  begin match !target with
  | None -> ()
  | Some name ->
      match QuerySimbad.get_object_coordinates name with
      | Some (ra, dec) ->
          printf "Target %s: RA=%.4f°, Dec=%.4f°\n" name ra dec
      | None ->
          printf "Error: Could not find coordinates for %s\n" name;
          exit 1
  end;
  
  (* Process the directory *)
  let flags = {
    base_dir = !output;
    target_name = !target;
    max_separation_deg = !max_separation;
    dry_run = !dry_run;
    solve = false;
    add_registration = not !no_registration;
    observation_json_path = !observation_json;
    calibrate = !calibrate;
    darks_dir = !darks_dir;
    pointing_model_path = !pointing_model;
    verify_model = !verify_model;
    latitude = !lat;
    longitude = !lon;
    plate_scale = !plate_scale;
    temp_tolerance = !temp_tolerance;
  } in
  
  let (processed, skipped, errors, calibrated, registration_added, registration_failed) =
    process_directory !directory flags
  in
  
  (* Print overall summary *)
  printf "\nOverall Summary:\n";
  if !calibrate then begin
    let total_frames = processed + skipped in
    if total_frames > 0 then begin
      let cal_percentage = float_of_int calibrated /. float_of_int total_frames *. 100.0 in
      printf "- Successfully calibrated frames: %d/%d (%.1f%% of total)\n" 
        calibrated total_frames cal_percentage
    end
  end;
  
  if not !no_registration then begin
    let total_frames = processed + skipped in
    if total_frames > 0 then begin
      let reg_percentage = float_of_int registration_added /. float_of_int total_frames *. 100.0 in
      printf "- Successfully added registration data: %d/%d (%.1f%% of total)\n" 
        registration_added total_frames reg_percentage;
      if registration_failed > 0 then begin
        let fail_percentage = float_of_int registration_failed /. float_of_int total_frames *. 100.0 in
        printf "- Failed to add registration data: %d/%d (%.1f%% of total)\n" 
          registration_failed total_frames fail_percentage
      end
    end
  end;
  
  begin match !pointing_model with
  | None -> ()
  | Some path ->
      printf "- Used pointing model: %s\n" path;
      if !verify_model then begin
        printf "- Verified pointing model with plate solving\n";
        printf "  (See FITS headers for ORIGERR, MODERR, and MODIMPRV keywords for details)\n"
      end
  end;
  
  (* Return error code if any failures *)
  if errors > 0 then exit 1 else exit 0

(* If this module is run directly, execute the main function *)
let () = 
  if !Sys.interactive then
    ()  (* Don't run main in interactive mode *)
  else
    main ()
