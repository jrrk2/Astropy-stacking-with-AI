open Types
open Fits
open Printf
open Plplot
(* Add these functions to util.ml *)

(* Function to save model to JSON file *)
let save_model_to_file model filename =
  let json = model_to_yojson model in
  let oc = open_out filename in
  Printf.fprintf oc "%s\n" (Yojson.Safe.pretty_to_string (json :> Yojson.Safe.t));
  close_out oc;
  printf "Model saved to JSON file: %s\n" filename

(* Function to load model from JSON file *)
let load_model_from_file filename =
  try
    let ic = open_in filename in
    let json = Yojson.Basic.from_channel ic in
    close_in ic;
    match model_of_yojson (json :> Yojson.Safe.t) with
    | Ok model -> 
        printf "Loaded model from JSON file %s with %d reference points\n" 
          filename (List.length model.reference_points);
        Some model
    | Error msg ->
        printf "Error parsing model JSON: %s\n" msg;
        None
  with
  | Sys_error msg -> 
      printf "Error opening model file: %s\n" msg;
      None
  | e -> 
      printf "Error loading model file: %s\n" (Printexc.to_string e);
      None

(* Extract temperature from FITS header *)
let get_temperature hdrh =
  try 
    parse_float hdrh "CCD-TEMP"
  with _ -> 
    try 
      parse_float hdrh "CCDTEMP"
    with _ ->
      try 
        parse_float hdrh "TEMP"
      with _ ->
        try 
          (parse_float hdrh "TEMP_K") -. 273.15
            with _ ->
              failwith "Could not find temperature in FITS header"

let get_timestamp hdrh =
  try
    let date_str = try String.trim (Hashtbl.find hdrh "DATE-OBS=")
      with _ -> String.trim (List.hd (List.tl (String.split_on_char '=' (Hashtbl.find hdrh "DATE")))) in
    (try Scanf.sscanf date_str "'%d-%d-%dT%d:%d:%d" 
      (fun yr mon day hr min sec ->
        let tim = fst (Unix.mktime {
          tm_sec=sec; tm_min=min; tm_hour=hr; tm_mday=day; 
          tm_mon=mon-1; tm_year=yr-1900; tm_wday=0; 
          tm_yday=0; tm_isdst=false }) in tim) 
    with _ -> 
      printf "Invalid DATE-OBS: %s\n" date_str; 
      0.0)
  with _ -> 
    failwith "Could not find DATE-OBS"

(* Full frame analysis *)
let analyze_frame filename =
  let hdrh = just_header filename in
  let mountra = parse_float hdrh "MOUNTRA" in
  let mountdec = parse_float hdrh "MOUNTDEC=" in
  let solvedra = parse_float hdrh "CRVAL1" in
  let solveddec = parse_float hdrh "CRVAL2" in
  let temp = get_temperature hdrh in
  let timestamp = get_timestamp hdrh in
  let focus = 0 in

  { filename; temperature = temp; mountra; 
    mountdec; solvedra; solveddec; focus; timestamp; hdrh }

let filter_by_temp_range stats range =
  match range with
  | None -> stats
  | Some (min_temp, max_temp) ->
      Array.of_list (
        Array.to_list stats |> 
        List.filter (fun s -> 
          s.temperature >= min_temp && s.temperature <= max_temp))

let plot_temp_distribution stats =
  let temps = Array.map (fun s -> s.temperature) stats in
  Array.sort compare temps;
  
  let nbins = 50 in
  let min_temp = temps.(0) in
  let max_temp = temps.(Array.length temps - 1) in
  let bin_width = (max_temp -. min_temp) /. float_of_int nbins in
  let bins = Array.make nbins 0 in
  Array.iter (fun t ->
    let bin = int_of_float ((t -. min_temp) /. bin_width) in
    if bin >= 0 && bin < nbins then
      bins.(bin) <- bins.(bin) + 1
  ) temps;
  
  let x = Array.init nbins (fun i -> min_temp +. (float_of_int i +. 0.5) *. bin_width) in
  let y = Array.map float_of_int bins in
  
  plsdev "pngcairo";
  plsfnam "distribution.png";
  plinit ();
  plenv (min_temp) (max_temp) 0.0 (Array.fold_left max 0.0 y) 0 0;
  pllab "Temperature (°C)" "Count" "Temperature Distribution";
  plbin x y [PL_BIN_DEFAULT];
  plend ();
  printf "Generated distribution.png\n"
      
let analyze_frames files flags =
  printf "Analyzing %d dark frames...\n" (Array.length files);
  
  (* Analyze all frames *)
  let all_stats = Array.map analyze_frame files in
  Array.sort (fun (a:frame_stats) (b:frame_stats) -> compare a.timestamp b.timestamp) all_stats;
  
  (* Filter by temperature range if specified *)
  let stats = filter_by_temp_range all_stats flags.temp_range in
  
  (* Generate requested outputs *)
(*
  if flags.show_temp_plot then plot_temp_vs_time stats;
*)
  if flags.show_dist_plot then plot_temp_distribution stats;
  
  if flags.show_stats then begin
    printf "\nStatistical Analysis:\n";
    printf "===================\n";
    printf "Temperature(°C)  RA  DEC  Timestamp\n";
    let datum = ref (Unix.gettimeofday()) in  
    Array.iter (fun (s:frame_stats) -> if !datum > s.timestamp then datum := s.timestamp;
    ) stats;
    Array.iter (fun s ->
      printf "%8.1f        %8.4f         %8.4f         %8.4f         %8.4f         %8.1f\n" 
        s.temperature s.mountra s.mountdec s.solvedra s.solveddec (s.timestamp -. !datum)
    ) stats
  end

(* Function to add to your get_new_filepath function in stellina_process.ml *)
let extract_date_from_fits hdrh =
  try
    (* Print raw DATE-OBS for debugging *)
    Printf.printf "  DATE-OBS raw: %s\n" 
      (try Hashtbl.find hdrh "DATE-OBS=" with Not_found -> "Not found");
    
    (* Get timestamp using existing function *)
    let timestamp = get_timestamp hdrh in
    let tm = Unix.localtime timestamp in
    
    (* Format date components *)
    let year = tm.tm_year + 1900 in
    let month = tm.tm_mon + 1 in
    let day = tm.tm_mday in
    let hour = tm.tm_hour in
    let minute = tm.tm_min in
    let second = tm.tm_sec in
    
    Printf.printf "  Extracted date: %04d-%02d-%02d %02d:%02d:%02d\n" 
      year month day hour minute second;
    
    Some (year, month, day, hour, minute, second)
  with e -> 
    Printf.printf "  Date extraction failed: %s\n" (Printexc.to_string e);
    None
