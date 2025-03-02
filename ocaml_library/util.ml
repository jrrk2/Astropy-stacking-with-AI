open Types
open Fits
open Printf
open Plplot
open Yojson.Basic.Util

(* Function to save model to file using Marshal *)
let save_model_to_file model filename =
  let oc = open_out_bin filename in
  Marshal.to_channel oc model [];
  close_out oc;
  printf "Model saved to %s\n" filename

(* Function to load model from file using Marshal *)
let load_model_from_file filename =
  try
    let ic = open_in_bin filename in
    let model = (Marshal.from_channel ic : model) in
    close_in ic;
    printf "Loaded model from %s with %d reference points\n" 
      filename (List.length model.reference_points);
    Some model
  with
  | Sys_error msg -> 
      printf "Error opening model file: %s\n" msg;
      None
  | e -> 
      printf "Error loading model file: %s\n" (Printexc.to_string e);
      None    

let calculate_temp_coefficients model =
  if List.length model.reference_points < 2 then
    model  (* Not enough data points *)
  else
    let temps = List.map (fun (p:reference_point) -> p.temperature) model.reference_points in
    let ra_errors = List.map (fun p -> p.mount_ra -. p.solved_ra) model.reference_points in
    let dec_errors = List.map (fun p -> p.mount_dec -. p.solved_dec) model.reference_points in
    
    (* Print all inputs *)
    if verbose then Printf.printf "DEBUG Temperature coefficients inputs:\n";
    let patht n s = let l = String.length s in if l < n then s else String.sub s (l-n) n in
    List.iter (fun (p:reference_point) -> 
      Printf.printf "%s Point: temp=%.4f, ra_err=%.4f, dec_err=%.4f\n" 
        (patht 60 p.src_file) p.temperature (p.mount_ra -. p.solved_ra) (p.mount_dec -. p.solved_dec)
    ) model.reference_points;
    
    (* Simple linear regression for RA vs temperature *)
    let n = float_of_int (List.length temps) in
    let sum_t = List.fold_left (+.) 0.0 temps in
    let sum_ra_err = List.fold_left (+.) 0.0 ra_errors in
    let sum_t_ra = List.fold_left2 (fun acc t ra -> acc +. t *. ra) 0.0 temps ra_errors in
    let sum_t2 = List.fold_left (fun acc t -> acc +. t *. t) 0.0 temps in
    
    (* Debug intermediate calculations *)
    if verbose then Printf.printf "DEBUG regression calculations: n=%.0f, sum_t=%.4f, sum_t2=%.4f\n" n sum_t sum_t2;
    if verbose then Printf.printf "DEBUG regression calculations: sum_ra_err=%.4f, sum_t_ra=%.4f\n" sum_ra_err sum_t_ra;
    
    let denominator = ((n *. sum_t2) -. (sum_t *. sum_t)) in
    
    (* Check for division by zero *)
    if abs_float denominator < 1e-10 then
      (Printf.printf "WARNING: Near-zero denominator in temperature coefficient calculation\n";
       { model with ra_temp_coeff = 0.0; dec_temp_coeff = 0.0 })
    else 
      let ra_coeff = ((n *. sum_t_ra) -. (sum_t *. sum_ra_err)) /. denominator in
      
      (* Simple linear regression for DEC vs temperature *)
      let sum_dec_err = List.fold_left (+.) 0.0 dec_errors in
      let sum_t_dec = List.fold_left2 (fun acc t dec -> acc +. t *. dec) 0.0 temps dec_errors in
      
      if verbose then Printf.printf "DEBUG regression calculations: sum_dec_err=%.4f, sum_t_dec=%.4f\n" sum_dec_err sum_t_dec;
      
      let dec_coeff = ((n *. sum_t_dec) -. (sum_t *. sum_dec_err)) /. denominator in
      
      if verbose then Printf.printf "DEBUG Calculated coefficients: ra=%.8f, dec=%.8f\n" ra_coeff dec_coeff;
      
      { model with ra_temp_coeff = ra_coeff; dec_temp_coeff = dec_coeff }

let calculate_temp_coefficient stats =
  let n = float_of_int (Array.length stats) in
  let sum_x = Array.fold_left (fun acc s -> acc +. s.temperature) 0.0 stats in
  let sum_y = Array.fold_left (fun acc s -> acc +. s.mountra) 0.0 stats in
  let sum_xy = Array.fold_left (fun acc s -> 
    acc +. (s.temperature *. s.mountra)) 0.0 stats in
  let sum_xx = Array.fold_left (fun acc s -> 
    acc +. (s.temperature *. s.temperature)) 0.0 stats in
  
  let slope = ((n *. sum_xy) -. (sum_x *. sum_y)) /. 
              ((n *. sum_xx) -. (sum_x *. sum_x)) in
  let intercept = (sum_y -. (slope *. sum_x)) /. n in
  
  (slope, intercept)

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
  let img = read_image filename in
  let hdrh, contents = find_header_end filename img in
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
  Array.sort (fun a b -> compare a.timestamp b.timestamp) all_stats;
  
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
    Array.iter (fun s -> if !datum > s.timestamp then datum := s.timestamp;
    ) stats;
    Array.iter (fun s ->
      printf "%8.1f        %8.4f         %8.4f         %8.4f         %8.4f         %8.1f\n" 
        s.temperature s.mountra s.mountdec s.solvedra s.solveddec (s.timestamp -. !datum)
    ) stats
  end;
  
  if flags.show_coeff then begin
    let (slope, intercept) = calculate_temp_coefficient stats in
    printf "\nTemperature Coefficient Analysis:\n";
    printf "================================\n";
    printf "Temperature coefficient: %.3f ADU/°C\n" slope;
    printf "Dark current at 0°C: %.1f ADU\n" intercept
  end
