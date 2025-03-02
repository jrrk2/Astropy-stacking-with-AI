open Types
open Fits
open Printf
open Plplot

(* Analysis flags *)
type analysis_flags = {
  mutable show_temp_plot: bool;
  mutable show_dist_plot: bool;
  mutable show_stats: bool;
  mutable show_coeff: bool;
  mutable temp_range: (float * float) option;
  mutable dry_run: bool;
  mutable base_dir: string;
}

(* Analysis results *)
type frame_stats = {
  filename: string;
  temperature: float;
  ra: float;
  dec: float;
  ra': float;
  dec': float;
  timestamp: float;
  hdrh: (string, string) Hashtbl.t
}

type temp_info = {
  filename: string;
  qhdr: (string, string) Hashtbl.t;
  temp: float;
}

(* Create directory if it doesn't exist *)
let ensure_directory dir =
  if not (Sys.file_exists dir) then
    Unix.mkdir dir 0o755
  else if not (Sys.is_directory dir) then
    failwith (sprintf "%s exists but is not a directory" dir)

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

(* Full frame analysis *)
let analyze_frame filename =
  let img = read_image filename in
  let hdrh, contents = find_header_end filename img in
  let ra = parse_float hdrh "MOUNTRA" in
  let dec = parse_float hdrh "MOUNTDEC=" in
  let ra' = parse_float hdrh "CRVAL1" in
  let dec' = parse_float hdrh "CRVAL2" in
  let temp = get_temperature hdrh in
  let timestamp = get_timestamp hdrh in
  
  { filename; temperature = temp; ra; 
    dec; ra'; dec'; timestamp; hdrh }

(* Temperature plots *)
let plot_temp_vs_time (stats:frame_stats array) =
  (* Get timestamps first *)
  let datum = ref (Unix.gettimeofday()) in
  let x' = Array.map (fun (s:frame_stats) -> 
    let stamp = get_timestamp s.hdrh in if !datum > stamp then datum := stamp; stamp) stats in
  let x = Array.map (fun itm -> 
    itm -. !datum) x' in
  let y = Array.map (fun s -> s.temperature) stats in
  
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
  pllab "Time (seconds from midnight)" "Temperature (°C)" "Temperature vs Time";
  plline x y;
  plend ();
  printf "Generated temperature.png\n"

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

(* Temperature coefficient calculation *)
let calculate_temp_coefficient stats =
  let n = float_of_int (Array.length stats) in
  let sum_x = Array.fold_left (fun acc s -> acc +. s.temperature) 0.0 stats in
  let sum_y = Array.fold_left (fun acc s -> acc +. s.ra) 0.0 stats in
  let sum_xy = Array.fold_left (fun acc s -> 
    acc +. (s.temperature *. s.ra)) 0.0 stats in
  let sum_xx = Array.fold_left (fun acc s -> 
    acc +. (s.temperature *. s.temperature)) 0.0 stats in
  
  let slope = ((n *. sum_xy) -. (sum_x *. sum_y)) /. 
              ((n *. sum_xx) -. (sum_x *. sum_x)) in
  let intercept = (sum_y -. (slope *. sum_x)) /. n in
  
  (slope, intercept)

(* Filter frames by temperature range *)
let filter_by_temp_range stats range =
  match range with
  | None -> stats
  | Some (min_temp, max_temp) ->
      Array.of_list (
        Array.to_list stats |> 
        List.filter (fun s -> 
          s.temperature >= min_temp && s.temperature <= max_temp))

(* Main analysis function *)
let analyze_frames files flags =
  printf "Analyzing %d dark frames...\n" (Array.length files);
  
  (* Analyze all frames *)
  let all_stats = Array.map analyze_frame files in
  Array.sort (fun a b -> compare a.timestamp b.timestamp) all_stats;
  
  (* Filter by temperature range if specified *)
  let stats = filter_by_temp_range all_stats flags.temp_range in
  
  (* Generate requested outputs *)
  if flags.show_temp_plot then plot_temp_vs_time stats;
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
        s.temperature s.ra s.dec s.ra' s.dec' (s.timestamp -. !datum)
    ) stats
  end;
  
  if flags.show_coeff then begin
    let (slope, intercept) = calculate_temp_coefficient stats in
    printf "\nTemperature Coefficient Analysis:\n";
    printf "================================\n";
    printf "Temperature coefficient: %.3f ADU/°C\n" slope;
    printf "Dark current at 0°C: %.1f ADU\n" intercept
  end

(* Parse command line arguments *)
let parse_args () =
  let flags = {
    show_temp_plot = false;
    show_dist_plot = false;
    show_stats = false;
    show_coeff = false;
    temp_range = None;
    dry_run = false;
    base_dir = "frame_temps"
  } in
  let files = ref [] in
  let specs = [
    ("-temp", Arg.Unit (fun () -> flags.show_temp_plot <- true), 
     "Show temperature vs time plot");
    ("-dist", Arg.Unit (fun () -> flags.show_dist_plot <- true),
     "Show temperature distribution");
    ("-stats", Arg.Unit (fun () -> flags.show_stats <- true),
     "Show RA/DEC analysis");
    ("-coeff", Arg.Unit (fun () -> flags.show_coeff <- true),
     "Show temperature coefficient");
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
  let usage = sprintf "Usage: %s [-temp] [-dist] [-stats] [-coeff] [-sort] [-dry-run] [-outdir dir] [-range min max] <dark1.fits> [dark2.fits ...]\n" Sys.argv.(0) in
  Arg.parse specs add_file usage;
  (flags, Array.of_list (List.rev !files))

(* Main entry point *)
let () =
  let flags, files = parse_args () in
  if Array.length files = 0 then
    failwith "No input files specified"
  else if flags.show_temp_plot && not (flags.show_dist_plot || flags.show_stats || flags.show_coeff) then
    (* Fast path: only temperature scan needed *)
    let temp_infos = Array.to_list files 
      |> List.filter_map scan_fits_temperature 
      |> Array.of_list in
    if Array.length temp_infos > 0 then
      let stats = Array.map (fun (ti:temp_info) -> 
        {filename = ti.filename; 
         temperature = ti.temp;
         ra = 0.0;  (* Not needed for temp plot *)
         dec = 0.0;     (* Not needed for temp plot *)
         ra' = 0.0;  (* Not needed for temp plot *)
         dec' = 0.0;     (* Not needed for temp plot *)
         timestamp = 0.0;
         hdrh = ti.qhdr})  (* Will be filled by get_timestamp *)
        temp_infos in
      plot_temp_vs_time stats
    else
      failwith "No valid temperature data found in files"
  else
    (* Full analysis needed *)
    analyze_frames files flags
