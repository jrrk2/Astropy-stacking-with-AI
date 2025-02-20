(* Astronomical YCbCr noise analysis *)
open Types

(* Modified RGB to YCbCr conversion with HDR handling *)
let astro_rgb_to_ycbcr (r, g, b) =
    (* Convert to float and apply log scaling for HDR *)
    let log_scale x =
        let x' = float_of_int x /. 65535.0 in
        if x' <= 0.0 then 0.0
        else log1p(x') /. log1p(1.0) in
    
    let r' = log_scale r in
    let g' = log_scale g in
    let b' = log_scale b in
    
    (* YCbCr conversion with astronomical weightings *)
    let y  =  0.299 *. r' +. 0.587 *. g' +. 0.114 *. b' in
    let cb = -0.169 *. r' -. 0.331 *. g' +. 0.500 *. b' in
    let cr =  0.500 *. r' -. 0.419 *. g' -. 0.081 *. b' in
    
    (y, cb, cr)

(* Background estimation using sigma-clipping *)
let estimate_background channel_data =
    let height = Array.length channel_data in
    let width = Array.length channel_data.(0) in
    let values = ref [] in
    
    (* Collect all values *)
    for y = 0 to height - 1 do
        for x = 0 to width - 1 do
            values := channel_data.(y).(x) :: !values
        done
    done;
    
    (* Sort values *)
    let sorted = Array.of_list !values |> Array.sort compare in
    let n = Array.length sorted in
    
    (* Initial median and MAD estimation *)
    let median = sorted.(n/2) in
    let mad = Array.map (fun x -> abs_float(x -. median)) sorted |> 
             Array.sort compare |> 
             (fun arr -> arr.(n/2)) in
    
    (* Sigma clip at 3σ *)
    let sigma = 1.4826 *. mad in
    let clipped = ref [] in
    Array.iter (fun x ->
        if abs_float(x -. median) < 3.0 *. sigma then
            clipped := x :: !clipped
    ) sorted;
    
    (* Calculate stats on clipped data *)
    let n_clipped = List.length !clipped in
    let mean = List.fold_left (+.) 0.0 !clipped /. float_of_int n_clipped in
    let variance = List.fold_left (fun acc x ->
        let diff = x -. mean in
        acc +. (diff *. diff)
    ) 0.0 !clipped /. float_of_int (n_clipped - 1) in
    
    (mean, sqrt variance)

(* Star detection for noise masking *)
let detect_stars y_channel threshold =
    let height = Array.length y_channel in
    let width = Array.length y_channel.(0) in
    let mask = Array.make_matrix height width false in
    
    (* Calculate background stats *)
    let bg_mean, bg_stddev = estimate_background y_channel in
    let detection_threshold = bg_mean +. threshold *. bg_stddev in
    
    (* Mark star pixels *)
    for y = 0 to height - 1 do
        for x = 0 to width - 1 do
            if y_channel.(y).(x) > detection_threshold then
                mask.(y).(x) <- true
        done
    done;
    
    mask

(* Analyze noise excluding stars *)
let analyze_astronomical_noise rgb_data =
    let ycbcr = Array.map (fun row ->
        Array.map astro_rgb_to_ycbcr row
    ) rgb_data in
    
    (* Extract Y channel *)
    let y_channel = Array.map (fun row ->
        Array.map (fun (y,_,_) -> y) row
    ) ycbcr in
    
    (* Detect stars *)
    let star_mask = detect_stars y_channel 5.0 in
    
    (* Collect background pixels *)
    let bg_y = ref [] in
    let bg_cb = ref [] in
    let bg_cr = ref [] in
    
    Array.iteri (fun y row ->
        Array.iteri (fun x (y_val, cb_val, cr_val) ->
            if not star_mask.(y).(x) then begin
                bg_y := y_val :: !bg_y;
                bg_cb := cb_val :: !bg_cb;
                bg_cr := cr_val :: !bg_cr
            end
        ) row
    ) ycbcr;
    
    (* Calculate statistics *)
    let calc_stats values =
        let n = float_of_int (List.length values) in
        let mean = List.fold_left (+.) 0.0 values /. n in
        let variance = List.fold_left (fun acc v ->
            let diff = v -. mean in
            acc +. (diff *. diff)
        ) 0.0 values /. (n -. 1.0) in
        (mean, sqrt variance)
    in
    
    let y_stats = calc_stats !bg_y in
    let cb_stats = calc_stats !bg_cb in
    let cr_stats = calc_stats !bg_cr in
    
    {
        background_noise = {
            y_stats;
            cb_stats;
            cr_stats
        };
        star_fraction = 
            let total = float_of_int (Array.length rgb_data * Array.length rgb_data.(0)) in
            let stars = Array.fold_left (fun acc row ->
                acc + Array.fold_left (fun a b -> if b then a + 1 else a) 0 row
            ) 0 star_mask in
            float_of_int stars /. total
    }961d3b0a54f4671cf33b115cbcbd9846
echo x - dark_temp_analysis.ml
sed 's/^X//' >dark_temp_analysis.ml << 'e66fa7e2f165c62575f2a2289b223dd9'
(* dark_temp_analysis.ml *)
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
}

(* Analysis results *)
type dark_stats = {
  filename: string;
  temperature: float;
  mean_level: float;
  std_dev: float;
  timestamp: float;
  hdrh: (string, string) Hashtbl.t
}

type temp_info = {
  filename: string;
  qhdr: (string, string) Hashtbl.t;
  temp: float;
}

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
    let date_str = String.trim (Hashtbl.find hdrh "DATE-OBS=") in
    (try Scanf.sscanf date_str "'%d-%d-%dT%d:%d:%d'" 
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
let analyze_dark_frame filename =
  let img = read_image filename in
  let hdrh, contents = find_header_end filename img in
  let width = parse_int hdrh "NAXIS1" in
  let height = parse_int hdrh "NAXIS2" in
  let temp = get_temperature hdrh in
  let timestamp = get_timestamp hdrh in
  let data = read_fits_data contents width height in
  
  let sum = ref 0.0 in
  let sum_sq = ref 0.0 in
  let count = width * height in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let value = float_of_int data.(y).(x) in
      sum := !sum +. value;
      sum_sq := !sum_sq +. (value *. value)
    done
  done;
  
  let mean = !sum /. float_of_int count in
  let variance = (!sum_sq /. float_of_int count) -. (mean *. mean) in
  let std_dev = sqrt variance in
  
  { filename; temperature = temp; mean_level = mean; 
    std_dev; timestamp; hdrh }

(* Temperature plots *)
let plot_temp_vs_time (stats:dark_stats array) =
  (* Get timestamps first *)
  let datum = ref (Unix.gettimeofday()) in
  let x' = Array.map (fun (s:dark_stats) -> 
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
  let sum_y = Array.fold_left (fun acc s -> acc +. s.mean_level) 0.0 stats in
  let sum_xy = Array.fold_left (fun acc s -> 
    acc +. (s.temperature *. s.mean_level)) 0.0 stats in
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
let analyze_dark_frames files flags =
  printf "Analyzing %d dark frames...\n" (Array.length files);
  
  (* Analyze all frames *)
  let all_stats = Array.map analyze_dark_frame files in
  Array.sort (fun a b -> compare a.timestamp b.timestamp) all_stats;
  
  (* Filter by temperature range if specified *)
  let stats = filter_by_temp_range all_stats flags.temp_range in
  
  (* Generate requested outputs *)
  if flags.show_temp_plot then plot_temp_vs_time stats;
  if flags.show_dist_plot then plot_temp_distribution stats;
  
  if flags.show_stats then begin
    printf "\nStatistical Analysis:\n";
    printf "===================\n";
    printf "Temperature(°C)  Mean Level(ADU)  Std Dev(ADU)  Timestamp\n";
    let datum = ref (Unix.gettimeofday()) in  
    Array.iter (fun s -> if !datum > s.timestamp then datum := s.timestamp;
    ) stats;
    Array.iter (fun s ->
      printf "%8.1f        %8.1f         %8.1f         %8.1f\n" 
        s.temperature s.mean_level s.std_dev (s.timestamp -. !datum)
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
    temp_range = None
  } in
  let files = ref [] in
  let specs = [
    ("-temp", Arg.Unit (fun () -> flags.show_temp_plot <- true), 
     "Show temperature vs time plot");
    ("-dist", Arg.Unit (fun () -> flags.show_dist_plot <- true),
     "Show temperature distribution");
    ("-stats", Arg.Unit (fun () -> flags.show_stats <- true),
     "Show mean/stddev analysis");
    ("-coeff", Arg.Unit (fun () -> flags.show_coeff <- true),
	       "Show temperature coefficient");
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
  let usage = sprintf "Usage: %s [-temp] [-dist] [-stats] [-coeff] [-range min max] <dark1.fits> [dark2.fits ...]\n" Sys.argv.(0) in
  Arg.parse specs add_file usage;
  (flags, Array.of_list (List.rev !files))

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
         mean_level = 0.0;  (* Not needed for temp plot *)
         std_dev = 0.0;     (* Not needed for temp plot *)
         timestamp = 0.0;
         hdrh = ti.qhdr})  (* Will be filled by get_timestamp *)
        temp_infos in
      plot_temp_vs_time stats
    else
      failwith "No valid temperature data found in files"
  else
    (* Full analysis needed *)
    analyze_dark_frames files flags
e66fa7e2f165c62575f2a2289b223dd9
echo x - dark_temp_analysis_anal.ml
sed 's/^X//' >dark_temp_analysis_anal.ml << '488219f4f039692cf309d0df1fc3896a'
(* dark_temp_analysis.ml *)
open Types
open Fits
open Printf

(* Types of analysis *)
type analysis_mode = 
  | Temperature  (* Just show temperature vs time *)
  | Full        (* Full analysis including statistics and coefficients *)
  | Distribution (* Just show temperature distribution *)

(* Structure for temperature data *)
type dark_stats = {
  filename: string;
  temperature: float;
  timestamp: float;
}

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
        failwith "Could not find temperature in FITS header"

let datum = fst (Unix.mktime {
      tm_sec=0;
      tm_min=0;
      tm_hour=0;
      tm_mday=14;
      tm_mon=1;
      tm_year=2025-1900;
      tm_wday=0;
      tm_yday=0;
      tm_isdst=false;
      })

let get_timestamp hdrh =
  try
    let date_str = String.trim (Hashtbl.find hdrh "DATE-OBS=") in
    (try Scanf.sscanf date_str "'%d-%d-%dT%d:%d:%d'" 
      (fun yr mon day hr min sec -> 
        fst (Unix.mktime {
          tm_sec=sec;
          tm_min=min;
          tm_hour=hr;
          tm_mday=day;
          tm_mon=mon-1;
          tm_year=yr-1900;
          tm_wday=0;
          tm_yday=0;
          tm_isdst=false;
        })) 
    with _ -> 
      printf "Invalid DATE-OBS: %s\n" date_str; 
      0.0)
  with _ -> 
    failwith "Could not find DATE-OBS"

(* Quick scan of just the FITS header *)
let analyze_frame filename =
  let fd = open_in_bin filename in
  let end_re = Str.regexp "END *$" in
  let rec read_header acc =
    let block = Bytes.create 2880 in
    really_input fd block 0 2880;
    let block_str = Bytes.to_string block in
    let acc = acc ^ block_str in
    try
      (* Search for END at 80-byte boundaries *)
      let rec find_end pos =
        if pos >= String.length acc then None
        else if pos mod 80 = 0 then
          let record = String.sub acc pos 80 in
          if Str.string_match end_re record 0 then Some acc
          else find_end (pos + 80)
        else find_end (pos + 80)
      in
      match find_end 0 with
      | Some header -> header
      | None -> read_header acc
    with _ -> read_header acc
  in
  try
    let header = read_header "" in
    let hdrh = Hashtbl.create 257 in
    let rec scan_header pos =
      if pos > String.length header - 80 then hdrh
      else 
        let key = String.sub header pos 80 in
        match String.trim (List.hd (String.split_on_char ' ' key)) with
        | "END" -> hdrh
        | "COMMENT" -> scan_header (pos + 80)
        | oth -> Hashtbl.add hdrh oth (String.sub key (String.length oth) 
                                    (80 - String.length oth));
                scan_header (pos + 80)
    in
    let hdrh = scan_header 0 in
    close_in fd;
    Some { 
      filename = filename;
      temperature = get_temperature hdrh;
      timestamp = get_timestamp hdrh 
    }
  with _ -> 
    close_in fd;
    printf "Warning: Could not read header from %s\n" filename;
    None

(* Show temperature distribution *)
let show_distribution stats =
  let temps = Array.map (fun s -> s.temperature) stats in
  Array.sort compare temps;
  let min_temp = temps.(0) in
  let max_temp = temps.(Array.length temps - 1) in
  let bins = Array.make (int_of_float (ceil (max_temp -. min_temp)) + 1) 0 in
  Array.iter (fun t -> 
    let idx = int_of_float (t -. min_temp) in
    bins.(idx) <- bins.(idx) + 1
  ) temps;
  
  printf "\nTemperature Distribution:\n";
  printf "========================\n";
  Array.iteri (fun i count ->
    if count > 0 then
      printf "%.1f°C: %d frames %s\n" 
        (min_temp +. float_of_int i) 
        count 
        (String.make (count / 10) '*')
  ) bins

(* Show temperature vs time *)
let show_temperature_time stats =
  printf "\nTime(s)     Temperature(°C)\n";
  Array.iter (fun s ->
    printf "%8.1f    %8.1f\n" 
      (s.timestamp -. datum) s.temperature
  ) stats

(* Main analysis function *)
let analyze_dark_frames dark_files mode =
  printf "Analyzing %d dark frames...\n" (Array.length dark_files);
  
  (* Analyze all frames *)
  let stats = Array.map analyze_frame dark_files |> Array.to_list |> 
              List.filter_map (fun x -> x) |> Array.of_list in
  
  (* Sort by timestamp *)
  Array.sort (fun a b -> compare a.timestamp b.timestamp) stats;
  
  match mode with
  | Temperature -> show_temperature_time stats
  | Distribution -> show_distribution stats
  | Full -> begin
      show_distribution stats;
      show_temperature_time stats
    end

(* Parse command line arguments *)
let parse_args () =
  let mode = ref Temperature in
  let files = ref [] in
  let specs = [
    ("-temp", Arg.Unit (fun () -> mode := Temperature), "Show temperature vs time");
    ("-dist", Arg.Unit (fun () -> mode := Distribution), "Show temperature distribution");
    ("-full", Arg.Unit (fun () -> mode := Full), "Show full analysis");
  ] in
  let add_file f = files := f :: !files in
  let usage = sprintf "Usage: %s [-temp|-dist|-full] <dark1.fits> [dark2.fits ...]\n" Sys.argv.(0) in
  Arg.parse specs add_file usage;
  (!mode, Array.of_list (List.rev !files))

(* Entry point *)
let () =
  let mode, dark_files = parse_args () in
  if Array.length dark_files = 0 then
    failwith "No input files specified"
  else
    analyze_dark_frames dark_files mode488219f4f039692cf309d0df1fc3896a
echo x - dark_temp_analysis_edited.ml
sed 's/^X//' >dark_temp_analysis_edited.ml << 'c7d7c2f212a042929b270e04f1686dd5'
(* dark_temp_analysis.ml *)
open Types
open Fits
open Printf

(* Structure to hold dark frame analysis results *)
type dark_stats = {
  filename: string;
  temperature: float;
  mean_level: float;
  std_dev: float;
  timestamp: float;
}

(* Structure for initial temperature scan *)
type temp_info = {
  filename: string;
  temp: float;
}

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
        failwith "Could not find temperature in FITS header"

let datum = fst (Unix.mktime {
      tm_sec=0;
      tm_min=0;
      tm_hour=0;
      tm_mday=14;
      tm_mon=1;
      tm_year=2025-1900;
      tm_wday=0;
      tm_yday=0;
      tm_isdst=false;
      })

(* Quick scan of just the FITS header for temperature *)
let scan_fits_temperature filename =
  let img = read_image filename in
  let hdrh, _ = find_header_end img in
  try
    Some { filename; temp = get_temperature hdrh }
  with _ -> 
    printf "Warning: Could not read temperature from %s\n" filename;
    None

(* Find relevant temperature range *)
let find_temp_range temps =
  let valid_temps = Array.to_list temps |> List.filter_map (fun x -> x) in
  match valid_temps with
  | [] -> failwith "No valid temperature readings found"
  | temps ->
      let sorted = List.sort (fun a b -> compare a.temperature b.temperature) temps in
      let min_temp = (List.hd sorted).temperature in
      let max_temp = (List.hd (List.rev sorted)).temperature in
      let temp_count = Array.make (int_of_float (ceil (max_temp -. min_temp)) + 1) 0 in
      List.iter (fun t -> 
        let idx = int_of_float (t.temperature -. min_temp) in
        temp_count.(idx) <- temp_count.(idx) + 1
      ) temps;
      
      printf "\nTemperature Distribution:\n";
      printf "========================\n";
      Array.iteri (fun i count ->
        if count > 0 then
          printf "%.1f°C: %d frames %s\n" 
            (min_temp +. float_of_int i) 
            count 
            (String.make (count / 10) '*')
      ) temp_count;
      (min_temp, max_temp)

let get_timestamp hdrh =
  try
    let date_str = String.trim (Hashtbl.find hdrh "DATE-OBS=") in
    (try Scanf.sscanf date_str "'%d-%d-%dT%d:%d:%d'" 
      (fun yr mon day hr min sec -> 
        fst (Unix.mktime {
          tm_sec=sec;
          tm_min=min;
          tm_hour=hr;
          tm_mday=day;
          tm_mon=mon-1;
          tm_year=yr-1900;
          tm_wday=0;
          tm_yday=0;
          tm_isdst=false;
        })) 
    with _ -> 
      printf "Invalid DATE-OBS: %s\n" date_str; 
      0.0)
  with _ -> 
    failwith "Could not find DATE-OBS"

let analyze_dark_frame filename =
  let img = read_image filename in
  let hdrh, contents = find_header_end img in
  let width = parse_int hdrh "NAXIS1" in
  let height = parse_int hdrh "NAXIS2" in
  let temp = get_temperature hdrh in
  let timestamp = get_timestamp hdrh in
  let data = read_fits_data contents width height in
  
  let sum = ref 0.0 in
  let sum_sq = ref 0.0 in
  let count = width * height in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let value = float_of_int data.(y).(x) in
      sum := !sum +. value;
      sum_sq := !sum_sq +. (value *. value)
    done
  done;
  
  let mean = !sum /. float_of_int count in
  let variance = (!sum_sq /. float_of_int count) -. (mean *. mean) in
  let std_dev = sqrt variance in
  
  { filename; temperature = temp; mean_level = mean; 
    std_dev; timestamp }

let calculate_temp_coefficient dark_stats temp_infos =
  let n = float_of_int (Array.length dark_stats) in
  (* Create lookup for quick temperature access *)
  let temp_map = Hashtbl.create (Array.length temp_infos) in
  Array.iter (fun temp_info -> 
    Hashtbl.add temp_map temp_info.filename temp_info.temp
  ) temp_infos;
  
  (* Calculate sums for regression *)
  let sum_x = ref 0.0 in
  let sum_y = ref 0.0 in
  let sum_xy = ref 0.0 in
  let sum_xx = ref 0.0 in
  
  Array.iter (fun (stats:dark_stats) ->
    match Hashtbl.find_opt temp_map stats.filename with
    | Some temp ->
        sum_x := !sum_x +. temp;
        sum_y := !sum_y +. stats.mean_level;
        sum_xy := !sum_xy +. (temp *. stats.mean_level);
        sum_xx := !sum_xx +. (temp *. temp)
    | None -> printf "Warning: No temperature data for %s\n" stats.filename
  ) dark_stats;
  
  let slope = ((n *. !sum_xy) -. (!sum_x *. !sum_y)) /. 
              ((n *. !sum_xx) -. (!sum_x *. !sum_x)) in
  let intercept = (!sum_y -. (slope *. !sum_x)) /. n in
  
  (slope, intercept)

(* Main analysis function with temperature range filtering *)
let analyze_dark_frames dark_files target_temp_range =
  printf "Quick scanning %d dark frames for temperature...\n" (Array.length dark_files);
  
  (* Quick temperature scan *)
  let temp_infos = Array.map scan_fits_temperature dark_files in
  
  (* Show temperature distribution *)
  let valid_temps = Array.to_list temp_infos |> List.filter_map (fun x -> x) in
  match valid_temps with
  | [] -> failwith "No valid temperature readings found"
  | temps ->
      let sorted = List.sort (fun a b -> compare a.temp b.temp) temps in
      let min_temp = (List.hd sorted).temp in
      let max_temp = (List.hd (List.rev sorted)).temp in
      let temp_count = Array.make (int_of_float (ceil (max_temp -. min_temp)) + 1) 0 in
      List.iter (fun t -> 
        let idx = int_of_float (t.temp -. min_temp) in
        temp_count.(idx) <- temp_count.(idx) + 1
      ) temps;
      
      printf "\nTemperature Distribution:\n";
      printf "========================\n";
      Array.iteri (fun i count ->
        if count > 0 then
          printf "%.1f°C: %d frames %s\n" 
            (min_temp +. float_of_int i) 
            count 
            (String.make (count / 10) '*')
      ) temp_count;
      
      (* Filter frames within target range *)
      let (target_min, target_max) = target_temp_range in
      let range_margin = 2.0 in (* Look at darks within ±2°C of target range *)
      let filtered_files = List.filter_map (fun info ->
        if info.temp >= (target_min -. range_margin) && 
           info.temp <= (target_max +. range_margin)
        then Some info.filename
        else None
      ) temps |> Array.of_list in
      
      printf "\nAnalyzing %d dark frames within temperature range %.1f°C to %.1f°C...\n" 
        (Array.length filtered_files) (target_min -. range_margin) (target_max +. range_margin);
      
      (* Only analyze the filtered subset *)
      let stats = Array.map analyze_dark_frame filtered_files in
      Array.sort (fun a b -> compare a.temperature b.temperature) stats;
      
      let filtered_temp_infos = Array.of_list (List.filter (fun t -> 
        Array.exists (fun f -> f = t.filename) filtered_files) temps) in
      let (slope, intercept) = calculate_temp_coefficient stats filtered_temp_infos in
  
  (* Output results *)
  printf "\nTemperature Coefficient Analysis:\n";
  printf "================================\n";
  printf "Temperature coefficient: %.3f ADU/°C\n" slope;
  printf "Dark current at 0°C: %.1f ADU\n" intercept;
  printf "\nDetailed measurements:\n";
  printf "Temperature(°C)  Mean Level(ADU)  Std Dev(ADU)  Timestamp\n";
  Array.iter (fun s ->
    printf "%8.1f        %8.1f         %8.1f         %8.1f\n" 
      s.temperature s.mean_level s.std_dev (s.timestamp -. datum)
  ) stats;
  
  (slope, intercept, stats)

(* Entry point *)
let () =
  if Array.length Sys.argv < 2 then
    (printf "Usage: %s <dark1.fits> [dark2.fits ...]\n" Sys.argv.(0);
     exit 1);
  
  let dark_files = Array.sub Sys.argv 1 (Array.length Sys.argv - 1) in
  ignore (analyze_dark_frames dark_files (12.0, 20.0))
c7d7c2f212a042929b270e04f1686dd5
echo x - dark_temp_analysis_filtered.ml
sed 's/^X//' >dark_temp_analysis_filtered.ml << '88a28a39a27548039d0038b7653ffea3'
(* dark_temp_analysis.ml *)
open Types
open Fits
open Printf

(* Structure to hold dark frame analysis results *)
type dark_stats = {
  filename: string;
  temperature: float;
  mean_level: float;
  std_dev: float;
  timestamp: float;
}

(* Structure for initial temperature scan *)
type temp_info = {
  filename: string;
  temp: float;
}

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
        failwith "Could not find temperature in FITS header"

let datum = fst (Unix.mktime {
      tm_sec=0;
      tm_min=0;
      tm_hour=0;
      tm_mday=14;
      tm_mon=1;
      tm_year=2025-1900;
      tm_wday=0;
      tm_yday=0;
      tm_isdst=false;
      })

(* Quick scan of just the FITS header for temperature *)
let scan_fits_temperature filename =
  let fd = open_in_bin filename in
  let end_re = Str.regexp "END *$" in
  let rec read_header acc =
    let block = Bytes.create 2880 in
    really_input fd block 0 2880;
    let block_str = Bytes.to_string block in
    let acc = acc ^ block_str in
    try
      (* Search for END at 80-byte boundaries *)
      let rec find_end pos =
        if pos >= String.length acc then None
        else if pos mod 80 = 0 then
          let record = String.sub acc pos 80 in
          if Str.string_match end_re record 0 then Some acc
          else find_end (pos + 80)
        else find_end (pos + 80)
      in
      match find_end 0 with
      | Some header -> header
      | None -> read_header acc
    with _ -> read_header acc
  in
  try
    let header = read_header "" in
    let hdrh = Hashtbl.create 257 in
    let rec scan_header pos =
      if pos > String.length header - 80 then hdrh
      else 
        let key = String.sub header pos 80 in
        match String.trim (List.hd (String.split_on_char ' ' key)) with
        | "END" -> hdrh
        | "COMMENT" -> scan_header (pos + 80)
        | oth -> Hashtbl.add hdrh oth (String.sub key (String.length oth) 
                                    (80 - String.length oth));
                scan_header (pos + 80)
    in
    let hdrh = scan_header 0 in
    close_in fd;
    Some { filename; temp = get_temperature hdrh }
  with _ -> 
    close_in fd;
    printf "Warning: Could not read temperature from %s\n" filename;
    None

(* Find relevant temperature range *)
let find_temp_range temps =
  let valid_temps = Array.to_list temps |> List.filter_map (fun x -> x) in
  match valid_temps with
  | [] -> failwith "No valid temperature readings found"
  | temps ->
      let sorted = List.sort (fun a b -> compare a.temp b.temp) temps in
      let min_temp = (List.hd sorted).temp in
      let max_temp = (List.hd (List.rev sorted)).temp in
      let temp_count = Array.make (int_of_float (ceil (max_temp -. min_temp)) + 1) 0 in
      List.iter (fun t -> 
        let idx = int_of_float (t.temp -. min_temp) in
        temp_count.(idx) <- temp_count.(idx) + 1
      ) temps;
      
      printf "\nTemperature Distribution:\n";
      printf "========================\n";
      Array.iteri (fun i count ->
        if count > 0 then
          printf "%.1f°C: %d frames %s\n" 
            (min_temp +. float_of_int i) 
            count 
            (String.make (count / 10) '*')
      ) temp_count;
      (min_temp, max_temp)

let get_timestamp hdrh =
  try
    let date_str = String.trim (Hashtbl.find hdrh "DATE-OBS=") in
    (try Scanf.sscanf date_str "'%d-%d-%dT%d:%d:%d'" 
      (fun yr mon day hr min sec -> 
        fst (Unix.mktime {
          tm_sec=sec;
          tm_min=min;
          tm_hour=hr;
          tm_mday=day;
          tm_mon=mon-1;
          tm_year=yr-1900;
          tm_wday=0;
          tm_yday=0;
          tm_isdst=false;
        })) 
    with _ -> 
      printf "Invalid DATE-OBS: %s\n" date_str; 
      0.0)
  with _ -> 
    failwith "Could not find DATE-OBS"

let analyze_dark_frame filename =
  let img = read_image filename in
  let hdrh, contents = find_header_end img in
  let width = parse_int hdrh "NAXIS1" in
  let height = parse_int hdrh "NAXIS2" in
  let temp = get_temperature hdrh in
  let timestamp = get_timestamp hdrh in
  let data = read_fits_data contents width height in
  
  let sum = ref 0.0 in
  let sum_sq = ref 0.0 in
  let count = width * height in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let value = float_of_int data.(y).(x) in
      sum := !sum +. value;
      sum_sq := !sum_sq +. (value *. value)
    done
  done;
  
  let mean = !sum /. float_of_int count in
  let variance = (!sum_sq /. float_of_int count) -. (mean *. mean) in
  let std_dev = sqrt variance in
  
  { filename; temperature = temp; mean_level = mean; 
    std_dev; timestamp }

let calculate_temp_coefficient dark_stats temp_infos =
  let n = float_of_int (Array.length dark_stats) in
  (* Create lookup for quick temperature access *)
  let temp_map = Hashtbl.create (Array.length temp_infos) in
  Array.iter (fun temp_info -> 
    Hashtbl.add temp_map temp_info.filename temp_info.temp
  ) temp_infos;
  
  (* Calculate sums for regression *)
  let sum_x = ref 0.0 in
  let sum_y = ref 0.0 in
  let sum_xy = ref 0.0 in
  let sum_xx = ref 0.0 in
  
  Array.iter (fun (stats:dark_stats) ->
    match Hashtbl.find_opt temp_map stats.filename with
    | Some temp ->
        sum_x := !sum_x +. temp;
        sum_y := !sum_y +. stats.mean_level;
        sum_xy := !sum_xy +. (temp *. stats.mean_level);
        sum_xx := !sum_xx +. (temp *. temp)
    | None -> printf "Warning: No temperature data for %s\n" stats.filename
  ) dark_stats;
  
  let slope = ((n *. !sum_xy) -. (!sum_x *. !sum_y)) /. 
              ((n *. !sum_xx) -. (!sum_x *. !sum_x)) in
  let intercept = (!sum_y -. (slope *. !sum_x)) /. n in
  
  (slope, intercept)

(* Main analysis function with temperature range filtering *)
let analyze_dark_frames dark_files target_temp_range =
  printf "Quick scanning %d dark frames for temperature...\n" (Array.length dark_files);
  
  (* Quick temperature scan *)
  let temp_infos = Array.map scan_fits_temperature dark_files in
  
  (* Show temperature distribution *)
  let valid_temps = Array.to_list temp_infos |> List.filter_map (fun x -> x) in
  match valid_temps with
  | [] -> failwith "No valid temperature readings found"
  | temps ->
      let sorted = List.sort (fun a b -> compare a.temp b.temp) temps in
      let min_temp = (List.hd sorted).temp in
      let max_temp = (List.hd (List.rev sorted)).temp in
      let temp_count = Array.make (int_of_float (ceil (max_temp -. min_temp)) + 1) 0 in
      List.iter (fun t -> 
        let idx = int_of_float (t.temp -. min_temp) in
        temp_count.(idx) <- temp_count.(idx) + 1
      ) temps;
      
      printf "\nTemperature Distribution:\n";
      printf "========================\n";
      Array.iteri (fun i count ->
        if count > 0 then
          printf "%.1f°C: %d frames %s\n" 
            (min_temp +. float_of_int i) 
            count 
            (String.make (count / 10) '*')
      ) temp_count;
      
      (* Filter frames within target range *)
      let (target_min, target_max) = target_temp_range in
      let range_margin = 2.0 in (* Look at darks within ±2°C of target range *)
      let filtered_files = List.filter_map (fun info ->
        if info.temp >= (target_min -. range_margin) && 
           info.temp <= (target_max +. range_margin)
        then Some info.filename
        else None
      ) temps |> Array.of_list in
      
      printf "\nAnalyzing %d dark frames within temperature range %.1f°C to %.1f°C...\n" 
        (Array.length filtered_files) (target_min -. range_margin) (target_max +. range_margin);
      
      (* Only analyze the filtered subset *)
      let stats = Array.map analyze_dark_frame filtered_files in
      Array.sort (fun a b -> compare a.temperature b.temperature) stats;
      
      let filtered_temp_infos = Array.of_list (List.filter (fun t -> 
        Array.exists (fun f -> f = t.filename) filtered_files) temps) in
      let (slope, intercept) = calculate_temp_coefficient stats filtered_temp_infos in
      
      (* Output results *)
      printf "\nTemperature Coefficient Analysis:\n";
      printf "================================\n";
      printf "Temperature coefficient: %.3f ADU/°C\n" slope;
      printf "Dark current at 0°C: %.1f ADU\n" intercept;
      printf "\nDetailed measurements:\n";
      printf "Temperature(°C)  Mean Level(ADU)  Std Dev(ADU)  Timestamp\n";
      Array.iter (fun s ->
        printf "%8.1f        %8.1f         %8.1f         %8.1f\n" 
          s.temperature s.mean_level s.std_dev (s.timestamp -. datum)
      ) stats;
      
      (slope, intercept, stats)

(* Entry point *)
let () =
  if Array.length Sys.argv < 2 then
    (printf "Usage: %s <dark1.fits> [dark2.fits ...]\n" Sys.argv.(0);
     exit 1);
  
  let dark_files = Array.sub Sys.argv 1 (Array.length Sys.argv - 1) in
  ignore (analyze_dark_frames dark_files (12.0, 20.0))88a28a39a27548039d0038b7653ffea3
echo x - dark_temp_analysis_full.ml
sed 's/^X//' >dark_temp_analysis_full.ml << '95ac460b2080dddb3f3137ae3a9558b3'
(* dark_temp_analysis.ml *)
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
}

(* Analysis results *)
type dark_stats = {
  filename: string;
  temperature: float;
  mean_level: float;
  std_dev: float;
  timestamp: float;
}

type temp_info = {
  filename: string;
  temp: float;
}

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
        failwith "Could not find temperature in FITS header"

let datum = fst (Unix.mktime {
      tm_sec=0; tm_min=0; tm_hour=0; tm_mday=14; tm_mon=1;
      tm_year=2025-1900; tm_wday=0; tm_yday=0; tm_isdst=false })

(* Quick scan of just the FITS header *)
let scan_fits_temperature filename =
  let fd = open_in_bin filename in
  let end_re = Str.regexp "END *$" in
  let rec read_header acc =
    let block = Bytes.create 2880 in
    really_input fd block 0 2880;
    let block_str = Bytes.to_string block in
    let acc = acc ^ block_str in
    try
      let rec find_end pos =
        if pos >= String.length acc then None
        else if pos mod 80 = 0 then
          let record = String.sub acc pos 80 in
          if Str.string_match end_re record 0 then Some acc
          else find_end (pos + 80)
        else find_end (pos + 80)
      in
      match find_end 0 with
      | Some header -> header
      | None -> read_header acc
    with _ -> read_header acc
  in
  try
    let header = read_header "" in
    let hdrh = Hashtbl.create 257 in
    let rec scan_header pos =
      if pos > String.length header - 80 then hdrh
      else 
        let key = String.sub header pos 80 in
        match String.trim (List.hd (String.split_on_char ' ' key)) with
        | "END" -> hdrh
        | "COMMENT" -> scan_header (pos + 80)
        | oth -> Hashtbl.add hdrh oth (String.sub key (String.length oth) 
                                    (80 - String.length oth));
                scan_header (pos + 80)
    in
    let hdrh = scan_header 0 in
    close_in fd;
    Some { filename; temp = get_temperature hdrh }
  with _ -> 
    close_in fd;
    printf "Warning: Could not read temperature from %s\n" filename;
    None

let get_timestamp hdrh =
  try
    let date_str = String.trim (Hashtbl.find hdrh "DATE-OBS=") in
    (try Scanf.sscanf date_str "'%d-%d-%dT%d:%d:%d'" 
      (fun yr mon day hr min sec -> 
        fst (Unix.mktime {
          tm_sec=sec; tm_min=min; tm_hour=hr; tm_mday=day; 
          tm_mon=mon-1; tm_year=yr-1900; tm_wday=0; 
          tm_yday=0; tm_isdst=false })) 
    with _ -> 
      printf "Invalid DATE-OBS: %s\n" date_str; 
      0.0)
  with _ -> 
    failwith "Could not find DATE-OBS"

(* Full frame analysis *)
let analyze_dark_frame filename =
  let img = read_image filename in
  let hdrh, contents = find_header_end img in
  let width = parse_int hdrh "NAXIS1" in
  let height = parse_int hdrh "NAXIS2" in
  let temp = get_temperature hdrh in
  let timestamp = get_timestamp hdrh in
  let data = read_fits_data contents width height in
  
  let sum = ref 0.0 in
  let sum_sq = ref 0.0 in
  let count = width * height in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let value = float_of_int data.(y).(x) in
      sum := !sum +. value;
      sum_sq := !sum_sq +. (value *. value)
    done
  done;
  
  let mean = !sum /. float_of_int count in
  let variance = (!sum_sq /. float_of_int count) -. (mean *. mean) in
  let std_dev = sqrt variance in
  
  { filename; temperature = temp; mean_level = mean; 
    std_dev; timestamp }

(* Temperature plots *)
let plot_temp_vs_time stats =
  let x = Array.map (fun s -> s.timestamp -. datum) stats in
  let y = Array.map (fun s -> s.temperature) stats in
  
  plsdev "pngcairo";
  plsfnam "temperature.png";
  plinit ();
  plenv (Array.fold_left min x.(0) x) (Array.fold_left max x.(0) x)
        (Array.fold_left min y.(0) y) (Array.fold_left max y.(0) y)
        0 0;
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
  plbin x y true;
  plend ();
  printf "Generated distribution.png\n"

(* Temperature coefficient calculation *)
let calculate_temp_coefficient stats =
  let n = float_of_int (Array.length stats) in
  let sum_x = Array.fold_left (fun acc s -> acc +. s.temperature) 0.0 stats in
  let sum_y = Array.fold_left (fun acc s -> acc +. s.mean_level) 0.0 stats in
  let sum_xy = Array.fold_left (fun acc s -> 
    acc +. (s.temperature *. s.mean_level)) 0.0 stats in
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
let analyze_dark_frames files flags =
  printf "Analyzing %d dark frames...\n" (Array.length files);
  
  (* Analyze all frames *)
  let all_stats = Array.map analyze_dark_frame files in
  Array.sort (fun a b -> compare a.timestamp b.timestamp) all_stats;
  
  (* Filter by temperature range if specified *)
  let stats = filter_by_temp_range all_stats flags.temp_range in
  
  (* Generate requested outputs *)
  if flags.show_temp_plot then plot_temp_vs_time stats;
  if flags.show_dist_plot then plot_temp_distribution stats;
  
  if flags.show_stats then begin
    printf "\nStatistical Analysis:\n";
    printf "===================\n";
    printf "Temperature(°C)  Mean Level(ADU)  Std Dev(ADU)  Timestamp\n";
    Array.iter (fun s ->
      printf "%8.1f        %8.1f         %8.1f         %8.1f\n" 
        s.temperature s.mean_level s.std_dev (s.timestamp -. datum)
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
    temp_range = None
  } in
  let files = ref [] in
  let specs = [
    ("-temp", Arg.Unit (fun () -> flags.show_temp_plot <- true), 
     "Show temperature vs time plot");
    ("-dist", Arg.Unit (fun () -> flags.show_dist_plot <- true),
     "Show temperature distribution");
    ("-stats", Arg.Unit (fun () -> flags.show_stats <- true),
     "Show mean/stddev analysis");
    ("-coeff", Arg.Unit (fun () -> flags.show_coeff <- true),
     "Show temperature coefficient");
    ("-range", Arg.Tuple [| 
      (fun min_t -> flags.temp_range <- Some(min_t, 0.0));
      (fun max_t -> flags.temp_range <- 
        match flags.temp_range with 
        | Some(min_t,_) -> Some(min_t,max_t)
        | None -> Some(0.0,max_t))
    |], "Specify temperature range min,max")
  ] in
  let add_file f = files := f :: !files in
  let usage = sprintf "Usage: %s [-temp] [-dist] [-stats] [-coeff] [-range min max] <dark1.fits> [dark2.fits ...]\n" Sys.argv.(0) in
  Arg.parse specs add_file usage;
  (flags, Array.of_list (List.rev !files))

(* Entry point *)
let () =
  let flags, files = parse_args () in
  if Array.length files = 0 then
    failwith "No input files specified"
  else
    analyze_dark_frames files flags95ac460b2080dddb3f3137ae3a9558b3
echo x - dark_temp_analysis_old.ml
sed 's/^X//' >dark_temp_analysis_old.ml << 'd448501e519bb9ec2e4a4fe355366606'
(* dark_temp_analysis.ml *)
open Types
open Fits
open Printf

(* Structure to hold dark frame analysis results *)
type dark_stats = {
  filename: string;
  temperature: float;
  mean_level: float;
  std_dev: float;
  timestamp: float;
}

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
        failwith "Could not find temperature in FITS header"

let datum = fst (Unix.mktime {
      tm_sec=0;
      tm_min=0;
      tm_hour=0;
      tm_mday=14;
      tm_mon=1; (* Month of year 0..11 *)
      tm_year=2025-1900; (* Year - 1900 *)
      tm_wday=0;
      tm_yday=0;
      tm_isdst=false;
      })

(* Extract timestamp from FITS header *)
let get_timestamp hdrh =
  try
    let date_str = String.trim (Hashtbl.find hdrh "DATE-OBS=") in
    (try Scanf.sscanf date_str "'%d-%d-%dT%d:%d:%d'" (fun yr mon day hr min sec -> fst (Unix.mktime {
      tm_sec=sec;
      tm_min=min;
      tm_hour=hr;
      tm_mday=day;
      tm_mon=mon-1; (* Month of year 0..11 *)
      tm_year=yr-1900; (* Year - 1900 *)
      tm_wday=0;
      tm_yday=0;
      tm_isdst=false;
      })) with _ -> print_endline ("Invalid DATE-OBS: "^date_str); 0.0); with _ -> failwith "Could not find DATE-OBS"

(* Calculate statistics for a dark frame *)
let analyze_dark_frame filename =
  let img = read_image filename in
  let hdrh, contents = find_header_end img in
  let width = parse_int hdrh "NAXIS1" in
  let height = parse_int hdrh "NAXIS2" in
  let temp = get_temperature hdrh in
  let timestamp = get_timestamp hdrh in
  let data = read_fits_data contents width height in
  
  (* Calculate mean and standard deviation *)
  let sum = ref 0.0 in
  let sum_sq = ref 0.0 in
  let count = width * height in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let value = float_of_int data.(y).(x) in
      sum := !sum +. value;
      sum_sq := !sum_sq +. (value *. value)
    done
  done;
  
  let mean = !sum /. float_of_int count in
  let variance = (!sum_sq /. float_of_int count) -. (mean *. mean) in
  let std_dev = sqrt variance in
  
  { filename; temperature = temp; mean_level = mean; 
    std_dev; timestamp }

(* Calculate temperature coefficient using linear regression *)
let calculate_temp_coefficient stats =
  let n = float_of_int (Array.length stats) in
  let sum_x = Array.fold_left (fun acc s -> acc +. s.temperature) 0.0 stats in
  let sum_y = Array.fold_left (fun acc s -> acc +. s.mean_level) 0.0 stats in
  let sum_xy = Array.fold_left (fun acc s -> 
    acc +. (s.temperature *. s.mean_level)) 0.0 stats in
  let sum_xx = Array.fold_left (fun acc s -> 
    acc +. (s.temperature *. s.temperature)) 0.0 stats in
  
  let slope = ((n *. sum_xy) -. (sum_x *. sum_y)) /. 
              ((n *. sum_xx) -. (sum_x *. sum_x)) in
  let intercept = (sum_y -. (slope *. sum_x)) /. n in
  
  (slope, intercept)

(* Main analysis function *)
let analyze_dark_frames dark_files =
  printf "Analyzing %d dark frames...\n" (Array.length dark_files);
  
  (* Analyze each dark frame *)
  let stats = Array.map analyze_dark_frame dark_files in
  
  (* Sort by temperature *)
  Array.sort (fun a b -> compare a.temperature b.temperature) stats;
  
  (* Calculate temperature coefficient *)
  let (slope, intercept) = calculate_temp_coefficient stats in
  
  (* Output results *)
  printf "\nTemperature Coefficient Analysis:\n";
  printf "================================\n";
  printf "Temperature coefficient: %.3f ADU/°C\n" slope;
  printf "Dark current at 0°C: %.1f ADU\n" intercept;
  printf "\nDetailed measurements:\n";
  printf "Temperature(°C)  Mean Level(ADU)  Std Dev(ADU)  Timestamp\n";
  Array.iter (fun s ->
    printf "%8.1f        %8.1f         %8.1f         %8.1f\n" 
      s.temperature s.mean_level s.std_dev (s.timestamp-.datum)
  ) stats;
  
  (slope, intercept, stats)

(* Entry point *)
let () =
  if Array.length Sys.argv < 2 then
    (printf "Usage: %s <dark1.fits> [dark2.fits ...]\n" Sys.argv.(0);
     exit 1);
  
  let dark_files = Array.sub Sys.argv 1 (Array.length Sys.argv - 1) in
  ignore (analyze_dark_frames dark_files)
d448501e519bb9ec2e4a4fe355366606
echo x - dark_temp_analysis_options.ml
sed 's/^X//' >dark_temp_analysis_options.ml << 'a9f6b5059385b942940ccb9d346ad3b7'
(* dark_temp_analysis.ml *)
open Types
open Fits
open Printf

(* Types of analysis *)
type analysis_mode = 
  | Temperature  (* Just show temperature vs time *)
  | Full        (* Full analysis including statistics and coefficients *)
  | Distribution (* Just show temperature distribution *)

(* Structure for temperature data *)
type dark_stats = {
  filename: string;
  temperature: float;
  timestamp: float;
}

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
        failwith "Could not find temperature in FITS header"

let datum = fst (Unix.mktime {
      tm_sec=0;
      tm_min=0;
      tm_hour=0;
      tm_mday=14;
      tm_mon=1;
      tm_year=2025-1900;
      tm_wday=0;
      tm_yday=0;
      tm_isdst=false;
      })

let get_timestamp hdrh =
  try
    let date_str = String.trim (Hashtbl.find hdrh "DATE-OBS=") in
    (try Scanf.sscanf date_str "'%d-%d-%dT%d:%d:%d'" 
      (fun yr mon day hr min sec -> 
        fst (Unix.mktime {
          tm_sec=sec;
          tm_min=min;
          tm_hour=hr;
          tm_mday=day;
          tm_mon=mon-1;
          tm_year=yr-1900;
          tm_wday=0;
          tm_yday=0;
          tm_isdst=false;
        })) 
    with _ -> 
      printf "Invalid DATE-OBS: %s\n" date_str; 
      0.0)
  with _ -> 
    failwith "Could not find DATE-OBS"

(* Quick scan of just the FITS header *)
let analyze_frame filename =
  let fd = open_in_bin filename in
  let end_re = Str.regexp "END *$" in
  let rec read_header acc =
    let block = Bytes.create 2880 in
    really_input fd block 0 2880;
    let block_str = Bytes.to_string block in
    let acc = acc ^ block_str in
    try
      (* Search for END at 80-byte boundaries *)
      let rec find_end pos =
        if pos >= String.length acc then None
        else if pos mod 80 = 0 then
          let record = String.sub acc pos 80 in
          if Str.string_match end_re record 0 then Some acc
          else find_end (pos + 80)
        else find_end (pos + 80)
      in
      match find_end 0 with
      | Some header -> header
      | None -> read_header acc
    with _ -> read_header acc
  in
  try
    let header = read_header "" in
    let hdrh = Hashtbl.create 257 in
    let rec scan_header pos =
      if pos > String.length header - 80 then hdrh
      else 
        let key = String.sub header pos 80 in
        match String.trim (List.hd (String.split_on_char ' ' key)) with
        | "END" -> hdrh
        | "COMMENT" -> scan_header (pos + 80)
        | oth -> Hashtbl.add hdrh oth (String.sub key (String.length oth) 
                                    (80 - String.length oth));
                scan_header (pos + 80)
    in
    let hdrh = scan_header 0 in
    close_in fd;
    Some { 
      filename = filename;
      temperature = get_temperature hdrh;
      timestamp = get_timestamp hdrh 
    }
  with _ -> 
    close_in fd;
    printf "Warning: Could not read header from %s\n" filename;
    None

(* Show temperature distribution using gnuplot *)
let show_distribution stats =
  let temps = Array.map (fun s -> s.temperature) stats in
  Array.sort compare temps;
  
  (* Write data to temporary file *)
  let tmp_data = Filename.temp_file "dist_data" ".txt" in
  let oc = open_out tmp_data in
  Array.iter (fun t -> fprintf oc "%f\n" t) temps;
  close_out oc;

  (* Create gnuplot script *)
  let tmp_script = Filename.temp_file "gnuplot" ".gp" in
  let oc = open_out tmp_script in
  fprintf oc "set terminal png size 1200,800\n";
  fprintf oc "set output 'distribution.png'\n";
  fprintf oc "set title 'Temperature Distribution'\n";
  fprintf oc "set xlabel 'Temperature (°C)'\n";
  fprintf oc "set ylabel 'Count'\n";
  fprintf oc "set grid\n";
  fprintf oc "binwidth = 0.5\n";  (* Half degree bins *)
  fprintf oc "bin(x,width)=width*floor(x/width)\n";
  fprintf oc "plot '%s' using (bin($1,binwidth)):(1.0) smooth freq with boxes title 'Temperature'\n" tmp_data;
  close_out oc;

  (* Run gnuplot *)
  ignore (Sys.command (sprintf "gnuplot %s" tmp_script));
  
  (* Cleanup *)
  Sys.remove tmp_data;
  Sys.remove tmp_script;
  
  printf "Generated distribution.png\n"

(* Show temperature vs time using gnuplot *)
let show_temperature_time stats =
  (* Write data to temporary file *)
  let tmp_data = Filename.temp_file "temp_data" ".txt" in
  let oc = open_out tmp_data in
  Array.iter (fun s ->
    fprintf oc "%f %f\n" (s.timestamp -. datum) s.temperature
  ) stats;
  close_out oc;

  (* Create gnuplot script *)
  let tmp_script = Filename.temp_file "gnuplot" ".gp" in
  let oc = open_out tmp_script in
  fprintf oc "set terminal png size 1200,800\n";
  fprintf oc "set output 'temperature.png'\n";
  fprintf oc "set title 'Temperature vs Time'\n";
  fprintf oc "set xlabel 'Time (seconds from midnight)'\n";
  fprintf oc "set ylabel 'Temperature (°C)'\n";
  fprintf oc "set grid\n";
  fprintf oc "plot '%s' using 1:2 with lines title 'Temperature'\n" tmp_data;
  close_out oc;

  (* Run gnuplot *)
  ignore (Sys.command (sprintf "gnuplot %s" tmp_script));
  
  (* Cleanup *)
  Sys.remove tmp_data;
  Sys.remove tmp_script;
  
  printf "Generated temperature.png\n"

(* Main analysis function *)
let analyze_dark_frames dark_files mode =
  printf "Analyzing %d dark frames...\n" (Array.length dark_files);
  
  (* Analyze all frames *)
  let stats = Array.map analyze_frame dark_files |> Array.to_list |> 
              List.filter_map (fun x -> x) |> Array.of_list in
  
  (* Sort by timestamp *)
  Array.sort (fun a b -> compare a.timestamp b.timestamp) stats;
  
  match mode with
  | Temperature -> show_temperature_time stats
  | Distribution -> show_distribution stats
  | Full -> begin
      show_distribution stats;
      show_temperature_time stats
    end

(* Parse command line arguments *)
let parse_args () =
  let mode = ref Temperature in
  let files = ref [] in
  let specs = [
    ("-temp", Arg.Unit (fun () -> mode := Temperature), "Show temperature vs time");
    ("-dist", Arg.Unit (fun () -> mode := Distribution), "Show temperature distribution");
    ("-full", Arg.Unit (fun () -> mode := Full), "Show full analysis");
  ] in
  let add_file f = files := f :: !files in
  let usage = sprintf "Usage: %s [-temp|-dist|-full] <dark1.fits> [dark2.fits ...]\n" Sys.argv.(0) in
  Arg.parse specs add_file usage;
  (!mode, Array.of_list (List.rev !files))

(* Entry point *)
let () =
  let mode, dark_files = parse_args () in
  if Array.length dark_files = 0 then
    failwith "No input files specified"
  else
    analyze_dark_frames dark_files modea9f6b5059385b942940ccb9d346ad3b7
echo x - dark_temp_analysis_slow.ml
sed 's/^X//' >dark_temp_analysis_slow.ml << '4bbcf88d45b060b0b8da8abd2c94bb7f'
(* dark_temp_analysis.ml *)
open Types
open Fits
open Printf

(* Structure to hold dark frame analysis results *)
type dark_stats = {
  filename: string;
  temperature: float;
  mean_level: float;
  std_dev: float;
  timestamp: float;
}

(* Structure for initial temperature scan *)
type temp_info = {
  filename: string;
  temp: float;
}

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
        failwith "Could not find temperature in FITS header"

let datum = fst (Unix.mktime {
      tm_sec=0;
      tm_min=0;
      tm_hour=0;
      tm_mday=14;
      tm_mon=1;
      tm_year=2025-1900;
      tm_wday=0;
      tm_yday=0;
      tm_isdst=false;
      })

(* Quick scan of just the FITS header for temperature *)
let scan_fits_temperature filename =
  let img = read_image filename in
  let hdrh, _ = find_header_end img in
  try
    Some { filename; temp = get_temperature hdrh }
  with _ -> 
    printf "Warning: Could not read temperature from %s\n" filename;
    None

(* Find relevant temperature range *)
let find_temp_range temps =
  let valid_temps = Array.to_list temps |> List.filter_map (fun x -> x) in
  match valid_temps with
  | [] -> failwith "No valid temperature readings found"
  | temps ->
      let sorted = List.sort (fun a b -> compare a.temp b.temp) temps in
      let min_temp = (List.hd sorted).temp in
      let max_temp = (List.hd (List.rev sorted)).temp in
      let temp_count = Array.make (int_of_float (ceil (max_temp -. min_temp)) + 1) 0 in
      List.iter (fun t -> 
        let idx = int_of_float (t.temp -. min_temp) in
        temp_count.(idx) <- temp_count.(idx) + 1
      ) temps;
      
      printf "\nTemperature Distribution:\n";
      printf "========================\n";
      Array.iteri (fun i count ->
        if count > 0 then
          printf "%.1f°C: %d frames %s\n" 
            (min_temp +. float_of_int i) 
            count 
            (String.make (count / 10) '*')
      ) temp_count;
      (min_temp, max_temp)

let get_timestamp hdrh =
  try
    let date_str = String.trim (Hashtbl.find hdrh "DATE-OBS=") in
    (try Scanf.sscanf date_str "'%d-%d-%dT%d:%d:%d'" 
      (fun yr mon day hr min sec -> 
        fst (Unix.mktime {
          tm_sec=sec;
          tm_min=min;
          tm_hour=hr;
          tm_mday=day;
          tm_mon=mon-1;
          tm_year=yr-1900;
          tm_wday=0;
          tm_yday=0;
          tm_isdst=false;
        })) 
    with _ -> 
      printf "Invalid DATE-OBS: %s\n" date_str; 
      0.0)
  with _ -> 
    failwith "Could not find DATE-OBS"

let analyze_dark_frame filename =
  let img = read_image filename in
  let hdrh, contents = find_header_end img in
  let width = parse_int hdrh "NAXIS1" in
  let height = parse_int hdrh "NAXIS2" in
  let temp = get_temperature hdrh in
  let timestamp = get_timestamp hdrh in
  let data = read_fits_data contents width height in
  
  let sum = ref 0.0 in
  let sum_sq = ref 0.0 in
  let count = width * height in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let value = float_of_int data.(y).(x) in
      sum := !sum +. value;
      sum_sq := !sum_sq +. (value *. value)
    done
  done;
  
  let mean = !sum /. float_of_int count in
  let variance = (!sum_sq /. float_of_int count) -. (mean *. mean) in
  let std_dev = sqrt variance in
  
  { filename; temperature = temp; mean_level = mean; 
    std_dev; timestamp }

let calculate_temp_coefficient dark_stats temp_infos =
  let n = float_of_int (Array.length dark_stats) in
  (* Create lookup for quick temperature access *)
  let temp_map = Hashtbl.create (Array.length temp_infos) in
  Array.iter (fun temp_info -> 
    Hashtbl.add temp_map temp_info.filename temp_info.temp
  ) temp_infos;
  
  (* Calculate sums for regression *)
  let sum_x = ref 0.0 in
  let sum_y = ref 0.0 in
  let sum_xy = ref 0.0 in
  let sum_xx = ref 0.0 in
  
  Array.iter (fun (stats:dark_stats) ->
    match Hashtbl.find_opt temp_map stats.filename with
    | Some temp ->
        sum_x := !sum_x +. temp;
        sum_y := !sum_y +. stats.mean_level;
        sum_xy := !sum_xy +. (temp *. stats.mean_level);
        sum_xx := !sum_xx +. (temp *. temp)
    | None -> printf "Warning: No temperature data for %s\n" stats.filename
  ) dark_stats;
  
  let slope = ((n *. !sum_xy) -. (!sum_x *. !sum_y)) /. 
              ((n *. !sum_xx) -. (!sum_x *. !sum_x)) in
  let intercept = (!sum_y -. (slope *. !sum_x)) /. n in
  
  (slope, intercept)

(* Main analysis function with temperature range filtering *)
let analyze_dark_frames dark_files target_temp_range =
  printf "Quick scanning %d dark frames for temperature...\n" (Array.length dark_files);
  
  (* Quick temperature scan *)
  let temp_infos = Array.map scan_fits_temperature dark_files in
  
  (* Show temperature distribution *)
  let valid_temps = Array.to_list temp_infos |> List.filter_map (fun x -> x) in
  match valid_temps with
  | [] -> failwith "No valid temperature readings found"
  | temps ->
      let sorted = List.sort (fun a b -> compare a.temp b.temp) temps in
      let min_temp = (List.hd sorted).temp in
      let max_temp = (List.hd (List.rev sorted)).temp in
      let temp_count = Array.make (int_of_float (ceil (max_temp -. min_temp)) + 1) 0 in
      List.iter (fun t -> 
        let idx = int_of_float (t.temp -. min_temp) in
        temp_count.(idx) <- temp_count.(idx) + 1
      ) temps;
      
      printf "\nTemperature Distribution:\n";
      printf "========================\n";
      Array.iteri (fun i count ->
        if count > 0 then
          printf "%.1f°C: %d frames %s\n" 
            (min_temp +. float_of_int i) 
            count 
            (String.make (count / 10) '*')
      ) temp_count;
      
      (* Filter frames within target range *)
      let (target_min, target_max) = target_temp_range in
      let range_margin = 2.0 in (* Look at darks within ±2°C of target range *)
      let filtered_files = List.filter_map (fun info ->
        if info.temp >= (target_min -. range_margin) && 
           info.temp <= (target_max +. range_margin)
        then Some info.filename
        else None
      ) temps |> Array.of_list in
      
      printf "\nAnalyzing %d dark frames within temperature range %.1f°C to %.1f°C...\n" 
        (Array.length filtered_files) (target_min -. range_margin) (target_max +. range_margin);
      
      (* Only analyze the filtered subset *)
      let stats = Array.map analyze_dark_frame filtered_files in
      Array.sort (fun a b -> compare a.temperature b.temperature) stats;
      
      let filtered_temp_infos = Array.of_list (List.filter (fun t -> 
        Array.exists (fun f -> f = t.filename) filtered_files) temps) in
      let (slope, intercept) = calculate_temp_coefficient stats filtered_temp_infos in
      
      (* Output results *)
      printf "\nTemperature Coefficient Analysis:\n";
      printf "================================\n";
      printf "Temperature coefficient: %.3f ADU/°C\n" slope;
      printf "Dark current at 0°C: %.1f ADU\n" intercept;
      printf "\nDetailed measurements:\n";
      printf "Temperature(°C)  Mean Level(ADU)  Std Dev(ADU)  Timestamp\n";
      Array.iter (fun s ->
        printf "%8.1f        %8.1f         %8.1f         %8.1f\n" 
          s.temperature s.mean_level s.std_dev (s.timestamp -. datum)
      ) stats;
      
      (slope, intercept, stats)

(* Entry point *)
let () =
  if Array.length Sys.argv < 2 then
    (printf "Usage: %s <dark1.fits> [dark2.fits ...]\n" Sys.argv.(0);
     exit 1);
  
  let dark_files = Array.sub Sys.argv 1 (Array.length Sys.argv - 1) in
  ignore (analyze_dark_frames dark_files (12.0, 20.0))4bbcf88d45b060b0b8da8abd2c94bb7f
echo x - fits.ml
sed 's/^X//' >fits.ml << '07ebcb73fede067e5e92da0c496e8fb7'
(* fits.ml *)
open Printf
open Types

(* FITS header block size *)
let block_size = 2880
let header_record_size = 80
let end_re = Str.regexp "END *$"

let rec read_header fd acc =
  let block = Bytes.create 2880 in
  really_input fd block 0 2880;
  let block_str = Bytes.to_string block in
  let acc = acc ^ block_str in
  try
    let rec find_end pos =
      if pos >= String.length acc then None
      else if pos mod 80 = 0 then
	let record = String.sub acc pos 80 in
	if Str.string_match end_re record 0 then Some acc
	else find_end (pos + 80)
      else find_end (pos + 80)
    in
    match find_end 0 with
    | Some header -> header
    | None -> read_header fd acc
  with _ -> read_header fd acc

let rec scan_header hdrh header pos =
  if pos > String.length header - 80 then ()
  else 
    let key = String.sub header pos 80 in
    match String.trim (List.hd (String.split_on_char ' ' key)) with
    | "END" -> ()
    | "COMMENT" -> scan_header hdrh header (pos + 80)
    | oth -> Hashtbl.add hdrh oth (String.sub key (String.length oth) 
				(80 - String.length oth));
	    scan_header hdrh header (pos + 80)

(* Read entire FITS file *)
let read_image image =
    let rs = ref "" in
    let fd = open_in_bin image in
    (try rs := really_input_string fd (in_channel_length fd) with End_of_file -> ());
    close_in fd;
    !rs

(* Parse FITS header, return header hashtable and data start *)
let find_header_end filename data =
    let hlen = ref block_size in
    let hdrh = Hashtbl.create 257 in
    let rec scan_for_end pos =
        if pos > String.length data - header_record_size then 
            error (filename^": header_end not found")
        else 
            let key = String.sub data pos header_record_size in 
            match String.trim (List.hd (String.split_on_char ' ' key)) with
            | "END" -> hlen := (((pos + block_size) / block_size) * block_size)
            | "COMMENT" -> scan_for_end (pos + header_record_size)
            | oth -> Hashtbl.add hdrh oth (String.sub key (String.length oth) 
                                         (header_record_size - String.length oth)); 
                    scan_for_end (pos + header_record_size)
    in
    scan_for_end 0;
    hdrh, String.sub data !hlen (String.length data - !hlen)

(* Parse WCS parameters from header *)
let parse_wcs hdrh =
    {
        ra_2000 = parse_float hdrh "CRVAL1";
        dec_2000 = parse_float hdrh "CRVAL2";
        crpix1 = parse_float hdrh "CRPIX1";
        crpix2 = parse_float hdrh "CRPIX2";
        cd1_1 = parse_float hdrh "CD1_1";
        cd1_2 = parse_float hdrh "CD1_2";
        cd2_1 = parse_float hdrh "CD2_1";
        cd2_2 = parse_float hdrh "CD2_2"
    }

(* Read raw FITS image data into array *)
let read_fits_data contents width height =
    let data = Array.make_matrix height width 0 in
    for y = 0 to height-1 do
        let row_offset = y * width * 2 in
        for x = 0 to width-1 do
            let off = row_offset + x * 2 in
            let value = (int_of_char contents.[off] lsl 8) lor 
                       (int_of_char contents.[off+1]) in
            data.(y).(x) <- value
        done
    done;
    data

(* Write RGB data as FITS *)
let write_rgb_fits filename data wcs exposure = let open Printf in
    let oc = open_out_bin filename in
    (* Write FITS header *)
    let header = sprintf "%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s%-80s"
        "SIMPLE  =                    T / file does conform to FITS standard"
        "BITPIX  =                   16 / number of bits per data pixel"
        "NAXIS   =                    3 / number of data axes"
        (sprintf "NAXIS1  =                 %4d / length of data axis 1" (Array.length data.(0)))
        (sprintf "NAXIS2  =                 %4d / length of data axis 2" (Array.length data))
        "NAXIS3  =                    3 / length of data axis 3 (RGB)"
        "EXTEND  =                    T / FITS dataset may contain extensions"
	"BZERO   =                32768 / offset data range to that of unsigned short"
	"BSCALE  =                    1 / default scaling factor"
	(sprintf "EXPOSURE=                 %4d / Exposure time in ms" exposure)
        "CTYPE1  = 'RA---TAN'           / Right ascension, gnomonic projection"
        "CTYPE2  = 'DEC--TAN'           / Declination, gnomonic projection"
        (sprintf "CRVAL1  =              %f / Reference right ascension" wcs.ra_2000)
        (sprintf "CRVAL2  =              %f / Reference declination" wcs.dec_2000)
        (sprintf "CRPIX1  =                %f / Reference pixel along axis 1" (wcs.crpix1 /. 2.0))  (* Adjust reference pixel for binning *)
        (sprintf "CRPIX2  =                %f / Reference pixel along axis 2" (wcs.crpix2 /. 2.0))  
        (sprintf "CD1_1   =          %f / CD matrix element" (wcs.cd1_1 *. 2.0))        (* Adjust pixel scale for binning *)
        (sprintf "CD1_2   =          %f / CD matrix element" (wcs.cd1_2 *. 2.0))
        (sprintf "CD2_1   =          %f / CD matrix element" (wcs.cd2_1 *. 2.0))
        (sprintf "CD2_2   =          %f / CD matrix element" (wcs.cd2_2 *. 2.0))
        "END" in
    output_string oc header;
    (* Pad header to multiple of 2880 bytes *)
    let padding = String.make (2880 - (String.length header mod 2880)) ' ' in
    output_string oc padding;
    (* Write RGB data *)
    let plane = Array.length data * Array.length data.(0) * 2 in
    let buf = Bytes.create (plane*3) in
    let posr = ref 0 in
    let posg = ref plane in
    let posb = ref (plane*2) in
    for y = 0 to Array.length data - 1 do
        for x = 0 to Array.length data.(0) - 1 do
            let (r,g,b) = data.(y).(x) in
            Bytes.set buf (!posr) (char_of_int ( r / 256));
            incr posr;
            Bytes.set buf (!posr) (char_of_int ( r mod 256));
            incr posr;
            Bytes.set buf (!posg) (char_of_int ( g / 256));
            incr posr;
            Bytes.set buf (!posg) (char_of_int ( g mod 256));
            incr posr;
            Bytes.set buf (!posb) (char_of_int ( b / 256));
            incr posr;
            Bytes.set buf (!posb) (char_of_int ( b mod 256));
            incr posr;
        done
    done;
    output_bytes oc buf;
    (* Pad image to multiple of 2880 bytes *)
    let padding = String.make (2880 - (Bytes.length buf mod 2880)) ' ' in
    output_string oc padding;
    close_out oc
07ebcb73fede067e5e92da0c496e8fb7
echo x - group.ml
sed 's/^X//' >group.ml << '96c57865882061cbbb3add54e9be37f6'
(* group.ml *)
open Printf
open Types
open Process

(* Initial grid-based grouping *)
let assign_to_grid images =
    let ra_vals = List.map (fun img -> img.wcs.ra_2000) images in
    let dec_vals = List.map (fun img -> img.wcs.dec_2000) images in
    let ra_min = List.fold_left min (List.hd ra_vals) (List.tl ra_vals) in
    let ra_max = List.fold_left max (List.hd ra_vals) (List.tl ra_vals) in
    let dec_min = List.fold_left min (List.hd dec_vals) (List.tl dec_vals) in
    let dec_max = List.fold_left max (List.hd dec_vals) (List.tl dec_vals) in

    (* Calculate field dimensions in degrees *)
    let ra_width = ra_max -. ra_min in
    let dec_height = dec_max -. dec_min in

    (* Typical pixel scale for this instrument is 1.25 arcsec/pixel *)
    let pixel_scale = 1.25 /. 3600.0 in  (* degrees per pixel *)

    (* Calculate pixel dimensions *)
    let naxis1 = int_of_float (ra_width /. pixel_scale) in
    let naxis2 = int_of_float (dec_height /. pixel_scale) in

    (* Calculate reference pixel at the center *)
    let crpix1 = float_of_int naxis1 /. 2.0 in
    let crpix2 = float_of_int naxis2 /. 2.0 in

    let tmpl = open_out "template.hdr" in
    fprintf tmpl "SIMPLE  = T\n";
    fprintf tmpl "BITPIX  = -64\n";
    fprintf tmpl "NAXIS   = 2\n";
    fprintf tmpl "NAXIS1  = %d\n" naxis1;
    fprintf tmpl "NAXIS2  = %d\n" naxis2;
    fprintf tmpl "CTYPE1  = 'RA---TAN'\n";
    fprintf tmpl "CTYPE2  = 'DEC--TAN'\n";
    fprintf tmpl "EQUINOX = 2000\n";
    fprintf tmpl "CRVAL1  = %4.4f\n" ((ra_min +. ra_max) /. 2.);
    fprintf tmpl "CRVAL2  = %4.4f\n" ((dec_min +. dec_max) /. 2.);
    fprintf tmpl "CRPIX1  = %4.4f\n" crpix1;
    fprintf tmpl "CRPIX2  = %4.4f\n" crpix2;
    fprintf tmpl "CDELT1  = %f\n" (-. pixel_scale);
    fprintf tmpl "CDELT2  = %f\n" pixel_scale;
    fprintf tmpl "CROTA2  = 358.501127767\n";
    fprintf tmpl "END\n";
    close_out tmpl;

    printf "RA range: %f to %f (width: %f degrees)\n" ra_min ra_max ra_width;
    printf "Dec range: %f to %f (height: %f degrees)\n" dec_min dec_max dec_height;
    printf "Calculated image size: %d x %d pixels\n" naxis1 naxis2;

    (* Rest of the existing function remains the same *)
    let ra_step = ra_width /. float_of_int ra_bins in
    let dec_step = dec_height /. float_of_int dec_bins in
    
    let initial_groups = Hashtbl.create (ra_bins * dec_bins) in
    
    (* Assign images to grid cells *)
    List.iter (fun img ->
        let r = int_of_float ((img.wcs.ra_2000 -. ra_min) /. ra_step) in
        let d = int_of_float ((img.wcs.dec_2000 -. dec_min) /. dec_step) in
        let r = max 0 (min (ra_bins - 1) r) in
        let d = max 0 (min (dec_bins - 1) d) in
        let id = sprintf "r%dd%d" r d in
        let group = try 
            let existing = Hashtbl.find_all initial_groups id in
            match existing with
            | [images] -> images @ [img]
            | _ -> [img]
        with Not_found -> [img] in
        Hashtbl.replace initial_groups id group
    ) images;
    
    (* Convert hashtbl to list of group_info *)
    Hashtbl.fold (fun id frames acc -> 
        if List.length frames > 0 then 
            {id=id; files=frames} :: acc 
        else acc
    ) initial_groups []

(* Split large groups *)
let rec split_if_needed group =
    if List.length group.files <= max_group_size then [group]
    else
        let n_subgroups = (List.length group.files + max_group_size - 1) / max_group_size in
        let chunk_size = min max_group_size (List.length group.files / n_subgroups) in
        let sorted = List.sort 
            (fun i1 i2 -> compare i1.wcs.ra_2000 i2.wcs.ra_2000) 
            group.files in
        let rec make_subgroups files n acc =
            if files = [] then List.rev acc
            else if n = 1 then List.rev ({id=sprintf "%s_%d" group.id n; files=files} :: acc)
            else
                let these, rest = List.fold_left
                    (fun (first, rest) item ->
                        if List.length first < chunk_size then
                            (item :: first, rest)
                        else
                            (first, item :: rest))
                    ([], []) files in
                make_subgroups rest (n-1) ({id=sprintf "%s_%d" group.id n; files=these} :: acc)
        in
        make_subgroups sorted n_subgroups []

(* Merge small groups *)
let rec merge_small_groups groups =
    let is_small g = List.length g.files < min_group_size in
    match List.partition is_small groups with
    | [], final -> final  (* No more small groups *)
    | small_groups, large_groups ->
        (* Find closest pair of small-to-large groups *)
        let best_merge = ref None in
        let min_dist = ref max_float in
        List.iter (fun small ->
            List.iter (fun large ->
                let d = distance_between_groups (small.id, small.files) (large.id, large.files) in
                if d < !min_dist then (
                    min_dist := d;
                    best_merge := Some(small, large)
                )
            ) large_groups
        ) small_groups;
        
        match !best_merge with
        | None -> groups (* No viable merges found *)
        | Some(small, large) ->
            let merged = {id=large.id; files=small.files @ large.files} in
            let remaining_small = List.filter (fun g -> g.id <> small.id) small_groups in
            let remaining_large = List.filter (fun g -> g.id <> large.id) large_groups in
            merge_small_groups (merged :: (remaining_small @ remaining_large))

(* Main grouping function *)
let create_groups images =
    let initial = assign_to_grid images in
    
    (* Split large groups *)
    let split_groups = List.fold_left (fun acc group ->
        acc @ split_if_needed group
    ) [] initial in
    
    (* Merge small groups *)
    let final = merge_small_groups split_groups in
    
    (* Print group info *)
    List.iter (fun group ->
        printf "Created group %s with %d images\n" group.id (List.length group.files)
    ) final;
    
    final
96c57865882061cbbb3add54e9be37f6
echo x - main.ml
sed 's/^X//' >main.ml << 'a4e93d185cd44ea8283e58f554c628ba'
(* main.ml *)
open Printf
open Types
open Fits
open Process
open Group
open Make

(* Main program *)
let () = match Array.length Sys.argv with
| 3 | 4 -> Process_fits.process_fits()
| 2 -> 
    begin
    let dark_frame = Sys.argv.(1) in
    
    (* Validate dark frame *)
    let _ = load_dark_frame dark_frame in

    let oc = open_out "Makefile" in
    emit_makefile_header oc;
    emit_build_rules oc;
    List.iter (fun colour -> let pth = "processed_"^colour in
    (* Collect input files *)
    let files = ref [] in
    let dir = Unix.opendir pth in
    (try while true do
        let fil = Unix.readdir dir in
        match String.split_on_char '.' fil with
        | hd::"fits"::[] when fil <> dark_frame -> 
            files := (pth^"/"^fil) :: !files
        | _ -> ()
    done with End_of_file -> Unix.closedir dir);
    let files = List.sort compare !files in
    
    (* Get WCS info *)
    let images = List.map get_image_info files in
    
    printf "Found %d FITS files\n" (List.length images);

    (* Create groups *)
    let groups = create_groups images in
    printf "\nFiles have been organized into spatial groups.\n";
    
    (* Generate Makefile *)
    generate_makefile oc dark_frame files groups colour;
    
    printf "Generated Makefile\n";
    printf "1. Run 'make -j8 process' to calibrate all frames\n";
    printf "2. Run 'make -j8 groups' to create regional mosaics\n";
    printf "3. Run 'make mosaic' to create final mosaic\n";
    printf "Or simply run 'make -j8' to do everything\n"

    ) ["r";"g";"b"];
    close_out oc;
    end
| oth -> failwith (Sys.argv.(0)^" master_dark.fits")
a4e93d185cd44ea8283e58f554c628ba
echo x - make.ml
sed 's/^X//' >make.ml << '22dbb274868b019b96ab203a67a6f19e'
(* make.ml *)
open Printf
open Types

(* Makefile header *)
let emit_makefile_header oc =
    fprintf oc "# Generated Makefile - do not edit\n";
    fprintf oc "SHELL := /bin/bash\n\n";
    fprintf oc ".PHONY: all clean_r clean_g clean_b groups_r groups_g groups_b s_r mosaic_r mosaic_g mosaic_b\n\n";
    fprintf oc "all: mosaic_r mosaic_g mosaic_b\n";
    fprintf oc "mosaic_r: groups_r\n\n";
    fprintf oc "mosaic_g: groups_g\n\n";
    fprintf oc "mosaic_b: groups_b\n\n"

(* Rules to build processors *)
let emit_build_rules oc =
    let dep = "types.ml fits.ml process.ml group.ml make.ml process_fits.ml main.ml" in
    fprintf oc "mosaic_fits: %s\n" dep;
    fprintf oc "\tocamlopt -g str.cmxa -I +unix -I +str unix.cmxa %s -o $@\n\n" dep

(* Rules for image processing *)
let emit_process_rules oc colour dark_frame files =
    fprintf oc "process:";
    List.iter (fun file ->
        fprintf oc " processed_%s/%s.fits" colour (Filename.chop_extension (Filename.basename file))
    ) files;
    fprintf oc "\n\n";
    
    List.iter (fun file ->
        let base = Filename.chop_extension (Filename.basename file) in
        fprintf oc "processed_%s/%s.fits: %s %s mosaic_fits | processed_r\n" 
            colour base file dark_frame;
        fprintf oc "\t./mosaic_fits %s processed_%s %s\n\n" dark_frame colour file
    ) files;
    
    fprintf oc "processed_%s:\n\tmkdir -p $@\n\n" colour

(* Rules for group processing *)
let emit_group_rules oc colour groups =
    (* Declare precious intermediate files *)
    fprintf oc ".PRECIOUS: %%/images.tbl %%/proj_images.tbl %%/diffs.tbl %%/fits.tbl %%/corrections.tbl %%/mosaic.tbl %%/symlinks.stamp\n\n";
    
    fprintf oc "groups_%s: " colour;
    List.iter (fun group ->
        fprintf oc "groups_%s/%s/region_mosaic_%s.fits " colour group.id colour
    ) groups;
    fprintf oc "\n\n";
    
    List.iter (fun group ->
        let dir = sprintf "groups_%s/%s" colour group.id in
        
        (* Directory creation rules *)
        fprintf oc "%s:\n\tmkdir -p $@\n\n" dir;
        fprintf oc "%s/projected:\n\tmkdir -p $@\n\n" dir;
        fprintf oc "%s/diffdir:\n\tmkdir -p $@\n\n" dir;
        fprintf oc "%s/corrdir:\n\tmkdir -p $@\n\n" dir;
        
        (* Symlinks with stamp file *)
        fprintf oc "%s/symlinks.stamp: | %s\n" dir dir;
        List.iter (fun file ->
            fprintf oc "\tln -sf ../../processed_%s/%s.fits %s/%s.fits\n" colour
                (Filename.chop_extension (Filename.basename file.filename))
                dir
                (Filename.chop_extension (Filename.basename file.filename))
        ) group.files;
        fprintf oc "\ttouch $@\n\n";
        
        (* Processing chain *)
        fprintf oc "%s/images.tbl: %s/symlinks.stamp | %s\n" dir dir dir;
        fprintf oc "\tcd %s && mImgtbl ./ $(@F)\n\n" dir;
        
        fprintf oc "%s/proj_images.tbl: %s/images.tbl template.hdr | %s/projected\n" dir dir dir;
        fprintf oc "\tcd %s && mProjExec -p ./ images.tbl ../../template.hdr projected stats.tbl\n" dir;
        fprintf oc "\tcd %s && mImgtbl projected $(@F)\n\n" dir;
        
        fprintf oc "%s/diffs.tbl: %s/proj_images.tbl\n" dir dir;
        fprintf oc "\tcd %s && mOverlaps proj_images.tbl $(@F)\n\n" dir;
        
        fprintf oc "%s/fits.tbl: %s/diffs.tbl | %s/diffdir\n" dir dir dir;
        fprintf oc "\tcd %s && mDiffExec -p projected diffs.tbl ../../template.hdr diffdir\n" dir;
        fprintf oc "\tcd %s && mFitExec diffs.tbl $(@F) diffdir\n\n" dir;
        
        fprintf oc "%s/corrections.tbl: %s/fits.tbl\n" dir dir;
        fprintf oc "\tcd %s && mBgModel proj_images.tbl fits.tbl $(@F)\n\n" dir;
        
        fprintf oc "%s/region_mosaic_%s.fits: %s/corrections.tbl | %s/corrdir\n" dir colour dir dir;
        fprintf oc "\tcd %s && mBgExec -p projected proj_images.tbl corrections.tbl corrdir\n" dir;
        fprintf oc "\tcd %s && mImgtbl corrdir mosaic.tbl\n" dir;
        fprintf oc "\tcd %s && mAdd -p corrdir mosaic.tbl ../../template.hdr region_mosaic_%s.fits\n\n" dir colour;
        
    ) groups

(* Rules for final_rgb mosaic_rgb *)
let emit_mosaic_rules oc colour groups =
    fprintf oc "mosaic_%s: final_%s/ic434_mosaic_%s.fits\n\n" colour colour colour;
    fprintf oc "final_%s/ic434_mosaic_%s.fits:" colour colour;
    List.iter (fun group ->
        fprintf oc " groups_%s/%s/region_mosaic_%s.fits" colour group.id colour
    ) groups;
    fprintf oc " | final_%s\n" colour;
    fprintf oc "\tcd final_%s && rm -f *.fits\n" colour;
    fprintf oc "\tcd final_%s && for i in `find .. -name \"region_mosaic_%s.fits\"`; do \\\n" colour colour;
    fprintf oc "\t\tln -sf $$i `echo $$i|sed 's=[^a-z0-9]=_=g'`.fits; \\\n";
    fprintf oc "\tdone\n";
    fprintf oc "\tcd final_%s && mImgtbl ./ images.tbl\n" colour;
    fprintf oc "\tcd final_%s && mkdir -p projected diffdir corrdir\n" colour;
    fprintf oc "\tcd final_%s && mProjExec -p ./ images.tbl ../template.hdr projected stats.tbl\n" colour;
    fprintf oc "\tcd final_%s && mImgtbl projected images_projected.tbl\n" colour;
    fprintf oc "\tcd final_%s && mOverlaps images_projected.tbl diffs.tbl\n" colour;
    fprintf oc "\tcd final_%s && mDiffExec -p projected diffs.tbl ../template.hdr diffdir\n" colour;
    fprintf oc "\tcd final_%s && mFitExec diffs.tbl fits.tbl diffdir\n" colour;
    fprintf oc "\tcd final_%s && mBgModel images_projected.tbl fits.tbl corrections.tbl\n" colour;
    fprintf oc "\tcd final_%s && mBgExec -p projected images_projected.tbl corrections.tbl corrdir\n" colour;
    fprintf oc "\tcd final_%s && mImgtbl corrdir images_corrected.tbl\n" colour;
    fprintf oc "\tcd final_%s && mAdd -p corrdir images_corrected.tbl ../template.hdr ic434_mosaic_%s.fits\n\n" colour colour;
    fprintf oc "final_%s:\n\tmkdir -p $@\n\n" colour

(* Cleanup rules *)
let emit_clean_rules oc colour =
    fprintf oc "clean_%s:\n" colour;
    fprintf oc "\trm -rf processed_%s groups_%s final_%s\n" colour colour colour;
    fprintf oc "\trm -f template.hdr mosaic_%s.fits *.o\n" colour

(* Generate complete Makefile *)
let generate_makefile oc dark_frame files groups colour =
(*
    emit_process_rules oc dark_frame files;
 *)
    emit_group_rules oc colour groups;
    emit_mosaic_rules oc colour groups;
    emit_clean_rules oc colour
22dbb274868b019b96ab203a67a6f19e
echo x - noise_analysis.ml
sed 's/^X//' >noise_analysis.ml << '8b97b8db9355b8d967ba6492c9ef9107'
(* YCbCr conversion and noise analysis *)
open Types

(* RGB to YCbCr conversion *)
let rgb_to_ycbcr (r, g, b) =
    (* Convert RGB values to float in 0-1 range *)
    let r' = float_of_int r /. 65535.0 in
    let g' = float_of_int g /. 65535.0 in
    let b' = float_of_int b /. 65535.0 in
    
    (* Standard RGB -> YCbCr transformation *)
    let y  =  0.299 *. r' +. 0.587 *. g' +. 0.114 *. b' in
    let cb = -0.169 *. r' -. 0.331 *. g' +. 0.500 *. b' in
    let cr =  0.500 *. r' -. 0.419 *. g' -. 0.081 *. b' in
    
    (y, cb, cr)

(* Convert entire RGB array to YCbCr *)
let convert_array_to_ycbcr rgb_data =
    let height = Array.length rgb_data in
    let width = Array.length rgb_data.(0) in
    let ycbcr = Array.make_matrix height width (0., 0., 0.) in
    
    for y = 0 to height - 1 do
        for x = 0 to width - 1 do
            ycbcr.(y).(x) <- rgb_to_ycbcr rgb_data.(y).(x)
        done
    done;
    ycbcr

(* Calculate mean and variance for a channel *)
let calculate_stats channel_data =
    let height = Array.length channel_data in
    let width = Array.length channel_data.(0) in
    let n = float_of_int (height * width) in
    
    (* Calculate mean *)
    let sum = ref 0.0 in
    for y = 0 to height - 1 do
        for x = 0 to width - 1 do
            sum := !sum +. channel_data.(y).(x)
        done
    done;
    let mean = !sum /. n in
    
    (* Calculate variance *)
    let sum_sq_diff = ref 0.0 in
    for y = 0 to height - 1 do
        for x = 0 to width - 1 do
            let diff = channel_data.(y).(x) -. mean in
            sum_sq_diff := !sum_sq_diff +. (diff *. diff)
        done
    done;
    let variance = !sum_sq_diff /. (n -. 1.0) in
    
    (mean, sqrt variance) (* Return mean and standard deviation *)

(* Extract single channel from YCbCr data *)
let extract_channel ycbcr_data channel =
    let height = Array.length ycbcr_data in
    let width = Array.length ycbcr_data.(0) in
    let result = Array.make_matrix height width 0.0 in
    
    for y = 0 to height - 1 do
        for x = 0 to width - 1 do
            let (y', cb, cr) = ycbcr_data.(y).(x) in
            result.(y).(x) <- match channel with
                | `Y  -> y'
                | `Cb -> cb
                | `Cr -> cr
        done
    done;
    result

(* Main noise analysis function *)
let analyze_noise rgb_data =
    let ycbcr = convert_array_to_ycbcr rgb_data in
    
    (* Extract and analyze each channel *)
    let y_channel = extract_channel ycbcr `Y in
    let cb_channel = extract_channel ycbcr `Cb in
    let cr_channel = extract_channel ycbcr `Cr in
    
    let y_mean, y_stddev = calculate_stats y_channel in
    let cb_mean, cb_stddev = calculate_stats cb_channel in
    let cr_mean, cr_stddev = calculate_stats cr_channel in
    
    (* Return analysis results *)
    {
        y_stats = (y_mean, y_stddev);
        cb_stats = (cb_mean, cb_stddev);
        cr_stats = (cr_mean, cr_stddev);
        snr_y = if y_stddev > 0.0 then y_mean /. y_stddev else 0.0;
        snr_cb = if cb_stddev > 0.0 then cb_mean /. cb_stddev else 0.0;
        snr_cr = if cr_stddev > 0.0 then cr_mean /. cr_stddev else 0.0;
    }

(* Calculate local noise variance in Y channel *)
let analyze_local_noise ycbcr_data window_size =
    let height = Array.length ycbcr_data in
    let width = Array.length ycbcr_data.(0) in
    let result = Array.make_matrix height width 0.0 in
    
    for y = window_size to height - window_size - 1 do
        for x = window_size to width - window_size - 1 do
            (* Calculate local statistics in window *)
            let values = ref [] in
            for wy = y - window_size to y + window_size do
                for wx = x - window_size to x + window_size do
                    let (y', _, _) = ycbcr_data.(wy).(wx) in
                    values := y' :: !values
                done
            done;
            
            (* Calculate variance in window *)
            let n = float_of_int ((2 * window_size + 1) * (2 * window_size + 1)) in
            let mean = List.fold_left (+.) 0.0 !values /. n in
            let variance = List.fold_left (fun acc v ->
                let diff = v -. mean in
                acc +. (diff *. diff)
            ) 0.0 !values /. (n -. 1.0) in
            
            result.(y).(x) <- sqrt variance
        done
    done;
    result8b97b8db9355b8d967ba6492c9ef9107
echo x - process.ml
sed 's/^X//' >process.ml << '389c54ede96c3eca8f1c3dd52fd59afd'
(* process.ml *)
open Printf
open Types
open Fits

(* Dark frame subtraction *)
let subtract_dark data dark =
    let height = Array.length data in
    let width = Array.length data.(0) in
    let result = Array.make_matrix height width 0 in
    for y = 0 to height-1 do
        for x = 0 to width-1 do
            result.(y).(x) <- max 0 (data.(y).(x) - dark.(y).(x))
        done
    done;
    result

(* 2x2 binning respecting RGGB Bayer pattern *)
let bin_2x2_bayer data =
    let height = Array.length data in
    let width = Array.length data.(0) in
    let bin_height = height / 2 in
    let bin_width = width / 2 in
    let binned = Array.make_matrix bin_height bin_width (0,0,0) in
    for y = 0 to bin_height-1 do
        for x = 0 to bin_width-1 do
            let r = data.(y*2).(x*2) in         (* R pixel *)
            let g1 = data.(y*2).(x*2+1) in      (* G1 pixel *)
            let g2 = data.(y*2+1).(x*2) in      (* G2 pixel *)
            let b = data.(y*2+1).(x*2+1) in     (* B pixel *)
            let g = (g1 + g2) / 2 in            (* Average G *)
            binned.(y).(x) <- (r, g, b)
        done
    done;
    binned

(* Load and validate dark frame *)
let load_dark_frame filename =
    let img = read_image filename in
    let hdrh, contents = find_header_end img in
    let width = parse_int hdrh "NAXIS1" in
    let height = parse_int hdrh "NAXIS2" in
    
    if width = 0 || height = 0 then
        error (sprintf "Invalid dark frame dimensions: %dx%d" width height);
        
    read_fits_data contents width height

(* Process single FITS file *)
let process_file dark_data infile outfile =
    let img = read_image infile in
    let hdrh, contents = find_header_end img in
    let width = parse_int hdrh "NAXIS1" in
    let height = parse_int hdrh "NAXIS2" in
    let exposure = parse_int hdrh "EXPOSURE" in
    let wcs = parse_wcs hdrh in
    
    if width = 0 || height = 0 then
        error (sprintf "Invalid dimensions in %s: %dx%d" infile width height);
    
    let data = read_fits_data contents width height in
    let calibrated = subtract_dark data dark_data in
    let binned = bin_2x2_bayer calibrated in
    write_rgb_fits outfile binned wcs exposure

(* Get image info without processing *)
let get_image_info filename =
    let img = read_image filename in
    let hdrh, _ = find_header_end img in
    let width = parse_int hdrh "NAXIS1" in
    let height = parse_int hdrh "NAXIS2" in
    let wcs = parse_wcs hdrh in
    
    if width = 0 || height = 0 then
        error (sprintf "Invalid dimensions in %s: %dx%d" filename width height);
    
    { width = width; height = height; wcs = wcs; filename = filename }389c54ede96c3eca8f1c3dd52fd59afd
echo x - process_fits.ml
sed 's/^X//' >process_fits.ml << '9350b3f2052513d834d690943325a27c'
open Types

(* Read FITS header and data *)
let find_header_end data =
    let hlen = ref 2880 in
    let hdrh = Hashtbl.create 257 in
    let rec scan_for_end pos =
        if pos > String.length data - 80 then failwith "header_end not found"
        else let key = String.sub data pos 80 in 
        (match String.trim (List.hd (String.split_on_char ' ' key)) with
        | "END" -> hlen := (((pos + 2880) / 2880) * 2880)
        | "COMMENT" -> scan_for_end (pos + 80)
        | oth -> Hashtbl.add hdrh oth (String.sub key (String.length oth) (80 - String.length oth)); 
                 scan_for_end (pos + 80))
    in scan_for_end 0;
    hdrh, String.sub data !hlen (String.length data - !hlen)

let read_image image =
    let rs = ref "" in
    let fd = open_in_bin image in
    (try rs := really_input_string fd (in_channel_length fd) with End_of_file -> ());
    close_in fd;
    !rs

(* Parse WCS parameters *)
let parse_wcs hdrh =
    let get_float key = 
        try Scanf.sscanf (Hashtbl.find hdrh key) " = %f /" (fun f->f)
        with Not_found -> 0.0
    in
    {
        ra_2000 = get_float "CRVAL1";
        dec_2000 = get_float "CRVAL2"; 
        crpix1 = get_float "CRPIX1";
        crpix2 = get_float "CRPIX2";
        cd1_1 = get_float "CD1_1";
        cd1_2 = get_float "CD1_2";
        cd2_1 = get_float "CD2_1";
        cd2_2 = get_float "CD2_2"
    }

(* Apply dark calibration *)
let subtract_dark data dark =
    let height = Array.length data in
    let width = Array.length data.(0) in
    let result = Array.make_matrix height width 0 in
    for y = 0 to height-1 do
        for x = 0 to width-1 do
            result.(y).(x) <- max 0 (data.(y).(x) - dark.(y).(x))
        done
    done;
    result

(* 2x2 binning with Bayer pattern awareness *)
let bin_2x2 data =
    let height = Array.length data in
    let width = Array.length data.(0) in
    let bin_height = height / 2 in
    let bin_width = width / 2 in
    let binned = Array.make_matrix bin_height bin_width (0,0,0) in
    for y = 0 to bin_height-1 do
        for x = 0 to bin_width-1 do
            let r = data.(y*2).(x*2) in         (* R pixel *)
            let g1 = data.(y*2).(x*2+1) in      (* G1 pixel *)
            let g2 = data.(y*2+1).(x*2) in      (* G2 pixel *)
            let b = data.(y*2+1).(x*2+1) in     (* B pixel *)
            let g = (g1 + g2) / 2 in            (* Average G *)
            binned.(y).(x) <- (r, g, b)
        done
    done;
    binned

(* Main processing *)
let process_image filename dark_data =
    let img = read_image filename in
    let hdrh, contents = find_header_end img in
    let get_int key = 
        try Scanf.sscanf (Hashtbl.find hdrh key) " = %d /" (fun i->i)
        with Not_found -> 0 in
    let width = get_int "NAXIS1" in
    let height = get_int "NAXIS2" in
    let expos = get_int "EXPOSURE" in
    let wcs = parse_wcs hdrh in
    let data = Fits.read_fits_data contents width height in
    let calibrated = subtract_dark data dark_data in
    let binned = bin_2x2 calibrated in
    { width = width/2; height = height/2; wcs = wcs; filename = filename }, binned, expos

let process_fits () = let open Printf in
    (* Read dark frame *)
    let dark_img = read_image Sys.argv.(1) in
    let dark_hdrh, dark_contents = find_header_end dark_img in
    let get_int key = 
        try Scanf.sscanf (Hashtbl.find dark_hdrh key) " = %d /" (fun i->i)
        with Not_found -> 
            (printf "Error: Could not find %s in dark frame header\n" key;
             exit 1) in
    let dark_width = get_int "NAXIS1" in
    let dark_height = get_int "NAXIS2" in
    if dark_width = 0 || dark_height = 0 then
        (printf "Error: Invalid dark frame dimensions: %dx%d\n" dark_width dark_height;
         exit 1);
    printf "Dark frame size: %dx%d\n" dark_width dark_height;
    let dark_data = Fits.read_fits_data dark_contents dark_width dark_height in
    
    (* Process each input file *)
    let output_dir = Sys.argv.(2) in
    (try Unix.mkdir output_dir 0o755 with Unix.Unix_error(Unix.EEXIST, _, _) -> ());
    
    for i = 3 to Array.length Sys.argv - 1 do
        let input_file = Sys.argv.(i) in
        let base = Filename.basename input_file in
        let info, binned, expos = process_image input_file dark_data in
        let output_file = Filename.concat output_dir 
            (Filename.chop_extension base ^ "_proc.fits") in
        Fits.write_rgb_fits output_file binned info.wcs expos; 
        printf "Processed %s -> %s\n" base (Filename.basename output_file)
    done
9350b3f2052513d834d690943325a27c
echo x - process_fits_old.ml
sed 's/^X//' >process_fits_old.ml << '207caf59ce3899834933ce63ddb323fa'
open Printf

type wcs_params = {
    ra_2000: float;
    dec_2000: float;
    crpix1: float;
    crpix2: float;
    cd1_1: float;
    cd1_2: float;
    cd2_1: float;
    cd2_2: float;
}

type image_info = {
    width: int;
    height: int;
    wcs: wcs_params;
    filename: string;
}

(* Read FITS header and data *)
let find_header_end data =
    let hlen = ref 2880 in
    let hdrh = Hashtbl.create 257 in
    let rec scan_for_end pos =
        if pos > String.length data - 80 then failwith "header_end not found"
        else let key = String.sub data pos 80 in 
        (match String.trim (List.hd (String.split_on_char ' ' key)) with
        | "END" -> hlen := (((pos + 2880) / 2880) * 2880)
        | "COMMENT" -> scan_for_end (pos + 80)
        | oth -> Hashtbl.add hdrh oth (String.sub key (String.length oth) (80 - String.length oth)); 
                 scan_for_end (pos + 80))
    in scan_for_end 0;
    hdrh, String.sub data !hlen (String.length data - !hlen)

let read_image image =
    let rs = ref "" in
    let fd = open_in_bin image in
    (try rs := really_input_string fd (in_channel_length fd) with End_of_file -> ());
    close_in fd;
    !rs

(* Parse WCS parameters *)
let parse_wcs hdrh =
    let get_float key = 
        try Scanf.sscanf (Hashtbl.find hdrh key) " = %f /" (fun f->f)
        with Not_found -> 0.0
    in
    {
        ra_2000 = get_float "CRVAL1";
        dec_2000 = get_float "CRVAL2"; 
        crpix1 = get_float "CRPIX1";
        crpix2 = get_float "CRPIX2";
        cd1_1 = get_float "CD1_1";
        cd1_2 = get_float "CD1_2";
        cd2_1 = get_float "CD2_1";
        cd2_2 = get_float "CD2_2"
    }

(* Process FITS data *)
let read_image_data contents width height =
    let data = Array.make_matrix height width 0 in
    for y = 0 to height-1 do
        let row_offset = y * width * 2 in
        for x = 0 to width-1 do
            let off = row_offset + x * 2 in
            let value = (int_of_char contents.[off]) lor (int_of_char contents.[off+1] lsl 8) in
            data.(y).(x) <- value
        done
    done;
    data

(* Apply dark calibration *)
let subtract_dark data dark =
    let height = Array.length data in
    let width = Array.length data.(0) in
    let result = Array.make_matrix height width 0 in
    for y = 0 to height-1 do
        for x = 0 to width-1 do
            result.(y).(x) <- max 0 (data.(y).(x) - dark.(y).(x))
        done
    done;
    result

(* 2x2 binning with Bayer pattern awareness *)
let bin_2x2 data =
    let height = Array.length data in
    let width = Array.length data.(0) in
    let bin_height = height / 2 in
    let bin_width = width / 2 in
    let binned = Array.make_matrix bin_height bin_width (0,0,0) in
    for y = 0 to bin_height-1 do
        for x = 0 to bin_width-1 do
            let r = data.(y*2).(x*2) in         (* R pixel *)
            let g1 = data.(y*2).(x*2+1) in      (* G1 pixel *)
            let g2 = data.(y*2+1).(x*2) in      (* G2 pixel *)
            let b = data.(y*2+1).(x*2+1) in     (* B pixel *)
            let g = (g1 + g2) / 2 in            (* Average G *)
            binned.(y).(x) <- (r, g, b)
        done
    done;
    binned

(* Write binned data as FITS *)
let write_binned_fits filename data wcs =
    let oc = open_out_bin filename in
    (* Write FITS header *)
    let header = sprintf "SIMPLE  =                    T / file does conform to FITS standard             
BITPIX  =                  -32 / number of bits per data pixel                  
NAXIS   =                    3 / number of data axes                            
NAXIS1  =                 %4d / length of data axis 1                          
NAXIS2  =                 %4d / length of data axis 2                          
NAXIS3  =                    3 / length of data axis 3 (RGB)                   
EXTEND  =                    T / FITS dataset may contain extensions            
CTYPE1  = 'RA---TAN'           / Right ascension, gnomonic projection          
CTYPE2  = 'DEC--TAN'           / Declination, gnomonic projection              
CRVAL1  =              %f / Reference right ascension                
CRVAL2  =              %f / Reference declination                    
CRPIX1  =                %f / Reference pixel along axis 1            
CRPIX2  =                %f / Reference pixel along axis 2            
CD1_1   =          %f / CD matrix element                        
CD1_2   =          %f / CD matrix element                        
CD2_1   =          %f / CD matrix element                        
CD2_2   =          %f / CD matrix element                        
END                                                                             "
        (Array.length data.(0)) (Array.length data)
        wcs.ra_2000 wcs.dec_2000 
        (wcs.crpix1 /. 2.0) (wcs.crpix2 /. 2.0)  (* Adjust reference pixel for binning *)
        (wcs.cd1_1 *. 2.0) (wcs.cd1_2 *. 2.0)    (* Adjust pixel scale for binning *)
        (wcs.cd2_1 *. 2.0) (wcs.cd2_2 *. 2.0) in
    output_string oc header;
    (* Pad header to multiple of 2880 bytes *)
    let padding = String.make (2880 - (String.length header mod 2880)) ' ' in
    output_string oc padding;
    (* Write RGB data *)
    let buf = Bytes.create (Array.length data * Array.length data.(0) * 12) in
    let pos = ref 0 in
    for y = 0 to Array.length data - 1 do
        for x = 0 to Array.length data.(0) - 1 do
            let (r,g,b) = data.(y).(x) in
            Bytes.set_int32_be buf !pos (Int32.of_float (float_of_int r));
            Bytes.set_int32_be buf (!pos + 4) (Int32.of_float (float_of_int g));
            Bytes.set_int32_be buf (!pos + 8) (Int32.of_float (float_of_int b));
            pos := !pos + 12
        done
    done;
    output_bytes oc buf;
    close_out oc

(* Main processing *)
let process_image filename dark_data =
    let img = read_image filename in
    let hdrh, contents = find_header_end img in
    let get_int key = 
        try Scanf.sscanf (Hashtbl.find hdrh key) " = %d /" (fun i->i)
        with Not_found -> 0 in
    let width = get_int "NAXIS1" in
    let height = get_int "NAXIS2" in
    let wcs = parse_wcs hdrh in
    let data = read_image_data contents width height in
    let calibrated = subtract_dark data dark_data in
    let binned = bin_2x2 calibrated in
    { width = width/2; height = height/2; wcs = wcs; filename = filename }, binned

let () =
    if Array.length Sys.argv < 3 then
        (printf "Usage: %s <dark.fits> <output_dir> <files...>\n" Sys.argv.(0);
         exit 1);

    (* Read dark frame *)
    let dark_img = read_image Sys.argv.(1) in
    let dark_hdrh, dark_contents = find_header_end dark_img in
    let get_int key = 
        try Scanf.sscanf (Hashtbl.find dark_hdrh key) " = %d /" (fun i->i)
        with Not_found -> 
            (printf "Error: Could not find %s in dark frame header\n" key;
             exit 1) in
    let dark_width = get_int "NAXIS1" in
    let dark_height = get_int "NAXIS2" in
    if dark_width = 0 || dark_height = 0 then
        (printf "Error: Invalid dark frame dimensions: %dx%d\n" dark_width dark_height;
         exit 1);
    printf "Dark frame size: %dx%d\n" dark_width dark_height;
    let dark_data = read_image_data dark_contents dark_width dark_height in
    
    (* Process each input file *)
    let output_dir = Sys.argv.(2) in
    (try Unix.mkdir output_dir 0o755 with Unix.Unix_error(Unix.EEXIST, _, _) -> ());
    
    for i = 3 to Array.length Sys.argv - 1 do
        let input_file = Sys.argv.(i) in
        let base = Filename.basename input_file in
        let info, binned = process_image input_file dark_data in
        let output_file = Filename.concat output_dir 
            (Filename.chop_extension base ^ "_proc.fits") in
        write_binned_fits output_file binned info.wcs;
        printf "Processed %s -> %s\n" base (Filename.basename output_file)
    done
207caf59ce3899834933ce63ddb323fa
echo x - test_pattern.ml
sed 's/^X//' >test_pattern.ml << '2254550ab98f70157b68471d1879ab62'
open Printf

let width = 3072
let height = 2080

let header = String.concat "" ((List.map (fun itm -> sprintf "%-80s" itm)) [
"SIMPLE  =                    T / file does conform to FITS standard";
"BITPIX  =                   16 / number of bits per data pixel";
"NAXIS   =                    2 / number of data axes";
(sprintf "NAXIS1  =          %d / length of data axis 1" width);
(sprintf "NAXIS2  =          %d / length of data axis 2" height);
"BAYERPAT= 'RGGB    '           / Bayer pattern";
"EXPOSURE=                 1000 / Exposure time in ms";
"BZERO   =                32768 / offset data range to unsigned 16-bit";
"BSCALE  =                    1 / default scaling factor";
"END"
])

let generate_test_pattern filename =
 let oc = open_out_bin filename in
 
 (* Write padded header *)
 output_string oc (header ^ String.make (2880 - String.length header) ' ');
 
 (* Generate Bayer RGGB pattern *)
 let buf = Bytes.create (width * height * 2) in
 for y = 0 to height-1 do
   for x = 0 to width-1 do
     let value = 
       if y mod 2 = 0 then
         if x mod 2 = 0 then         (* R *)
           int_of_float (65535. *. float_of_int x /. float_of_int width)
         else                        (* G1 *)
           int_of_float (65535. *. float_of_int y /. float_of_int height)
       else
         if x mod 2 = 0 then         (* G2 *)
           int_of_float (65535. *. float_of_int (width-x) /. float_of_int width)
         else                        (* B *)
           int_of_float (65535. *. float_of_int (height-y) /. float_of_int height)
     in
     let pos = 2 * (y * width + x) in
     Bytes.set buf pos (char_of_int (value lsr 8));
     Bytes.set buf (pos+1) (char_of_int (value land 0xff))
   done
 done;
 
 output_bytes oc buf;
 output_string oc (String.make (2880 - (Bytes.length buf) mod 2880) ' ');
 close_out oc

let generate_dark_pattern filename =
 let oc = open_out_bin filename in
 
 (* Write padded header *)
 output_string oc (header ^ String.make (2880 - String.length header) ' ');
 
 (* Generate Bayer RGGB pattern *)
 let buf = Bytes.create (width * height * 2) in
 for y = 0 to height-1 do
   for x = 0 to width-1 do
     let value = 256 in
     let pos = 2 * (y * width + x) in
     Bytes.set buf pos (char_of_int (value lsr 8));
     Bytes.set buf (pos+1) (char_of_int (value land 0xff))
   done
 done;
 
 output_bytes oc buf;
 output_string oc (String.make (2880 - (Bytes.length buf) mod 2880) ' ');
 close_out oc

let () =
generate_test_pattern "test_pattern.fits";
generate_dark_pattern "dark_pattern.fits"
2254550ab98f70157b68471d1879ab62
echo x - types.ml
sed 's/^X//' >types.ml << 'f251e1d1485a5e66739f02b30dc879a9'
(* types.ml *)

(* World Coordinate System parameters *)
type wcs_params = {
    ra_2000: float;
    dec_2000: float;
    crpix1: float;
    crpix2: float;
    cd1_1: float;
    cd1_2: float;
    cd2_1: float;
    cd2_2: float;
}

(* FITS image information *)
type image_info = {
    width: int;
    height: int;
    wcs: wcs_params;
    filename: string;
}

(* Group of images for mosaic processing *)
type group_info = {
    id: string;
    files: image_info list;
}

(* Common error handling *)
let error msg =
    print_endline ("Error: " ^ msg);
    exit 1

(* Configuration constants *)
let verbose = ref false
let min_group_size = 12
let max_group_size = 30
let overlap_degrees = 0.1
let ra_bins = 7
let dec_bins = 5

(* Common FITS header parsing *)
let parse_int hdr key = 
  try let key' = Hashtbl.find hdr key in
  if !verbose then print_endline key';
  Scanf.sscanf key' " = %d" (fun i->i) with Not_found -> 0

let parse_float hdr key =
    let key' = Hashtbl.find hdr key in
    try Scanf.sscanf key' " = %f /" (fun f->f)
    with Not_found -> failwith key'

(* Group distance calculation *)
let distance_between_groups (_, files1) (_, files2) =
    let group_center files =
        let n = float_of_int (List.length files) in
        let sum_ra = List.fold_left (fun acc img -> acc +. img.wcs.ra_2000) 0.0 files in
        let sum_dec = List.fold_left (fun acc img -> acc +. img.wcs.dec_2000) 0.0 files in
        (sum_ra /. n, sum_dec /. n)
    in
    let c1_ra, c1_dec = group_center files1 in
    let c2_ra, c2_dec = group_center files2 in
    sqrt((c1_ra -. c2_ra) ** 2.0 +. (c1_dec -. c2_dec) ** 2.0)
f251e1d1485a5e66739f02b30dc879a9
exit

