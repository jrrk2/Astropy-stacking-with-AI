(* Add to types.ml *)
open Printf
type sort_options = {
  mutable sort_darks: bool;
  mutable dry_run: bool;
  mutable base_dir: string;
}

(* Add to dark_temp_analysis.ml *)

(* Create directory if it doesn't exist *)
let ensure_directory dir =
  if not (Sys.file_exists dir) then
    Unix.mkdir dir 0o755
  else if not (Sys.is_directory dir) then
    failwith (sprintf "%s exists but is not a directory" dir)

(* Sort darks into temperature binned directories *)
let sort_dark_frames files base_dir dry_run =
  printf "Sorting %d dark frames into temperature bins...\n" (Array.length files);
  
  (* First scan temperatures quickly *)
  let temp_infos = Array.to_list files 
    |> List.filter_map scan_fits_temperature 
    |> Array.of_list in
    
  if Array.length temp_infos = 0 then
    failwith "No valid temperature data found in files"
  else begin
    (* Create base directory *)
    if not dry_run then
      ensure_directory base_dir;
      
    (* Process each file *)
    Array.iter (fun ti -> 
      (* Convert to Kelvin and round to nearest degree *)
      let temp_kelvin = ti.temp +. 273.15 in
      let temp_bin = int_of_float (floor (temp_kelvin +. 0.5)) in
      let dir_name = sprintf "%s/temp_%d" base_dir temp_bin in
      let new_path = sprintf "%s/%s" dir_name (Filename.basename ti.filename) in
      
      if dry_run then
        printf "Would move %s -> %s\n" ti.filename new_path
      else begin
        ensure_directory dir_name;
        printf "Moving %s -> %s\n" ti.filename new_path;
        Sys.rename ti.filename new_path
      end
    ) temp_infos;
    
    printf "\nSorted %d files into temperature-binned directories%s\n" 
      (Array.length temp_infos)
      (if dry_run then " (dry run)" else "");
    printf "Note: Directories are named by Kelvin temperature (0°C = 273K)\n"
  end

(* Update parse_args function *)
let parse_args () =
  let flags = {
    show_temp_plot = false;
    show_dist_plot = false;
    show_stats = false;
    show_coeff = false;
    temp_range = None
  } in
  let sort_opts = {
    sort_darks = false;
    dry_run = false;
    base_dir = "dark_temps"
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
    ("-sort", Arg.Unit (fun () -> sort_opts.sort_darks <- true),
     "Sort darks into temperature-binned directories");
    ("-dry-run", Arg.Unit (fun () -> sort_opts.dry_run <- true),
     "Show what would be done without actually moving files");
    ("-outdir", Arg.String (fun dir -> sort_opts.base_dir <- dir),
     "Base directory for sorted files (default: dark_temps)");
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
  (flags, sort_opts, Array.of_list (List.rev !files))

(* Update main function *)
let () =
  let flags, sort_opts, files = parse_args () in
  if Array.length files = 0 then
    failwith "No input files specified"
  else if sort_opts.sort_darks then
    sort_dark_frames files sort_opts.base_dir sort_opts.dry_run
  else if flags.show_temp_plot && not (flags.show_dist_plot || flags.show_stats || flags.show_coeff) then
    (* Fast path: only temperature scan needed *)
    let temp_infos = Array.to_list files 
      |> List.filter_map scan_fits_temperature 
      |> Array.of_list in
    if Array.length temp_infos > 0 then
      let stats = Array.map (fun (ti:temp_info) -> 
        {filename = ti.filename; 
         temperature = ti.temp;
         mean_level = 0.0;
         std_dev = 0.0;
         timestamp = 0.0;
         hdrh = ti.qhdr})
        temp_infos in
      plot_temp_vs_time stats
    else
      failwith "No valid temperature data found in files"
  else
    (* Full analysis needed *)
    analyze_dark_frames files flags