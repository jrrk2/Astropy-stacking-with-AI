(* plate_solve_verification.ml - Refactored as a library *)

open Types
open Fits
open Printf

(* Plate-solving verification for Stellina images *)
type solve_result = {
  success: bool;
  filename: string;
  mount_ra: float;
  mount_dec: float;
  solved_ra: float option;
  solved_dec: float option;
  ra_error: float option;
  dec_error: float option;
  total_error: float option;
  solve_time: float;
}

type solve_options = {
  scale_low: float;       (* Lower bound of image scale estimate in arcsec/pixel *)
  scale_high: float;      (* Upper bound of image scale estimate in arcsec/pixel *)
  scale_units: string;    (* Units for scale (usually "arcsecperpix") *)
  downsample: int;        (* Downsample factor - speeds up solving for large images *)
  cpulimit: int;          (* Maximum time to spend on solving in seconds *)
  no_plots: bool;         (* Skip generating plots *)
  no_verify: bool;        (* Skip verification step *)
  overwrite: bool;        (* Overwrite existing output files *)
  use_sextractor: bool;   (* Use SExtractor for star extraction *)
  odds_ratio: float;      (* Odds ratio threshold for solution *)
  depth: string option;   (* Depth of search (1-30, or specific list like "10,20,30") *)
  extension: string;      (* Extension for generated files *)
  verbose: bool;          (* Verbose file reporting *)
  radius: float;          (* search radius *)
  mutable mount_ra: float;        (* Estimated RA from telescope pointing *)
  mutable mount_dec: float;       (* Estimated DEC from telescope pointing *)
}

(* Default options for Stellina images *)
let default_options = {
  scale_low = 1.1;            (* Lower bound for Stellina RGB images (original ~1.2 arcsec/px) *)
  scale_high = 2.6;           (* Upper bound for Stellina RGB images (doubled to ~2.4 arcsec/px) *)
  scale_units = "arcsecperpix";
  downsample = 1;              (* No downsampling needed for RGB images (already 2x from debayering) *)
  cpulimit = 30;               (* 30 second cpulimit *)
  no_plots = true;             (* Skip generating plots to save time *)
  no_verify = false;           (* Keep verification step *)
  overwrite = true;            (* Overwrite existing output files *)
  use_sextractor = false;      (* SExtractor not needed for Stellina images *)
  odds_ratio = 1e9;            (* Higher odds ratio for more confidence *)
  depth = Some "10,20,30,40";  (* Progressive depths to try *)
  extension = "solved";        (* Extension for generated files *)
  verbose = false;             (* not verbose *)
  radius = 1.0;                (* search radius *)
  mount_ra = 0.0;              (* placeholders *)
  mount_dec = 0.0;
}

(* Build solve-field command with given options *)
let build_solve_command options filename output_dir =
  let base_name = Filename.basename filename |> Filename.remove_extension in
  let output_path = Filename.concat output_dir base_name in
  
  (* Check if this is an RGB file - if so, don't downsample *)
  let is_rgb = 
    try
      let hdrh = Fits.just_header filename in
      let naxis = parse_int hdrh "NAXIS" in
      naxis = 3  (* RGB files have NAXIS=3 *)
    with _ -> false  (* Default to false if we can't determine *)
  in
  
  let effective_downsample = if is_rgb then 1 else options.downsample in
  
  let cmd_parts = [
    "solve-field";
    sprintf "--ra %.4f" options.mount_ra;
    sprintf "--dec %.4f" options.mount_dec;
    sprintf "--radius %.4f" options.radius;
    sprintf "--scale-low %.2f" options.scale_low;
    sprintf "--scale-high %.2f" options.scale_high;
    sprintf "--scale-units %s" options.scale_units;
    sprintf "--downsample %d" effective_downsample;
    sprintf "--odds-to-solve %.1e" options.odds_ratio;
    sprintf "--dir %s" output_dir;
  ] in
  
  let cmd_parts = cmd_parts @ [
    if options.no_plots then "--no-plot" else "";
    if options.no_verify then "--no-verify" else "";
    if options.overwrite then "--overwrite" else "";
    if options.use_sextractor then "--use-sextractor" else "";
  ] in
  
  let cmd_parts = cmd_parts @ [
    
    (match options.depth with
    | Some depth -> sprintf "--depth %s" depth
    | None -> "");
    
    sprintf "--new-fits %s.fits" output_path;
    filename
  ] in
  
  (* Filter out empty strings and join with spaces *)
  List.filter (fun s -> s <> "") cmd_parts
  |> String.concat " "

(* A more explicit approach using direct process management *)

(* Create a temporary file for process output *)
let create_temp_file prefix suffix =
  let temp_dir = Filename.get_temp_dir_name () in
  let rec try_name counter =
    let name = Printf.sprintf "%s/%s_%d%s" temp_dir prefix counter suffix in
    if Sys.file_exists name then
      try_name (counter + 1)
    else
      name
  in
  try_name 0

(* Replace run_command_async with this improved version *)
let run_command_async command =
  let output_file = create_temp_file "solve_field" ".log" in
  
  (* Create proper redirection *)
  let stdout_fd = Unix.openfile output_file [Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC] 0o644 in
  let stderr_fd = stdout_fd in  (* Redirect stderr to the same file *)
  
  (* Split the command into program and arguments *)
  let args = Str.split (Str.regexp "[ \t]+") command in
  let prog = List.hd args in
  let args_array = Array.of_list args in
  
  (* Create the process with proper redirection *)
  let pid = Unix.create_process prog args_array Unix.stdin stdout_fd stderr_fd in
  
  (* Close file descriptors in the parent process *)
  Unix.close stdout_fd;
  
  (pid, output_file)

(* Replace the wait_for_process function with this improved version *)
let wait_for_process pid =
  try
    let (_, status) = Unix.waitpid [] pid in
    match status with
    | Unix.WEXITED code -> code
    | Unix.WSIGNALED _ -> -1  (* Process terminated by signal *)
    | Unix.WSTOPPED _ -> -2   (* Process stopped by signal *)
  with Unix.Unix_error (Unix.ECHILD, _, _) ->
    (* Process no longer exists *)
    Printf.printf "Warning: Process %d no longer exists\n" pid;
    -1

(* Run solve-field using direct process management *)
let solve_field_with_process options filename output_dir =
  let start_time = Unix.gettimeofday () in
  
  (* Read mount coordinates from FITS header *)
  let hdrh = just_header filename in
  options.mount_ra <-
    (try parse_float hdrh "MOUNTRA" 
    with _ -> parse_float hdrh "OBJCTRA");

  options.mount_dec <-
    (try parse_float hdrh "MOUNTDEC=" 
    with _ -> parse_float hdrh "OBJCTDEC"); 
  
  (* Build solve-field command *)
  let command = build_solve_command options filename output_dir in
  Printf.printf "Starting: %s\n" (Filename.basename filename);
  Printf.printf "Command: %s\n" command;
  flush stdout;
  
  (* Run the command as a separate process *)
  let (pid, output_file) = run_command_async command in
  
  (* This is now a non-blocking return point - the process is running in the background *)
  (pid, start_time, output_file, filename, options, output_dir)

(* Collect results from a finished solve-field process *)
let collect_solve_result (pid, start_time, output_file, filename, options, output_dir) =
  (* Wait for process to finish *)
  let exit_code = wait_for_process pid in
  let end_time = Unix.gettimeofday() in
  let solve_time = end_time -. start_time in
  
  (* Read process output *)
  let output = 
    try
      let ic = open_in output_file in
      let content = really_input_string ic (in_channel_length ic) in
      close_in ic;
      Some content
    with _ -> None
  in
  
  if options.verbose && output <> None then
    Printf.printf "Process output: %s\n" (Option.get output);
  
  (* Delete temporary output file *)
  (try Sys.remove output_file with _ -> ());
  
  Printf.printf "Finished: %s (exit code %d) in %.1f seconds\n" 
    (Filename.basename filename) exit_code solve_time;
  
  (* Check if solving was successful *)
  let base_name = Filename.basename filename |> Filename.remove_extension in
  let wcs_file = Filename.concat output_dir (base_name ^ ".wcs") in
  let solved_flag = Filename.concat output_dir (base_name ^ ".solved") in
  
  if exit_code = 0 && (Sys.file_exists wcs_file || Sys.file_exists solved_flag) then
    (* Solving succeeded - read solved coordinates from solved FITS file *)
    let solved_fits = Filename.concat output_dir (base_name ^ ".fits") in
    
    if Sys.file_exists solved_fits then
      try
        let solved_hdrh = just_header solved_fits in
        let solved_ra = parse_float solved_hdrh "CRVAL1" in
        let solved_dec = parse_float solved_hdrh "CRVAL2" in
        
        (* Calculate difference *)
        let ra_error = solved_ra -. options.mount_ra in
        let dec_error = solved_dec -. options.mount_dec in
        let total_error = sqrt (ra_error *. ra_error +. dec_error *. dec_error) in
        
        Printf.printf "SUCCESS: %s RA=%.4f° (mount=%.4f°, error=%.4f°), Dec=%.4f° (mount=%.4f°, error=%.4f°)\n"
          (Filename.basename filename) solved_ra options.mount_ra ra_error solved_dec options.mount_dec dec_error;
        Printf.printf "Total error: %.4f°\n" total_error;
        
        {
          success = true;
          filename = Filename.basename filename;
          mount_ra = options.mount_ra;
          mount_dec = options.mount_dec;
          solved_ra = Some solved_ra;
          solved_dec = Some solved_dec;
          ra_error = Some ra_error;
          dec_error = Some dec_error;
          total_error = Some total_error;
          solve_time;
        }
      with e ->
        Printf.printf "ERROR reading solved coordinates: %s\n" (Printexc.to_string e);
        {
          success = false;
          filename = Filename.basename filename;
          mount_ra = options.mount_ra;
          mount_dec = options.mount_dec;
          solved_ra = None;
          solved_dec = None;
          ra_error = None;
          dec_error = None;
          total_error = None;
          solve_time;
        }
    else
      (* WCS file exists but no solved FITS *)
      {
        success = false;
        filename = Filename.basename filename;
        mount_ra = options.mount_ra;
        mount_dec = options.mount_dec;
        solved_ra = None;
        solved_dec = None;
        ra_error = None;
        dec_error = None;
        total_error = None;
        solve_time;
      }
  else
    (* Solving failed *)
    {
      success = false;
      filename = Filename.basename filename;
      mount_ra = options.mount_ra;
      mount_dec = options.mount_dec;
      solved_ra = None;
      solved_dec = None;
      ra_error = None;
      dec_error = None;
      total_error = None;
      solve_time;
    }

(* Replace verify_fits_batch_with_processes with this fixed version *)
let verify_fits_batch_with_processes ?(worker_count=0) files output_dir options =
  (* Create output directory if it doesn't exist *)
  if not (Sys.file_exists output_dir) then
    Unix.mkdir output_dir 0o755;
  
  (* Determine number of workers *)
  let cpu_count = 8 in
  let max_workers = 
    if worker_count > 0 then worker_count
    else max 1 (cpu_count * 3 / 4) (* Default to 75% of CPU cores *)
  in
  
  Printf.printf "Running with %d parallel workers (detected %d CPUs)\n" 
    max_workers cpu_count;
  
  (* Set to track active processes *)
  let active_processes = Hashtbl.create max_workers in
  let results = ref [] in
  let remaining_files = ref files in
  
  (* Start the initial batch of processes *)
  for _ = 1 to min max_workers (List.length !remaining_files) do
    match !remaining_files with
    | file :: rest ->
        remaining_files := rest;
        let process_info = solve_field_with_process options file output_dir in
        let pid = match process_info with (pid, _, _, _, _, _) -> pid in
        Hashtbl.add active_processes pid process_info;
        (* Add a small delay between starting processes to prevent race conditions *)
        Unix.sleepf 0.1;
    | [] -> ()
  done;  


(* Update process_loop in verify_fits_batch_with_processes to use blocking waitpid for one process at a time *)
let rec process_loop active_processes remaining_files results max_workers =
  if Hashtbl.length active_processes = 0 && List.length !remaining_files = 0 then
    (* All done *)
    !results
  else begin
    (* Check for any completed process with blocking waitpid, but only if we have active processes *)
    if Hashtbl.length active_processes > 0 then begin
      (* Wait for any child process to complete *)
      try
        let (pid, status) = Unix.wait () in
        Printf.printf "Process %d completed with status: %s\n" pid
          (match status with
           | Unix.WEXITED code -> sprintf "exit code %d" code
           | Unix.WSIGNALED sig_num -> sprintf "killed by signal %d" sig_num
           | Unix.WSTOPPED sig_num -> sprintf "stopped by signal %d" sig_num);
        
        (* If this process is in our table, collect its result *)
        if Hashtbl.mem active_processes pid then begin
          let process_info = Hashtbl.find active_processes pid in
          let result = collect_solve_result process_info in
          results := result :: !results;
          Hashtbl.remove active_processes pid;
        end else
          Printf.printf "Warning: Completed process %d was not in our tracking table\n" pid;
      with Unix.Unix_error (Unix.ECHILD, _, _) ->
        (* No children to wait for, which shouldn't happen if we have active processes *)
        Printf.printf "Warning: waitpid returned ECHILD when we expected active processes\n";
        Unix.sleepf 0.5;  (* Wait a bit before trying again *)
    end;
    
    (* Start new processes if slots available *)
    while Hashtbl.length active_processes < max_workers && List.length !remaining_files > 0 do
      match !remaining_files with
      | file :: rest ->
          remaining_files := rest;
          let process_info = solve_field_with_process options file output_dir in
          let pid = match process_info with (pid, _, _, _, _, _) -> pid in
          Hashtbl.add active_processes pid process_info;
          (* Add a small delay between starting processes to prevent race conditions *)
          Unix.sleepf 0.1;
      | [] -> ()
    done;
    
    (* Print status update *)
    Printf.printf "Status: %d active processes, %d files remaining, %d completed\n"
      (Hashtbl.length active_processes) (List.length !remaining_files) (List.length !results);
    
    (* Continue processing *)
    process_loop active_processes remaining_files results max_workers
  end in
  (* Run the processing loop with our updated implementation *)
  let all_results = process_loop active_processes remaining_files results max_workers in
  
  (* Generate summary report *)
  let csv_path = Filename.concat output_dir "verification_results.csv" in
  let html_path = Filename.concat output_dir "verification_report.html" in
  
  (* Write CSV results *)
  let csv = open_out csv_path in
  fprintf csv "Filename,Success,MountRA,MountDec,SolvedRA,SolvedDec,RAError,DecError,TotalError,SolveTime\n";
  
  List.iter (fun r ->
    fprintf csv "%s,%b,%.6f,%.6f,%s,%s,%s,%s,%s,%.1f\n"
      r.filename
      r.success
      r.mount_ra
      r.mount_dec
      (match r.solved_ra with Some v -> sprintf "%.6f" v | None -> "")
      (match r.solved_dec with Some v -> sprintf "%.6f" v | None -> "")
      (match r.ra_error with Some v -> sprintf "%.6f" v | None -> "")
      (match r.dec_error with Some v -> sprintf "%.6f" v | None -> "")
      (match r.total_error with Some v -> sprintf "%.6f" v | None -> "")
      r.solve_time
  ) !results;
  
  close_out csv;
  printf "Results saved to %s\n" csv_path;
  
  (* Calculate statistics *)
  let successful = List.filter (fun r -> r.success) !results in
  let success_count = List.length successful in
  let total_count = List.length !results in
  let success_rate = if total_count > 0 then 
    float_of_int success_count /. float_of_int total_count *. 100.0 
  else 0.0 in
  
  (* Calculate average errors *)
  let avg_ra_error, avg_dec_error, avg_total_error =
    if success_count > 0 then
      let sum_ra = List.fold_left (fun acc r -> 
        acc +. match r.ra_error with Some e -> e | None -> 0.0
      ) 0.0 successful in
      let sum_dec = List.fold_left (fun acc r -> 
        acc +. match r.dec_error with Some e -> e | None -> 0.0
      ) 0.0 successful in
      let sum_total = List.fold_left (fun acc r -> 
        acc +. match r.total_error with Some e -> e | None -> 0.0
      ) 0.0 successful in
      
      sum_ra /. float_of_int success_count,
      sum_dec /. float_of_int success_count,
      sum_total /. float_of_int success_count
    else
      0.0, 0.0, 0.0
  in
  
  (* Write HTML report *)
  let html = open_out html_path in
  
  fprintf html "<!DOCTYPE html>\n<html>\n<head>\n";
  fprintf html "<title>Stellina Plate-Solving Verification Report</title>\n";
  fprintf html "<style>\n";
  fprintf html "body { font-family: Arial, sans-serif; margin: 20px; }\n";
  fprintf html "h1, h2 { color: #333; }\n";
  fprintf html "table { border-collapse: collapse; width: 100%%; }\n";
  fprintf html "th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }\n";
  fprintf html "tr:hover { background-color: #f5f5f5; }\n";
  fprintf html "th { background-color: #4CAF50; color: white; }\n";
  fprintf html ".success { color: green; }\n";
  fprintf html ".failed { color: red; }\n";
  fprintf html ".summary { margin: 20px 0; }\n";
  fprintf html ".summary-item { margin: 10px 0; }\n";
  fprintf html "</style>\n</head>\n<body>\n";
  
  fprintf html "<h1>Stellina Plate-Solving Verification Report</h1>\n";
  
  (* Summary statistics *)
  fprintf html "<div class='summary'>\n";
  fprintf html "<h2>Summary</h2>\n";
  fprintf html "<div class='summary-item'>Total images: <strong>%d</strong></div>\n" total_count;
  fprintf html "<div class='summary-item'>Successfully solved: <strong>%d</strong> (%.1f%%)</div>\n" 
    success_count success_rate;
  fprintf html "<div class='summary-item'>Average error: <strong>%.4f°</strong> (%.1f arcmin)</div>\n" 
    avg_total_error (avg_total_error *. 60.0);
  fprintf html "<div class='summary-item'>Average RA error: <strong>%.4f°</strong></div>\n" avg_ra_error;
  fprintf html "<div class='summary-item'>Average Dec error: <strong>%.4f°</strong></div>\n" avg_dec_error;
  fprintf html "<div class='summary-item'>Parallel processing: <strong>%d workers</strong></div>\n" max_workers;
  fprintf html "</div>\n";
  
  (* Results table *)
  fprintf html "<h2>Results</h2>\n";
  fprintf html "<table>\n";
  fprintf html "<tr><th>Filename</th><th>Status</th><th>Mount RA</th><th>Mount Dec</th>";
  fprintf html "<th>Solved RA</th><th>Solved Dec</th><th>RA Error</th><th>Dec Error</th>";
  fprintf html "<th>Total Error</th><th>Solve Time</th></tr>\n";
  
  List.iter (fun r ->
    fprintf html "<tr>\n";
    fprintf html "  <td>%s</td>\n" r.filename;
    fprintf html "  <td class='%s'>%s</td>\n" 
      (if r.success then "success" else "failed")
      (if r.success then "Success" else "Failed");
    fprintf html "  <td>%.4f°</td>\n" r.mount_ra;
    fprintf html "  <td>%.4f°</td>\n" r.mount_dec;
    fprintf html "  <td>%s</td>\n" (match r.solved_ra with Some v -> sprintf "%.4f°" v | None -> "-");
    fprintf html "  <td>%s</td>\n" (match r.solved_dec with Some v -> sprintf "%.4f°" v | None -> "-");
    fprintf html "  <td>%s</td>\n" (match r.ra_error with Some v -> sprintf "%.4f°" v | None -> "-");
    fprintf html "  <td>%s</td>\n" (match r.dec_error with Some v -> sprintf "%.4f°" v | None -> "-");
    fprintf html "  <td>%s</td>\n" (match r.total_error with 
                                   | Some v when v > 1.0 -> sprintf "<strong>%.4f°</strong>" v
                                   | Some v -> sprintf "%.4f°" v 
                                   | None -> "-");
    fprintf html "  <td>%.1f sec</td>\n" r.solve_time;
    fprintf html "</tr>\n";
  ) !results;
  
  fprintf html "</table>\n";
  fprintf html "</body>\n</html>\n";
  
  close_out html;
  printf "HTML report saved to %s\n" html_path;
  
  (* Return statistics *)
  (success_count, total_count, avg_total_error)
