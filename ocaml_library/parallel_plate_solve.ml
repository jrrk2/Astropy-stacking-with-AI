(* Add this to the top of plate_solve_verification.ml *)
open Lwt.Infix

(* A simpler worker pool implementation using Lwt_list *)
let parallel_map_limited ~limit f items =
  (* Initialize shared state *)
  let remaining_items = ref items in
  let results = ref [] in
  let queue_mutex = Lwt_mutex.create () in
  let results_mutex = Lwt_mutex.create () in
  
  let rec worker () =
    let res = ref None in
    let task = Lwt_mutex.with_lock queue_mutex (fun () ->
      match !remaining_items with
      | [] -> Lwt.return_false
      | item :: rest ->
        remaining_items := rest;
        res := Some item;
        Lwt.return_true
    ) in
    
    task >>= function
    | false -> Lwt.return_unit  (* No more work *)
    | true ->
      begin match !res with
      | None -> Lwt.return_unit  (* Should never happen *)
      | Some item ->
        Lwt.catch
          (fun () -> 
            f item >>= fun result ->
            Lwt_mutex.with_lock results_mutex (fun () ->
              results := result :: !results;
              Lwt.return_unit
            )
          )
          (fun exn ->
            Printf.eprintf "Task failed with exception: %s\n" (Printexc.to_string exn);
            Lwt.return_unit
          ) >>= fun () ->
        worker ()  (* Process next item *)
      end
  in
  
  (* Create worker threads *)
  let workers = List.init (min limit (List.length items)) (fun _ -> worker ()) in
  
  (* Wait for all workers to finish *)
  Lwt.join workers >>= fun () ->
  Lwt.return (List.rev !results)

(* Run a single solve-field job with proper error handling *)
let solve_field_job filename output_dir options =
  try
    let result = solve_field options filename output_dir in
    Lwt.return result
  with exn ->
    Printf.eprintf "Error processing %s: %s\n" filename (Printexc.to_string exn);
    Lwt.return {
      success = false;
      filename = Filename.basename filename;
      mount_ra = 0.0;
      mount_dec = 0.0;
      solved_ra = None;
      solved_dec = None;
      ra_error = None;
      dec_error = None;
      total_error = None;
      solve_time = 0.0
    }

(* Replace verify_fits_batch with this parallel version *)
let verify_fits_batch files output_dir options =
  (* Create output directory if it doesn't exist *)
  if not (Sys.file_exists output_dir) then
    Unix.mkdir output_dir 0o755;
  
  (* Determine number of CPU cores for parallel processing *)
  let cpu_count = 
    try int_of_string (Sys.getenv "NUMBER_OF_PROCESSORS")
    with _ -> 
      try
        let ic = Unix.open_process_in "nproc" in
        let cores = input_line ic in
        let _ = Unix.close_process_in ic in
        int_of_string cores
      with _ -> 
        try
          let ic = Unix.open_process_in "sysctl -n hw.ncpu" in
          let cores = input_line ic in
          let _ = Unix.close_process_in ic in
          int_of_string cores
        with _ -> 4  (* Default to 4 cores if detection fails *)
  in
  
  (* Use 75% of available cores, minimum 1 *)
  let worker_count = max 1 (cpu_count * 3 / 4) in
  Printf.printf "Running with %d parallel workers (detected %d CPUs)\n" worker_count cpu_count;
  
  (* Process files in parallel with limited concurrency *)
  let solve_job file = solve_field_job file output_dir options in
  let results = Lwt_main.run (parallel_map_limited ~limit:worker_count solve_job files) in
  
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
  ) results;
  
  close_out csv;
  printf "Results saved to %s\n" csv_path;
  
  (* Calculate statistics *)
  let successful = List.filter (fun r -> r.success) results in
  let success_count = List.length successful in
  let total_count = List.length results in
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
  fprintf html "<div class='summary-item'>Parallel processing: <strong>%d workers</strong></div>\n" worker_count;
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
  ) results;
  
  fprintf html "</table>\n";
  fprintf html "</body>\n</html>\n";
  
  close_out html;
  printf "HTML report saved to %s\n" html_path;
  
  (* Return statistics *)
  (success_count, total_count, avg_total_error)

(* Update the main function to add a new parameter for worker count *)
let main () =
  (* Parse command line arguments *)
  let input_dir = ref "" in
  let output_dir = ref "solved_verification" in
  let scale_low = ref default_options.scale_low in
  let scale_high = ref default_options.scale_high in
  let use_all_files = ref false in
  let cpulimit = ref 30 in
  let workers = ref 0 in  (* 0 means auto-detect *)

  let specs = [
    ("-i", Arg.Set_string input_dir, "Input directory containing FITS files");
    ("-o", Arg.Set_string output_dir, "Output directory for solving results");
    ("-scale-low", Arg.Set_float scale_low, "Lower bound of image scale (arcsec/pixel)");
    ("-scale-high", Arg.Set_float scale_high, "Upper bound of image scale (arcsec/pixel)");
    ("-cpulimit", Arg.Set_int cpulimit, "CPU limit in seconds for each solve");
    ("-workers", Arg.Set_int workers, "Number of parallel workers (default: auto-detect)");
    ("-all", Arg.Set use_all_files, "Process all FITS files (not just those with MOUNTRA/DEC)");
  ] in
  
  let usage = "Usage: verify_plate_solving -i input_dir [-o output_dir] [-scale-low val] [-scale-high val] [-cpulimit sec] [-workers n] [-all]" in
  
  Arg.parse specs (fun _ -> ()) usage;
  
  if !input_dir = "" then begin
    Printf.printf "Error: Input directory must be specified with -i\n";
    Arg.usage specs usage;
    exit 1
  end;
  
  (* Find all FITS files in the input directory *)
  let files = 
    try
      Sys.readdir !input_dir
      |> Array.to_list
      |> List.filter (fun f -> 
          Filename.check_suffix f ".fits" || 
          Filename.check_suffix f ".fit" ||
          Filename.check_suffix f ".FITS" ||
          Filename.check_suffix f ".FIT")
      |> List.map (fun f -> Filename.concat !input_dir f)
    with _ -> begin
      Printf.printf "Error reading directory %s\n" !input_dir;
      exit 1
    end
  in
  
  if List.length files = 0 then begin
    Printf.printf "No FITS files found in %s\n" !input_dir;
    exit 1
  end;
  
  (* Filter files to those with mount coordinates, unless -all is specified *)
  let files_to_process =
    if !use_all_files then files
    else
      List.filter (fun file ->
        try
          let hdrh = just_header file in
          let _ = 
            try parse_float hdrh "MOUNTRA" 
            with _ -> parse_float hdrh "OBJCTRA" 
          in
          let _ = 
            try parse_float hdrh "MOUNTDEC=" 
            with _ -> parse_float hdrh "OBJCTDEC" 
          in
          true
        with _ -> false
      ) files
  in
  
  Printf.printf "Found %d FITS files with mount coordinates (out of %d total)\n"
    (List.length files_to_process) (List.length files);
  
  if List.length files_to_process = 0 then begin
    Printf.printf "No files with mount coordinates found. Use -all to process all files.\n";
    exit 1
  end;
  
  (* Create custom options from command line args *)
  let options = { default_options with
    scale_low = !scale_low;
    scale_high = !scale_high;
    cpulimit = !cpulimit;
  } in
  
  (* Process files *)
  let (success_count, total_count, avg_error) = 
    verify_fits_batch files_to_process !output_dir options 
  in
  
  Printf.printf "\nVerification complete\n";
  Printf.printf "  %d/%d images successfully solved (%.1f%%)\n" 
    success_count total_count 
    (float_of_int success_count /. float_of_int total_count *. 100.0);
  Printf.printf "  Average error: %.4f° (%.1f arcmin)\n" 
    avg_error (avg_error *. 60.0);
  Printf.printf "  Results saved to %s\n" !output_dir;
  
  exit (if success_count = 0 then 1 else 0)
