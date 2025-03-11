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
  timeout: int;           (* Maximum time to spend on solving in seconds *)
  no_plots: bool;         (* Skip generating plots *)
  no_verify: bool;        (* Skip verification step *)
  overwrite: bool;        (* Overwrite existing output files *)
  use_sextractor: bool;   (* Use SExtractor for star extraction *)
  cpulimit: int option;   (* Limit CPU time in seconds *)
  odds_ratio: float;      (* Odds ratio threshold for solution *)
  depth: string option;   (* Depth of search (1-30, or specific list like "10,20,30") *)
  extension: string;      (* Extension for generated files *)
}

(* Default options for Stellina images - 0.57 arcsec/pixel plate scale *)
let default_options = {
  scale_low = 1.1;            (* ~10% lower than nominal Stellina scale *)
  scale_high = 2.6;           (* ~10% higher than nominal Stellina scale *)
  scale_units = "arcsecperpix";
  downsample = 2;              (* Speeds up solving, Stellina images are large enough *)
  timeout = 30;                (* 30 second timeout *)
  no_plots = true;             (* Skip generating plots to save time *)
  no_verify = false;           (* Keep verification step *)
  overwrite = true;            (* Overwrite existing output files *)
  use_sextractor = false;      (* SExtractor not needed for Stellina images *)
  cpulimit = Some 25;          (* Limit CPU time to 25 seconds *)
  odds_ratio = 1e9;            (* Higher odds ratio for more confidence *)
  depth = Some "10,20,30,40";  (* Progressive depths to try *)
  extension = "solved";        (* Extension for generated files *)
}

(* Build solve-field command with given options *)
let build_solve_command options filename output_dir =
  let base_name = Filename.basename filename |> Filename.remove_extension in
  let output_path = Filename.concat output_dir base_name in
  
  let cmd_parts = [
    "solve-field";
    sprintf "--scale-low %.2f" options.scale_low;
    sprintf "--scale-high %.2f" options.scale_high;
    sprintf "--scale-units %s" options.scale_units;
    sprintf "--downsample %d" options.downsample;
    sprintf "--odds-to-solve %.1e" options.odds_ratio;
    sprintf "--cpulimit %d" options.timeout;
    sprintf "--dir %s" output_dir;
(*    sprintf "--basename %s" base_name; *)
  ] in
  
  let cmd_parts = cmd_parts @ [
    if options.no_plots then "--no-plot" else "";
    if options.no_verify then "--no-verify" else "";
    if options.overwrite then "--overwrite" else "";
    if options.use_sextractor then "--use-sextractor" else "";
  ] in
  
  let cmd_parts = cmd_parts @ [
    (match options.cpulimit with
    | Some limit -> sprintf "--cpulimit %d" limit
    | None -> "");
    
    (match options.depth with
    | Some depth -> sprintf "--depth %s" depth
    | None -> "");
    
    sprintf "--new-fits %s.fits" output_path;
    filename
  ] in
  
  (* Filter out empty strings and join with spaces *)
  List.filter (fun s -> s <> "") cmd_parts
  |> String.concat " "

(* Run solve-field on a single file *)
let solve_field options filename output_dir =
  let start_time = Unix.gettimeofday() in
  
  (* Read mount coordinates from FITS header *)
  let hdrh = just_header filename in
  let mount_ra = 
    try parse_float hdrh "MOUNTRA" 
    with _ -> parse_float hdrh "OBJCTRA" 
  in
  let mount_dec = 
    try parse_float hdrh "MOUNTDEC=" 
    with _ -> parse_float hdrh "OBJCTDEC" 
  in
  
  (* Build and execute solve-field command *)
  let command = build_solve_command options filename output_dir in
  printf "Running: %s\n" command;
  flush stdout;
  
  let exit_code = Sys.command command in
  let end_time = Unix.gettimeofday() in
  let solve_time = end_time -. start_time in
  
  printf "solve-field finished with exit code %d in %.1f seconds\n" exit_code solve_time;
  
  (* After solve-field command *)
  printf "Checking for output files in: %s\n" output_dir;
  let dir_contents = Sys.readdir output_dir |> Array.to_list in
  List.iter (fun file -> 
    printf "  Found: %s\n" file
  ) dir_contents;

  (* Check if solving was successful *)
  let base_name = Filename.basename filename |> Filename.remove_extension in
  let wcs_file = Filename.concat output_dir (base_name ^ ".wcs") in
  
  if exit_code = 0 && Sys.file_exists wcs_file then
    (* Solving succeeded - read solved coordinates from WCS file *)
    let solved_fits = Filename.concat output_dir (base_name ^ "." ^ options.extension ^ ".fits") in
    
    if Sys.file_exists solved_fits then
      try
        let solved_hdrh = just_header solved_fits in
        let solved_ra = parse_float solved_hdrh "CRVAL1" in
        let solved_dec = parse_float solved_hdrh "CRVAL2" in
        
        (* Calculate difference *)
        let ra_error = solved_ra -. mount_ra in
        let dec_error = solved_dec -. mount_dec in
        let total_error = sqrt (ra_error *. ra_error +. dec_error *. dec_error) in
        
        printf "SUCCESS: RA=%.4f° (mount=%.4f°, error=%.4f°), Dec=%.4f° (mount=%.4f°, error=%.4f°)\n" 
          solved_ra mount_ra ra_error solved_dec mount_dec dec_error;
        printf "Total error: %.4f°\n" total_error;
        
        {
          success = true;
          filename = Filename.basename filename;
          mount_ra;
          mount_dec;
          solved_ra = Some solved_ra;
          solved_dec = Some solved_dec;
          ra_error = Some ra_error;
          dec_error = Some dec_error;
          total_error = Some total_error;
          solve_time;
        }
      with e ->
        printf "ERROR reading solved coordinates: %s\n" (Printexc.to_string e);
        {
          success = false;
          filename = Filename.basename filename;
          mount_ra;
          mount_dec;
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
        mount_ra;
        mount_dec;
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
      mount_ra;
      mount_dec;
      solved_ra = None;
      solved_dec = None;
      ra_error = None;
      dec_error = None;
      total_error = None;
      solve_time;
    }

(* Process a batch of FITS files *)
let verify_fits_batch files output_dir options =
  (* Create output directory if it doesn't exist *)
  if not (Sys.file_exists output_dir) then
    Unix.mkdir output_dir 0o755;
  
  (* Process each file *)
  let results = List.map (fun file ->
    printf "\nProcessing %s...\n" (Filename.basename file);
    solve_field options file output_dir
  ) files in
  
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

(* Main function for command-line operation *)
let main () =
  (* Parse command line arguments *)
  let input_dir = ref "" in
  let output_dir = ref "solved_verification" in
  let scale_low = ref default_options.scale_low in
  let scale_high = ref default_options.scale_high in
  let timeout = ref default_options.timeout in
  let use_all_files = ref false in
  
  let specs = [
    ("-i", Arg.Set_string input_dir, "Input directory containing FITS files");
    ("-o", Arg.Set_string output_dir, "Output directory for solving results");
    ("-scale-low", Arg.Set_float scale_low, "Lower bound of image scale (arcsec/pixel)");
    ("-scale-high", Arg.Set_float scale_high, "Upper bound of image scale (arcsec/pixel)");
    ("-timeout", Arg.Set_int timeout, "Timeout in seconds for each solve");
    ("-all", Arg.Set use_all_files, "Process all FITS files (not just those with MOUNTRA/DEC)");
  ] in
  
  let usage = "Usage: verify_plate_solving -i input_dir [-o output_dir] [-scale-low val] [-scale-high val] [-timeout sec] [-all]" in
  
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
    timeout = !timeout;
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

(* Run main function if executed directly *)
let () = main ()
