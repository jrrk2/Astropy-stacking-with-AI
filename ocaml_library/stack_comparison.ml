(* stack_comparison.ml - Compare star alignment vs plate solving for image stacking *)

open Types
open Printf
open Fits
open Fits_utils
open Plate_solve_verification
open Astrometric_alignment
open Triangle_alignment

(* Log levels *)
type log_level = Debug | Info | Warning | Error

(* Structure to hold alignment results for comparison *)
type alignment_result = {
  method_name: string;
  filename: string;
  success: bool;
  reference_stars: star_point list;
  detected_stars: star_point list;
  matched_pairs: (star_point * star_point) list;
  transform: alignment_parameters;
  error_stats: float * float * float;  (* mean, max, stddev *)
  runtime: float;
}

(* Structure to hold comparison results *)
type comparison_result = {
  filename: string;
  plate_solve_success: bool;
  star_align_success: bool;
  plate_solve_stars: int;
  star_align_stars: int;
  plate_solve_error: float;
  star_align_error: float;
  runtime_ratio: float;
}

let log level msg =
  let level_str = match level with
    | Debug -> "DEBUG"
    | Info -> "INFO"
    | Warning -> "WARNING"
    | Error -> "ERROR"
  in
  printf "[%s] %s\n" level_str msg;
  flush stdout

(* Function to detect stars in an image using the star detection module *)
let detect_stars filename threshold =
  let start_time = Unix.gettimeofday() in
  log Info (sprintf "Detecting stars in %s with threshold %.1f" 
    (Filename.basename filename) threshold);
  
  try
    (* Read the FITS file *)
    let (hdrh, data) = read_fits_large filename in
    let stats = compute_image_stats data in
    
    (* Detect stars *)
    let stars = detect_stars_in_image data stats threshold in
    
    let end_time = Unix.gettimeofday() in
    log Info (sprintf "Detected %d stars in %.2f seconds" 
      (List.length stars) (end_time -. start_time));
    Some (hdrh, stars, end_time -. start_time)
  with e ->
    log Error (sprintf "Error detecting stars: %s" (Printexc.to_string e));
    None

let align_with_stars reference_file target_file _threshold _max_stars =
  let start_time = Unix.gettimeofday() in
  log Info (sprintf "Aligning %s to %s using RGB star pattern matching" 
    (Filename.basename target_file) (Filename.basename reference_file));
  
  match Rgb_star_alignment.align_rgb_images reference_file target_file ~debug:true () with
  | Some (transform, matches, error, align_time) ->
      (* Convert matches to the format expected by the comparison framework *)
      let matched_pairs = List.map (fun m -> 
        ({ x = m.ref_star.x; y = m.ref_star.y; flux = m.ref_star.flux; fwhm = m.ref_star.fwhm },
         { x = m.target_star.x; y = m.target_star.y; flux = m.target_star.flux; fwhm = m.target_star.fwhm })
      ) matches in
      
      let result = {
        method_name = "rgb_star_alignment";
        filename = target_file;
        success = true;
        reference_stars = List.map (fun (ref_star, _) -> ref_star) matched_pairs;
        detected_stars = List.map (fun (_, target_star) -> target_star) matched_pairs;
        matched_pairs;
        transform;
        error_stats = (error, error *. 1.5, error /. 2.0);  (* mean, max, stddev estimates *)
        runtime = align_time;
      } in
      
      Some result
      
  | None ->
      log Error "RGB star alignment failed";
      None

(* Helper function for WCS-based alignment *)
let align_with_wcs ref_hdrh target_hdrh reference_file target_file start_time =
  try
    (* Extract WCS information *)
    let ref_wcs = extract_wcs_params ref_hdrh in
    let target_wcs = extract_wcs_params target_hdrh in
    
    (match ref_wcs, target_wcs with
    | Some ref_wcs, Some target_wcs ->
        (* Now we have two valid WCS structures *)
        log Info "Successfully retrieved WCS information for both images";
        
        (* For comparison, we'll generate a transform between the two images *)
        (* Create a grid of sample points in reference image *)
        let width = parse_int ref_hdrh "NAXIS1" in
        let height = parse_int ref_hdrh "NAXIS2" in
        
        let sample_points = ref [] in
        let step = 50 in (* Sample every 50 pixels *)
        for y = 1 to height/step - 1 do
          for x = 1 to width/step - 1 do
            sample_points := (float_of_int (x*step), float_of_int (y*step)) :: !sample_points
          done
        done;
        
        (* For each sample point, calculate the corresponding position in target image *)
        let transforms = List.map (fun (x, y) ->
          (* Convert reference pixel to sky coordinates *)
          let (ra, dec) = pixel_to_sky ref_wcs x y in
          
          (* Convert sky coordinates to target pixel coordinates *)
          let (target_x, target_y) = sky_to_pixel target_wcs ra dec in
          
          (* Calculate transform parameters for this point *)
          (x -. target_x, y -. target_y)
        ) !sample_points in
        
        (* Calculate average transform *)
        let dx_sum = ref 0.0 in
        let dy_sum = ref 0.0 in
        List.iter (fun (dx, dy) ->
          dx_sum := !dx_sum +. dx;
          dy_sum := !dy_sum +. dy;
        ) transforms;
        
        let count = float_of_int (List.length transforms) in
        let avg_dx = !dx_sum /. count in
        let avg_dy = !dy_sum /. count in
        
        (* Calculate error statistics *)
        let errors = List.map (fun (dx, dy) ->
          let err_x = dx -. avg_dx in
          let err_y = dy -. avg_dy in
          sqrt (err_x *. err_x +. err_y *. err_y)
        ) transforms in
        
        let mean_error = 
          if List.length errors > 0 then
            List.fold_left (+.) 0.0 errors /. count
          else 0.0
        in
        
        let max_error =
          if List.length errors > 0 then
            List.fold_left max 0.0 errors
          else 0.0
        in
        
        (* Calculate standard deviation *)
        let variance = 
          if List.length errors > 1 then
            List.fold_left (fun acc err ->
              let diff = err -. mean_error in
              acc +. (diff *. diff)
            ) 0.0 errors /. float_of_int (List.length errors - 1)
          else 0.0
        in
        let stddev = sqrt variance in
        
        (* Create a transform struct *)
        let transform = {
          dx = avg_dx;
          dy = avg_dy;
          rotation = 0.0;  (* We're not calculating rotation from WCS here *)
          scale = 1.0;     (* We're not calculating scale from WCS here *)
        } in
        
        (* Create dummy stars for compatibility *)
        let ref_stars = List.map (fun (x, y) ->
          { x; y; flux = 0.0; fwhm = 0.0 }
        ) !sample_points in
        
        let target_stars = List.map (fun (x, y) ->
          let (tx, ty) = (x -. avg_dx, y -. avg_dy) in
          { x = tx; y = ty; flux = 0.0; fwhm = 0.0 }
        ) !sample_points in
        
        let matched_pairs = List.map2 (fun r t -> (r, t)) ref_stars target_stars in
        
        let end_time = Unix.gettimeofday() in
        let result = {
          method_name = "plate_solve";
          filename = target_file;
          success = true;
          reference_stars = ref_stars;
          detected_stars = target_stars;
          matched_pairs;
          transform;
          error_stats = (mean_error, max_error, stddev);
          runtime = end_time -. start_time;
        } in
        
        Some result
        
    | _ ->
        log Error "Failed to extract valid WCS parameters";
        None)
  with e ->
    log Error (sprintf "Error in WCS alignment: %s" (Printexc.to_string e));
    None

(* Function to align images using plate solving *)
let align_with_plate_solve reference_file target_file =
  let start_time = Unix.gettimeofday() in
  log Info (sprintf "Aligning %s to %s using plate solving" 
    (Filename.basename target_file) (Filename.basename reference_file));
  
  try
    (* Read WCS information from FITS headers *)
    let ref_hdrh = just_header reference_file in
    let target_hdrh = just_header target_file in
    
    (* Check if both files have WCS (plate solve) information *)
    let has_wcs hdrh =
      Hashtbl.mem hdrh "CRVAL1" && Hashtbl.mem hdrh "CRVAL2" &&
      Hashtbl.mem hdrh "CRPIX1" && Hashtbl.mem hdrh "CRPIX2" &&
      Hashtbl.mem hdrh "CD1_1" && Hashtbl.mem hdrh "CD1_2" &&
      Hashtbl.mem hdrh "CD2_1" && Hashtbl.mem hdrh "CD2_2"
    in
    
    if not (has_wcs ref_hdrh && has_wcs target_hdrh) then begin
      (* If no WCS info, try to plate solve the images *)
      log Info "Missing WCS information, attempting to plate solve";
      
      (* Use the plate solving function from plate_solve_verification.ml *)
      let options = default_options in
      let ref_result = solve_field options reference_file "/tmp" in
      let target_result = solve_field options target_file "/tmp" in
      
      if not (ref_result.success && target_result.success) then begin
        log Error "Plate solving failed for one or both images";
        None
      end else begin
        (* Read the solved FITS headers *)
        let ref_solved = just_header (Filename.concat "/tmp" 
          (Filename.remove_extension (Filename.basename reference_file) ^ ".fits")) in
        let target_solved = just_header (Filename.concat "/tmp"
          (Filename.remove_extension (Filename.basename target_file) ^ ".fits")) in
          
        (* Continue with the solved headers *)
        align_with_wcs ref_solved target_solved reference_file target_file start_time
      end
    end else
      (* Already have WCS info, proceed with alignment *)
      align_with_wcs ref_hdrh target_hdrh reference_file target_file start_time
      
  with e ->
    log Error (sprintf "Error in plate solve alignment: %s" (Printexc.to_string e));
    None

(* Function to compare the two methods *)
let compare_methods reference_file target_file =
  log Info (sprintf "Comparing alignment methods for %s to %s" 
    (Filename.basename target_file) (Filename.basename reference_file));
    
  let star_threshold = 3.0 in
  let max_stars = 50 in
  
  let plate_result = align_with_plate_solve reference_file target_file in
  let star_result = align_with_stars reference_file target_file star_threshold max_stars in
  
  match plate_result, star_result with
  | Some plate, Some star ->
      let (plate_mean, _, _) = plate.error_stats in
      let (star_mean, _, _) = star.error_stats in
      
      let comparison = {
        filename = Filename.basename target_file;
        plate_solve_success = plate.success;
        star_align_success = star.success;
        plate_solve_stars = List.length plate.matched_pairs;
        star_align_stars = List.length star.matched_pairs;
        plate_solve_error = plate_mean;
        star_align_error = star_mean;
        runtime_ratio = plate.runtime /. star.runtime;
      } in
      
      log Info "\nComparison Results:";
      log Info "==================";
      log Info (sprintf "File: %s" comparison.filename);
      log Info (sprintf "Plate solve success: %b" comparison.plate_solve_success);
      log Info (sprintf "Star alignment success: %b" comparison.star_align_success);
      log Info (sprintf "Plate solve stars: %d" comparison.plate_solve_stars);
      log Info (sprintf "Star alignment stars: %d" comparison.star_align_stars);
      log Info (sprintf "Plate solve error: %.2f pixels" comparison.plate_solve_error);
      log Info (sprintf "Star alignment error: %.2f pixels" comparison.star_align_error);
      log Info (sprintf "Runtime ratio (plate/star): %.2fx" comparison.runtime_ratio);
      
      let winner = 
        if not plate.success then "Star alignment (plate solve failed)"
        else if not star.success then "Plate solving (star alignment failed)"
        else if plate_mean < star_mean then "Plate solving (lower error)"
        else "Star alignment (lower error)"
      in
      
      log Info (sprintf "Winner: %s" winner);
      Some comparison
      
  | Some plate, None ->
      log Info "\nOnly plate solving succeeded";
      let (mean, _, _) = plate.error_stats in
      
      let comparison = {
        filename = Filename.basename target_file;
        plate_solve_success = plate.success;
        star_align_success = false;
        plate_solve_stars = List.length plate.matched_pairs;
        star_align_stars = 0;
        plate_solve_error = mean;
        star_align_error = 0.0;
        runtime_ratio = 0.0;  (* N/A *)
      } in
      
      log Info (sprintf "Plate solve error: %.2f pixels" comparison.plate_solve_error);
      log Info "Winner: Plate solving (star alignment failed)";
      Some comparison
      
  | None, Some star ->
      log Info "\nOnly star alignment succeeded";
      let (mean, _, _) = star.error_stats in
      
      let comparison = {
        filename = Filename.basename target_file;
        plate_solve_success = false;
        star_align_success = star.success;
        plate_solve_stars = 0;
        star_align_stars = List.length star.matched_pairs;
        plate_solve_error = 0.0;
        star_align_error = mean;
        runtime_ratio = 0.0;  (* N/A *)
      } in
      
      log Info (sprintf "Star alignment error: %.2f pixels" comparison.star_align_error);
      log Info "Winner: Star alignment (plate solve failed)";
      Some comparison
      
  | None, None ->
      log Error "Both methods failed";
      None

(* Process a directory of images *)
let process_directory reference_file image_dir output_dir =
  log Info (sprintf "Processing directory: %s" image_dir);
  log Info (sprintf "Reference file: %s" reference_file);
  log Info (sprintf "Output directory: %s" output_dir);
  
  (* Create output directory if it doesn't exist *)
  if not (Sys.file_exists output_dir) then
    Unix.mkdir output_dir 0o755;
    
  (* Find all FITS files in the directory *)
  let files = 
    try 
      Sys.readdir image_dir
      |> Array.to_list
      |> List.filter (fun f -> 
          Filename.check_suffix f ".fits" || 
          Filename.check_suffix f ".fit")
      |> List.map (fun f -> Filename.concat image_dir f)
    with _ -> 
      log Error (sprintf "Error reading directory %s" image_dir);
      []
  in
  
  log Info (sprintf "Found %d FITS files" (List.length files));
  
  (* Process each file *)
  let results = List.filter_map (fun file ->
    compare_methods reference_file file
  ) files in
  
  (* Generate summary report *)
  let csv_path = Filename.concat output_dir "comparison_results.csv" in
  let html_path = Filename.concat output_dir "comparison_report.html" in
  
  (* Write CSV results *)
  let csv = open_out csv_path in
  fprintf csv "Filename,PlateSolveSuccess,StarAlignSuccess,PlateSolveStars,";
  fprintf csv "StarAlignStars,PlateSolveError,StarAlignError,RuntimeRatio,Winner\n";
  
  List.iter (fun r ->
    let winner = 
      if not r.plate_solve_success then "Star"
      else if not r.star_align_success then "Plate"
      else if r.plate_solve_error < r.star_align_error then "Plate"
      else "Star"
    in
    
    fprintf csv "%s,%b,%b,%d,%d,%.2f,%.2f,%.2f,%s\n"
      r.filename
      r.plate_solve_success
      r.star_align_success
      r.plate_solve_stars
      r.star_align_stars
      r.plate_solve_error
      r.star_align_error
      r.runtime_ratio
      winner
  ) results;
  
  close_out csv;
  log Info (sprintf "Results saved to %s" csv_path);
  
  (* Write HTML report *)
  let html = open_out html_path in
  
  fprintf html "<!DOCTYPE html>\n<html>\n<head>\n";
  fprintf html "<title>Alignment Method Comparison Report</title>\n";
  fprintf html "<style>\n";
  fprintf html "body { font-family: Arial, sans-serif; margin: 20px; }\n";
  fprintf html "h1, h2 { color: #333; }\n";
  fprintf html "table { border-collapse: collapse; width: 100%%; }\n";
  fprintf html "th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }\n";
  fprintf html "tr:hover { background-color: #f5f5f5; }\n";
  fprintf html "th { background-color: #4CAF50; color: white; }\n";
  fprintf html ".plate { color: blue; }\n";
  fprintf html ".star { color: green; }\n";
  fprintf html ".failed { color: red; }\n";
  fprintf html ".summary { margin: 20px 0; }\n";
  fprintf html ".summary-item { margin: 10px 0; }\n";
  fprintf html "</style>\n</head>\n<body>\n";
  
  fprintf html "<h1>Alignment Method Comparison Report</h1>\n";
  
  (* Calculate summary statistics *)
  let total_files = List.length results in
  let plate_successes = List.filter (fun r -> r.plate_solve_success) results |> List.length in
  let star_successes = List.filter (fun r -> r.star_align_success) results |> List.length in
  let both_successes = List.filter (fun r -> r.plate_solve_success && r.star_align_success) results |> List.length in
  
  let plate_winners = List.filter (fun r -> 
    r.plate_solve_success && (not r.star_align_success || r.plate_solve_error < r.star_align_error)
  ) results |> List.length in
  
  let star_winners = List.filter (fun r -> 
    r.star_align_success && (not r.plate_solve_success || r.star_align_error <= r.plate_solve_error)
  ) results |> List.length in
  
  (* Calculate average errors *)
  let plate_errors = List.filter_map (fun r -> 
    if r.plate_solve_success then Some r.plate_solve_error else None
  ) results in
  
  let star_errors = List.filter_map (fun r -> 
    if r.star_align_success then Some r.star_align_error else None
  ) results in
  
  let avg_plate_error = 
    if List.length plate_errors > 0 then
      List.fold_left (+.) 0.0 plate_errors /. float_of_int (List.length plate_errors)
    else 0.0
  in
  
  let avg_star_error = 
    if List.length star_errors > 0 then
      List.fold_left (+.) 0.0 star_errors /. float_of_int (List.length star_errors)
    else 0.0
  in
  
  (* Calculate average runtime ratio *)
  let runtime_ratios = List.filter_map (fun r -> 
    if r.runtime_ratio > 0.0 then Some r.runtime_ratio else None
  ) results in
  
  let avg_runtime_ratio = 
    if List.length runtime_ratios > 0 then
      List.fold_left (+.) 0.0 runtime_ratios /. float_of_int (List.length runtime_ratios)
    else 0.0
  in
  
  (* Print summary *)
  fprintf html "<div class='summary'>\n";
  fprintf html "<h2>Summary</h2>\n";
  fprintf html "<div class='summary-item'>Total images: <strong>%d</strong></div>\n" total_files;
  fprintf html "<div class='summary-item'>Plate solve successes: <strong>%d</strong> (%.1f%%)</div>\n" 
    plate_successes (float_of_int plate_successes *. 100.0 /. float_of_int total_files);
  fprintf html "<div class='summary-item'>Star alignment successes: <strong>%d</strong> (%.1f%%)</div>\n" 
    star_successes (float_of_int star_successes *. 100.0 /. float_of_int total_files);
  fprintf html "<div class='summary-item'>Both methods successful: <strong>%d</strong> (%.1f%%)</div>\n" 
    both_successes (float_of_int both_successes *. 100.0 /. float_of_int total_files);
    
  fprintf html "<div class='summary-item'>Average plate solve error: <strong>%.2f</strong> pixels</div>\n" 
    avg_plate_error;
  fprintf html "<div class='summary-item'>Average star alignment error: <strong>%.2f</strong> pixels</div>\n" 
    avg_star_error;
  fprintf html "<div class='summary-item'>Average runtime ratio (plate/star): <strong>%.2fx</strong></div>\n" 
    avg_runtime_ratio;
    
  fprintf html "<div class='summary-item'>Plate solve winners: <strong>%d</strong> (%.1f%%)</div>\n" 
    plate_winners (float_of_int plate_winners *. 100.0 /. float_of_int total_files);
  fprintf html "<div class='summary-item'>Star alignment winners: <strong>%d</strong> (%.1f%%)</div>\n" 
    star_winners (float_of_int star_winners *. 100.0 /. float_of_int total_files);
  fprintf html "</div>\n";
  
  (* Results table *)
  fprintf html "<h2>Results</h2>\n";
  fprintf html "<table>\n";
  fprintf html "<tr><th>Filename</th><th>Plate Solve</th><th>Star Align</th>";
  fprintf html "<th>Plate Stars</th><th>Star Matches</th><th>Plate Error</th>";
  fprintf html "<th>Star Error</th><th>Runtime Ratio</th><th>Winner</th></tr>\n";
  
  List.iter (fun r ->
    let winner = 
      if not r.plate_solve_success then "Star"
      else if not r.star_align_success then "Plate"
      else if r.plate_solve_error < r.star_align_error then "Plate"
      else "Star"
    in
    
    fprintf html "<tr>\n";
    fprintf html "  <td>%s</td>\n" r.filename;
    fprintf html "  <td class='%s'>%s</td>\n" 
      (if r.plate_solve_success then "plate" else "failed")
      (if r.plate_solve_success then "Success" else "Failed");
    fprintf html "  <td class='%s'>%s</td>\n"
      (if r.star_align_success then "star" else "failed")
      (if r.star_align_success then "Success" else "Failed");
    fprintf html "  <td>%d</td>\n" r.plate_solve_stars;
    fprintf html "  <td>%d</td>\n" r.star_align_stars;
    fprintf html "  <td>%.2f</td>\n" r.plate_solve_error;
    fprintf html "  <td>%.2f</td>\n" r.star_align_error;
    fprintf html "  <td>%.2fx</td>\n" r.runtime_ratio;
    fprintf html "  <td class='%s'>%s</td>\n"
      (String.lowercase_ascii winner)
      winner;
    fprintf html "</tr>\n";
  ) results;
  
  fprintf html "</table>\n";
  fprintf html "</body>\n</html>\n";
  
  close_out html;
  log Info (sprintf "HTML report saved to %s" html_path);
  
  (* Return statistics *)
  (plate_successes, star_successes, both_successes)

(* Main function *)
let main () =
  (* Parse command line arguments *)
  let reference_file = ref "" in
  let image_dir = ref "" in
  let output_dir = ref "comparison_results" in
  
  let specs = [
    ("-ref", Arg.Set_string reference_file, "Reference image file");
    ("-dir", Arg.Set_string image_dir, "Directory containing images to align");
    ("-out", Arg.Set_string output_dir, "Output directory for results");
  ] in
  
  let usage = "Usage: stack_comparison -ref reference.fits -dir image_directory [-out result_directory]" in
  
  Arg.parse specs (fun _ -> ()) usage;
  
  if !reference_file = "" then begin
    printf "Error: Reference file must be specified with -ref\n";
    Arg.usage specs usage;
    exit 1
  end;
  
  if !image_dir = "" then begin
    printf "Error: Image directory must be specified with -dir\n";
    Arg.usage specs usage;
    exit 1
  end;
  
  (* Run the comparison *)
  let (plate_successes, star_successes, both_successes) = 
    process_directory !reference_file !image_dir !output_dir in
  
  (* Print final summary *)
  printf "\nFinal Summary:\n";
  printf "=============\n";
  printf "Plate solve successes: %d\n" plate_successes;
  printf "Star alignment successes: %d\n" star_successes;
  printf "Both methods successful: %d\n" both_successes;
  
  (* Exit with success status *)
  exit 0

(* Run the main function *)
let () = main ()
  
