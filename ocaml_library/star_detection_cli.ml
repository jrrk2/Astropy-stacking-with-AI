(* star_detection_cli.ml - Command-line interface for star detection *)

open Bigarray
open Types
open Printf
open Fits_utils
open Plate_solve_verification

(* Generate visualization of detected stars *)
let generate_star_visualization filename stars output_dir =
  let base_name = Filename.basename filename |> Filename.remove_extension in
  let viz_path = Filename.concat output_dir (base_name ^ "_stars.png") in
  
  (* Simple visualization using PPM format (can be converted to PNG later) *)
  try
    (* Read image data to get dimensions *)
    let (hdrh, data) = read_fits_large filename in
    let width = Array2.dim2 data in
    let height = Array2.dim1 data in
    
    (* Create PPM file instead (easier to generate) *)
    let ppm_path = Filename.concat output_dir (base_name ^ "_stars.ppm") in
    let oc = open_out ppm_path in
    
    (* PPM header *)
    fprintf oc "P3\n";
    fprintf oc "%d %d\n" width height;
    fprintf oc "255\n";
    
    (* Find max value for scaling *)
    let max_val = ref 0 in
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        max_val := max !max_val (get_pixel data y x)
      done
    done;
    
    (* Scale factor *)
    let scale = 255.0 /. float_of_int !max_val in
    
    (* Create star position map *)
    let star_map = Array.make_matrix height width false in
    List.iter (fun star ->
      let x = int_of_float (star.x +. 0.5) in
      let y = int_of_float (star.y +. 0.5) in
      if x >= 0 && x < width && y >= 0 && y < height then begin
        (* Mark a cross centered on the star *)
        let mark_if_valid y x =
          if x >= 0 && x < width && y >= 0 && y < height then
            star_map.(y).(x) <- true
        in
        (* Central pixel *)
        mark_if_valid y x;
        (* Cross pattern *)
        for d = 1 to 5 do
          mark_if_valid y (x + d);
          mark_if_valid y (x - d);
          mark_if_valid (y + d) x;
          mark_if_valid (y - d) x;
        done;
      end
    ) stars;
    
    (* Write pixel data with star markers *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let val16 = get_pixel data y x in
        let scaled = int_of_float (float_of_int val16 *. scale) in
        if star_map.(y).(x) then
          (* Red marker for stars *)
          fprintf oc "255 0 0 "
        else
          (* Grayscale for image *)
          fprintf oc "%d %d %d " scaled scaled scaled;
      done;
      fprintf oc "\n";
    done;
    
    close_out oc;
    printf "  Generated visualization: %s\n" ppm_path;
    
    (* Try to convert to PNG if ImageMagick is available *)
    let convert_cmd = sprintf "magick %s %s" ppm_path viz_path in
    let convert_result = Sys.command convert_cmd in
    
    if convert_result = 0 then begin
      printf "  Converted to PNG: %s\n" viz_path;
      Sys.remove ppm_path  (* Remove PPM file *)
    end
  with e ->
    printf "  Error generating visualization for %s: %s\n" 
      (Filename.basename filename) (Printexc.to_string e)

(* Generate summary visualization (e.g., histogram of star counts) *)
let generate_summary_visualization results output_dir =
  (* Create histogram of star counts *)
  let star_counts = List.map (fun (_, _, stars) -> List.length stars) results in
  
  (* Simple text-based histogram for now *)
  let hist_path = Filename.concat output_dir "star_count_histogram.txt" in
  let oc = open_out hist_path in
  
  (* Find range *)
  let min_count = List.fold_left min (List.hd star_counts) star_counts in
  let max_count = List.fold_left max (List.hd star_counts) star_counts in
  
  (* Create bins *)
  let bin_count = 20 in
  let bin_size = max 1 ((max_count - min_count + 1) / bin_count) in
  let bins = Array.make bin_count 0 in
  
  (* Fill bins *)
  List.iter (fun count ->
    let bin = (count - min_count) / bin_size in
    let bin_idx = min (bin_count - 1) (max 0 bin) in
    bins.(bin_idx) <- bins.(bin_idx) + 1
  ) star_counts;
  
  (* Write histogram *)
  fprintf oc "Star Count Histogram\n";
  fprintf oc "------------------\n";
  
  for i = 0 to bin_count - 1 do
    let bin_start = min_count + i * bin_size in
    let bin_end = bin_start + bin_size - 1 in
    fprintf oc "%4d-%-4d | " bin_start bin_end;
    output_string oc (String.make bins.(i) '#');
    fprintf oc " (%d)\n" bins.(i)
  done;
  
  close_out oc;
  printf "  Generated star count histogram: %s\n" hist_path

(* Main function for command-line operation *)
let main () =
  (* Parse command line arguments *)
  let input_dir = ref "" in
  let output_dir = ref "star_detection_results" in
  let threshold = ref 3.0 in
  let workers = ref 0 in  (* 0 means auto-detect *)
  let visualize = ref false in

  let specs = [
    ("-i", Arg.Set_string input_dir, "Input directory containing FITS files");
    ("-o", Arg.Set_string output_dir, "Output directory for detection results");
    ("-threshold", Arg.Set_float threshold, "Detection threshold in sigma above background (default: 3.0)");
    ("-workers", Arg.Set_int workers, "Number of parallel workers (default: auto-detect)");
    ("-visualize", Arg.Set visualize, "Generate visualization images for detected stars");
  ] in
  
  let usage = "Usage: star_detection_cli -i input_dir [-o output_dir] [-threshold value] [-workers n] [-visualize]" in
  
  Arg.parse specs (fun _ -> ()) usage;
  
  if !input_dir = "" then begin
    printf "Error: Input directory must be specified with -i\n";
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
      printf "Error reading directory %s\n" !input_dir;
      exit 1
    end
  in
  
  if List.length files = 0 then begin
    printf "No FITS files found in %s\n" !input_dir;
    exit 1
  end;
  
  (* Create output directory if it doesn't exist *)
  if not (Sys.file_exists !output_dir) then
    create_dir !output_dir;
  
  (* Determine number of workers *)
  let cpu_count = detect_cpu_count () in
  let max_workers = 
    if !workers > 0 then !workers
    else max 1 (cpu_count * 3 / 4) (* Default to 75% of CPU cores *)
  in
  
  printf "Found %d FITS files to process\n" (List.length files);
  printf "Running with %d parallel workers (detected %d CPUs)\n" 
    max_workers cpu_count;
  printf "Using detection threshold of %.1f sigma\n" !threshold;
  
  (* Process files in parallel *)
  let results = parallel_star_detection files ~threshold:!threshold ~max_workers in
  
  (* Generate summary report *)
  let csv_path = Filename.concat !output_dir "star_detection_results.csv" in
  
  (* Write CSV results *)
  let csv = open_out csv_path in
  fprintf csv "Filename,Width,Height,Stars,MinValue,MaxValue,Mean,StdDev,MedianFWHM,DateTimeObs\n";
  
  List.iter2 (fun filename (hdrh, stats, stars) ->
    (* Calculate median FWHM *)
    let sorted_fwhms = List.map (fun (star:star_point) -> star.fwhm) stars 
                      |> List.sort compare in
    let median_fwhm = 
      if List.length sorted_fwhms > 0 then
        List.nth sorted_fwhms (List.length sorted_fwhms / 2)
      else
        0.0
    in
    
    (* Get date-time if available *)
    let date_time = 
      try
        let timestamp = get_timestamp hdrh in
        let tm = Unix.localtime timestamp in
        sprintf "%04d-%02d-%02d %02d:%02d:%02d"
          (tm.tm_year + 1900) (tm.tm_mon + 1) tm.tm_mday
          tm.tm_hour tm.tm_min tm.tm_sec
      with _ -> ""
    in
    
    fprintf csv "%s,%d,%d,%d,%d,%d,%.1f,%.1f,%.2f,%s\n"
      (Filename.basename filename)
      stats.width
      stats.height
      (List.length stars)
      stats.min_value
      stats.max_value
      stats.mean
      stats.stddev
      median_fwhm
      date_time
  ) files results;
  
  close_out csv;
  printf "Results saved to %s\n" csv_path;
  
  (* Generate individual star lists *)
  List.iter2 (fun filename (hdrh, stats, stars) ->
    let base_name = Filename.basename filename |> Filename.remove_extension in
    let star_list_path = Filename.concat !output_dir (base_name ^ "_stars.csv") in
    
    let star_csv = open_out star_list_path in
    fprintf star_csv "X,Y,Flux,FWHM\n";
    
    List.iter (fun (star:star_point) ->
      fprintf star_csv "%.2f,%.2f,%.1f,%.2f\n"
        star.x star.y star.flux star.fwhm
    ) stars;
    
    close_out star_csv;
    
    (* Generate visualization if requested *)
    if !visualize then 
      generate_star_visualization filename stars !output_dir
  ) files results;
  
  (* Calculate some aggregate statistics *)
  let total_stars = List.fold_left (fun acc (_, _, stars) -> 
    acc + List.length stars
  ) 0 results in
  
  let avg_stars_per_image = 
    float_of_int total_stars /. float_of_int (List.length results) in
  
  printf "\nStar detection complete\n";
  printf "  Processed %d images\n" (List.length results);
  printf "  Detected %d stars total (average %.1f per image)\n" 
    total_stars avg_stars_per_image;
  printf "  Results saved to %s\n" !output_dir;
  
  (* Generate summary visualization if requested *)
  if !visualize then begin
    printf "  Generating detection summary visualizations\n";
    generate_summary_visualization results !output_dir;
  end;
  
  (* Exit with success *)
  exit 0

(* Run the main function *)
let () = main ()
