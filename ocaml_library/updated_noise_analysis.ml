(* Modified noise analysis functions with debayering support *)

(* Import the noise analysis and debayering modules *)
module NoiseAnalysis = Astro_noise_analysis
module Debayer = Debayer

(* Analyze noise in a single image file with debayering support *)
let analyze_image_noise filename threshold =
  try
    printf "Reading image data from %s...\n" filename;
    let img = Fits.read_image filename in
    let hdrh, contents = Fits.find_header_end filename img in
    let width = parse_int hdrh "NAXIS1" in
    let height = parse_int hdrh "NAXIS2" in
    
    printf "Image dimensions: %dx%d\n" width height;
    
    (* Check for Bayer pattern *)
    let bayer_pattern = Debayer.detect_bayer_pattern filename hdrh in
    
    (* Log detected pattern *)
    (match bayer_pattern with
    | Some pattern -> 
        printf "Detected %s Bayer pattern\n" (Debayer.debayer_pattern_to_string pattern)
    | None -> 
        printf "No Bayer pattern detected, treating as monochrome\n");
    
    printf "Reading image data...\n";
    let data = Fits.read_fits_data contents width height in
    
    (* Process the image data based on Bayer pattern *)
    let rgb_data = match bayer_pattern with
    | Some pattern ->
        printf "Applying 2x2 binning with %s pattern...\n" 
          (Debayer.debayer_pattern_to_string pattern);
        Debayer.bin_bayer_pattern data width height pattern
    | None ->
        (* No Bayer pattern, use simple 2x2 binning for monochrome *)
        if width > 1024 || height > 1024 then begin
          printf "No Bayer pattern, applying simple 2x2 binning...\n";
          Debayer.bin_2x2_mono data width height
        end else begin
          (* Small image, no binning needed *)
          printf "Small image, no binning applied...\n";
          let mono_data = Array.make_matrix height width (0, 0, 0) in
          for y = 0 to height - 1 do
            for x = 0 to width - 1 do
              let value = data.(y).(x) in
              mono_data.(y).(x) <- (value, value, value)
            done
          done;
          mono_data
        end
    in
    
    (* Get dimensions of processed data (might be different after binning) *)
    let proc_height = Array.length rgb_data in
    let proc_width = Array.length rgb_data.(0) in
    printf "Processed dimensions: %dx%d\n" proc_width proc_height;
    
    printf "Analyzing noise with star detection threshold of %.1f...\n" threshold;
    let noise_result = NoiseAnalysis.analyze_astronomical_noise rgb_data in
    
    (* Extract and display results *)
    let (y_mean, y_stddev) = noise_result.background_noise.y_stats in
    let (cb_mean, cb_stddev) = noise_result.background_noise.cb_stats in
    let (cr_mean, cr_stddev) = noise_result.background_noise.cr_stats in
    
    printf "\nNoise Analysis Results for %s:\n" (Filename.basename filename);
    printf "==============================%s\n" (String.make (String.length (Filename.basename filename)) '=');
    printf "Background Statistics (YCbCr color space):\n";
    printf "  Y channel:  Mean=%.4f, StdDev=%.4f\n" y_mean y_stddev;
    printf "  Cb channel: Mean=%.4f, StdDev=%.4f\n" cb_mean cb_stddev;
    printf "  Cr channel: Mean=%.4f, StdDev=%.4f\n" cr_mean cr_stddev;
    printf "Star Fraction: %.2f%% of image contains stars\n" (noise_result.star_fraction *. 100.0);
    
    (* Signal-to-Noise ratio estimate *)
    if y_stddev > 0.0 then
      printf "Estimated S/N ratio: %.2f\n" (y_mean /. y_stddev)
    else
      printf "Estimated S/N ratio: N/A (stddev is zero)\n";
    
    (* Additional information based on Bayer pattern *)
    (match bayer_pattern with
    | Some _ -> 
        printf "\nNote: Analysis performed on debayered RGB data\n";
        if cb_stddev < 0.0001 || cr_stddev < 0.0001 then
          printf "Warning: Very low color channel variation. Check debayering or image source.\n"
    | None -> ());
    
    (* Additional information about the image quality *)
    if noise_result.star_fraction < 0.01 then
      printf "\nNote: Very few stars detected (%.2f%%). Consider checking exposure settings or focus.\n" 
        (noise_result.star_fraction *. 100.0)
    else if noise_result.star_fraction > 0.3 then
      printf "\nNote: Large fraction of image contains stars (%.2f%%). Background noise estimate may be affected.\n"
        (noise_result.star_fraction *. 100.0);
    
    true
  with e ->
    printf "Error analyzing image noise: %s\n" (Printexc.to_string e);
    false

(* Analyze noise in a directory of FITS files with debayering support *)
let analyze_directory_noise files threshold =
  printf "Analyzing noise in %d files...\n" (Array.length files);
  
  let total_files = Array.length files in
  let successful = ref 0 in
  let failed = ref 0 in
  
  (* Arrays to store statistics for summary *)
  let y_means = ref [] in
  let y_stddevs = ref [] in
  let cb_means = ref [] in
  let cb_stddevs = ref [] in
  let cr_means = ref [] in
  let cr_stddevs = ref [] in
  let star_fractions = ref [] in
  
  (* Track number of files with Bayer patterns *)
  let bayer_files = ref 0 in
  let bayer_patterns = ref [] in
  
  Array.iteri (fun i file ->
    printf "\n[%d/%d] Processing %s\n" (i+1) total_files (Filename.basename file);
    
    try
      let img = Fits.read_image file in
      let hdrh, contents = Fits.find_header_end file img in
      let width = parse_int hdrh "NAXIS1" in
      let height = parse_int hdrh "NAXIS2" in
      
      (* Detect Bayer pattern *)
      let bayer_pattern = Debayer.detect_bayer_pattern file hdrh in
      
      (match bayer_pattern with
      | Some pattern -> 
          printf "  Detected %s Bayer pattern\n" (Debayer.debayer_pattern_to_string pattern);
          incr bayer_files;
          bayer_patterns := (Debayer.debayer_pattern_to_string pattern) :: !bayer_patterns
      | None -> 
          printf "  No Bayer pattern detected, treating as monochrome\n");
      
      let data = Fits.read_fits_data contents width height in
      
      (* Process the image data based on Bayer pattern *)
      let rgb_data = match bayer_pattern with
      | Some pattern ->
          Debayer.bin_bayer_pattern data width height pattern
      | None ->
          (* No Bayer pattern, use simple 2x2 binning for monochrome if large *)
          if width > 1024 || height > 1024 then
            Debayer.bin_2x2_mono data width height
          else begin
            (* Small image, no binning needed *)
            let mono_data = Array.make_matrix height width (0, 0, 0) in
            for y = 0 to height - 1 do
              for x = 0 to width - 1 do
                let value = data.(y).(x) in
                mono_data.(y).(x) <- (value, value, value)
              done
            done;
            mono_data
          end
      in
      
      let noise_result = NoiseAnalysis.analyze_astronomical_noise rgb_data in
      let (y_mean, y_stddev) = noise_result.background_noise.y_stats in
      let (cb_mean, cb_stddev) = noise_result.background_noise.cb_stats in
      let (cr_mean, cr_stddev) = noise_result.background_noise.cr_stats in
      
      (* Store results for summary *)
      y_means := y_mean :: !y_means;
      y_stddevs := y_stddev :: !y_stddevs;
      cb_means := cb_mean :: !cb_means;
      cb_stddevs := cb_stddev :: !cb_stddevs;
      cr_means := cr_mean :: !cr_means;
      cr_stddevs := cr_stddev :: !cr_stddevs;
      star_fractions := noise_result.star_fraction :: !star_fractions;
      
      printf "  Background Y: mean=%.4f, stddev=%.4f\n" y_mean y_stddev;
      printf "  Star fraction: %.2f%%\n" (noise_result.star_fraction *. 100.0);
      if bayer_pattern <> None then
        printf "  Color: Cb(%.4f, %.4f), Cr(%.4f, %.4f)\n" 
          cb_mean cb_stddev cr_mean cr_stddev;
      
      incr successful
    with e ->
      printf "  Error analyzing %s: %s\n" (Filename.basename file) (Printexc.to_string e);
      incr failed
  ) files;
  
  (* Print summary statistics *)
  if !successful > 0 then begin
    printf "\nNoise Analysis Summary (%d files processed, %d failed):\n" !successful !failed;
    printf "===============================================\n";
    
    (* Calculate average statistics *)
    let avg_y_mean = List.fold_left (+.) 0.0 !y_means /. float_of_int !successful in
    let avg_y_stddev = List.fold_left (+.) 0.0 !y_stddevs /. float_of_int !successful in
    let avg_star_fraction = List.fold_left (+.) 0.0 !star_fractions /. float_of_int !successful in
    
    (* Calculate color channel averages if any Bayer files *)
    let avg_cb_mean = List.fold_left (+.) 0.0 !cb_means /. float_of_int !successful in
    let avg_cb_stddev = List.fold_left (+.) 0.0 !cb_stddevs /. float_of_int !successful in
    let avg_cr_mean = List.fold_left (+.) 0.0 !cr_means /. float_of_int !successful in
    let avg_cr_stddev = List.fold_left (+.) 0.0 !cr_stddevs /. float_of_int !successful in
    
    (* Calculate range *)
    let sorted_means = List.sort compare !y_means in
    let sorted_stddevs = List.sort compare !y_stddevs in
    
    let min_mean = List.hd sorted_means in
    let max_mean = List.hd (List.rev sorted_means) in
    let min_stddev = List.hd sorted_stddevs in
    let max_stddev = List.hd (List.rev sorted_stddevs) in
    
    printf "Background Mean Level (Y channel):\n";
    printf "  Average: %.4f\n" avg_y_mean;
    printf "  Range: %.4f to %.4f\n" min_mean max_mean;
    
    printf "Background Noise (Y channel StdDev):\n";
    printf "  Average: %.4f\n" avg_y_stddev;
    printf "  Range: %.4f to %.4f\n" min_stddev max_stddev;
    
    printf "Average S/N Ratio: %.2f\n" (avg_y_mean /. avg_y_stddev);
    printf "Average Star Coverage: %.2f%%\n" (avg_star_fraction *. 100.0);
    
    (* Report on Bayer patterns *)
    if !bayer_files > 0 then begin
      printf "\nDebayering Information:\n";
      printf "  Files with Bayer pattern: %d of %d\n" !bayer_files !successful;
      
      (* Count pattern occurrences *)
      let pattern_counts = Hashtbl.create 4 in
      List.iter (fun pattern ->
        let count = try Hashtbl.find pattern_counts pattern with Not_found -> 0 in
        Hashtbl.replace pattern_counts pattern (count + 1)
      ) !bayer_patterns;
      
      printf "  Patterns detected:\n";
      Hashtbl.iter (fun pattern count ->
        printf "    %s: %d files\n" pattern count
      ) pattern_counts;
      
      (* Color channel statistics *)
      printf "\nColor Channel Analysis:\n";
      printf "  Cb channel: Mean=%.4f, StdDev=%.4f\n" avg_cb_mean avg_cb_stddev;
      printf "  Cr channel: Mean=%.4f, StdDev=%.4f\n" avg_cr_mean avg_cr_stddev;
    end;
    
    true
  end else begin
    printf "\nFailed to analyze any files successfully.\n";
    false
  end