(* alignment_checker.ml - Functions to visualize and check image alignment *)
open Types
open Fits
open Printf

(* Create a difference image between two aligned images *)
let create_difference_image ref_data aligned_data width height output_path =
  (* Create output array *)
  let diff_data = Array.make_matrix height width 0 in
  
  (* Calculate abs difference for each pixel *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let diff = abs (aligned_data.(y).(x) - ref_data.(y).(x)) in
      diff_data.(y).(x) <- diff;
    done;
  done;
  
  (* Write to FITS file *)
  let oc = open_out_bin output_path in
  
  (* Create a basic FITS header *)
  let hdrh = Hashtbl.create 20 in
  Hashtbl.add hdrh "SIMPLE" " = T / FITS format";
  Hashtbl.add hdrh "BITPIX" " = 16 / 16-bit signed integers";
  Hashtbl.add hdrh "NAXIS" " = 2 / Number of axes";
  Hashtbl.add hdrh "NAXIS1" (Printf.sprintf " = %d / Width in pixels" width);
  Hashtbl.add hdrh "NAXIS2" (Printf.sprintf " = %d / Height in pixels" height);
  Hashtbl.add hdrh "BSCALE" " = 1.0 / Scale factor";
  Hashtbl.add hdrh "BZERO" " = 0.0 / Offset";
  Hashtbl.add hdrh "IMGTYPE" " = 'DIFFERENCE' / Difference image";
  
  (* Write header *)
  let _ = write_fits_header oc hdrh in
  
  (* Write data *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let value = diff_data.(y).(x) in
      output_byte oc (value lsr 8);      (* High byte *)
      output_byte oc (value land 0xFF);  (* Low byte *)
    done;
  done;
  
  (* Pad data to multiple of 2880 bytes *)
  let data_size = width * height * 2 in
  let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
  output_string oc (String.make padding_size '\000');
  
  close_out oc;
  printf "Difference image saved to %s\n" output_path;
  flush stdout

(* Create color overlay image with reference in red, aligned in blue *)
let create_color_overlay ref_data aligned_data width height output_path =
  (* Create RGB data array *)
  let rgb_data = Array.make_matrix height width (0, 0, 0) in
  
  (* Assign reference image to red channel, aligned to blue *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let r_val = min 65535 (max 0 ref_data.(y).(x)) in
      let b_val = min 65535 (max 0 aligned_data.(y).(x)) in
      
      (* Use both values for green to highlight perfectly aligned areas in white *)
      let g_val = min r_val b_val in
      
      rgb_data.(y).(x) <- (r_val, g_val, b_val);
    done;
  done;
  
  (* Write RGB FITS file *)
  let hdrh = Hashtbl.create 20 in
  Hashtbl.add hdrh "SIMPLE" " = T / FITS format";
  Hashtbl.add hdrh "BITPIX" " = 16 / 16-bit signed integers";
  Hashtbl.add hdrh "NAXIS" " = 3 / Number of axes";
  Hashtbl.add hdrh "NAXIS1" (Printf.sprintf " = %d / Width in pixels" width);
  Hashtbl.add hdrh "NAXIS2" (Printf.sprintf " = %d / Height in pixels" height);
  Hashtbl.add hdrh "NAXIS3" " = 3 / Number of color planes (RGB)";
  Hashtbl.add hdrh "BZERO" " = 32768 / Offset to unsigned short range";
  Hashtbl.add hdrh "BSCALE" " = 1 / Default scaling factor";
  Hashtbl.add hdrh "IMGTYPE" " = 'OVERLAY' / Color overlay image";
  
  (* Write RGB FITS file manually to ensure all planes are included *)
  let oc = open_out_bin output_path in
  
  (* Write header *)
  ignore (write_fits_header oc hdrh);
  
  (* Write red plane *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let (r, _, _) = rgb_data.(y).(x) in
      output_byte oc (r lsr 8);
      output_byte oc (r land 0xFF);
    done;
  done;
  
  (* Write green plane *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let (_, g, _) = rgb_data.(y).(x) in
      output_byte oc (g lsr 8);
      output_byte oc (g land 0xFF);
    done;
  done;
  
  (* Write blue plane *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let (_, _, b) = rgb_data.(y).(x) in
      output_byte oc (b lsr 8);
      output_byte oc (b land 0xFF);
    done;
  done;
  
  (* Pad data to multiple of 2880 bytes *)
  let data_size = width * height * 2 * 3 in
  let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
  if padding_size > 0 then
    output_string oc (String.make padding_size '\000');
  
  close_out oc;
  printf "Color overlay image saved to %s\n" output_path;
  flush stdout

(* Calculate and print alignment statistics *)
let calculate_alignment_stats ref_data aligned_data width height =
  let diff_sum = ref 0.0 in
  let diff_sq_sum = ref 0.0 in
  let max_diff = ref 0 in
  let pixel_count = width * height in
  
  (* Calculate difference statistics *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let diff = aligned_data.(y).(x) - ref_data.(y).(x) in
      diff_sum := !diff_sum +. float_of_int diff;
      diff_sq_sum := !diff_sq_sum +. float_of_int (diff * diff);
      max_diff := max !max_diff (abs diff);
    done;
  done;
  
  let mean_diff = !diff_sum /. float_of_int pixel_count in
  let variance = !diff_sq_sum /. float_of_int pixel_count -. mean_diff *. mean_diff in
  let stddev = sqrt variance in
  
  printf "Alignment Statistics:\n";
  printf "  Mean difference: %.2f ADU\n" mean_diff;
  printf "  Standard deviation: %.2f ADU\n" stddev;
  printf "  Maximum difference: %d ADU\n" !max_diff;
  printf "  Signal-to-Noise ratio: %.2f\n" 
    (if stddev > 0.0 then abs_float mean_diff /. stddev else 0.0);
  flush stdout;
  
  (mean_diff, stddev, float_of_int !max_diff)

(* Evaluate correlation between aligned images *)
let evaluate_correlation ref_data aligned_data width height =
  (* Calculate means *)
  let ref_sum = ref 0.0 in
  let align_sum = ref 0.0 in
  let pixel_count = width * height in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      ref_sum := !ref_sum +. float_of_int ref_data.(y).(x);
      align_sum := !align_sum +. float_of_int aligned_data.(y).(x);
    done;
  done;
  
  let ref_mean = !ref_sum /. float_of_int pixel_count in
  let align_mean = !align_sum /. float_of_int pixel_count in
  
  (* Calculate correlation coefficient *)
  let numerator = ref 0.0 in
  let ref_variance = ref 0.0 in
  let align_variance = ref 0.0 in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let ref_val = float_of_int ref_data.(y).(x) -. ref_mean in
      let align_val = float_of_int aligned_data.(y).(x) -. align_mean in
      
      numerator := !numerator +. ref_val *. align_val;
      ref_variance := !ref_variance +. ref_val *. ref_val;
      align_variance := !align_variance +. align_val *. align_val;
    done;
  done;
  
  let correlation = 
    if !ref_variance > 0.0 && !align_variance > 0.0 then
      !numerator /. (sqrt !ref_variance *. sqrt !align_variance)
    else
      0.0
  in
  
  printf "Correlation coefficient: %.6f\n" correlation;
  printf "  Perfect alignment would be 1.0\n";
  printf "  Random alignment would be near 0.0\n";
  if correlation < 0.8 then
    printf "  WARNING: Correlation is poor (<0.8), alignment may have failed\n";
  flush stdout;
  
  correlation

(* Function to check alignment quality *)
let check_alignment ref_img aligned_img output_dir =
  try
    (* Read reference image *)
    let ref_hdrh, ref_contents = find_header_end ref_img (read_image ref_img) in
    let width = parse_int ref_hdrh "NAXIS1" in
    let height = parse_int ref_hdrh "NAXIS2" in
    let ref_data = read_fits_data ref_contents width height in
    
    (* Read aligned image *)
    let aligned_hdrh, aligned_contents = find_header_end aligned_img (read_image aligned_img) in
    let aligned_data = read_fits_data aligned_contents width height in
    
    (* Create output directory if it doesn't exist *)
    (try Unix.mkdir output_dir 0o755 with Unix.Unix_error(Unix.EEXIST, _, _) -> ());
    
    (* Base name for output files *)
    let ref_base = Filename.basename ref_img |> Filename.remove_extension in
    let aligned_base = Filename.basename aligned_img |> Filename.remove_extension in
    let base_name = Printf.sprintf "%s_vs_%s" ref_base aligned_base in
    
    (* Create difference image *)
    let diff_path = Filename.concat output_dir (base_name ^ "_diff.fits") in
    create_difference_image ref_data aligned_data width height diff_path;
    
    (* Create color overlay *)
    let overlay_path = Filename.concat output_dir (base_name ^ "_overlay.fits") in
    create_color_overlay ref_data aligned_data width height overlay_path;
    
    (* Calculate statistics *)
    let (mean_diff, stddev, max_diff) = calculate_alignment_stats ref_data aligned_data width height in
    
    (* Evaluate correlation *)
    let correlation = evaluate_correlation ref_data aligned_data width height in
    
    (correlation, mean_diff, stddev, max_diff)
  with e ->
    printf "Error checking alignment: %s\n" (Printexc.to_string e);
    flush stdout;
    (-1.0, 0.0, 0.0, 0.0)

(* Standalone alignment checker tool *)
let check_alignment_batch reference_image aligned_images output_dir =
  printf "Alignment Checker\n";
  printf "=================\n";
  printf "Reference image: %s\n" reference_image;
  printf "Output directory: %s\n" output_dir;
  printf "Number of images to check: %d\n\n" (List.length aligned_images);
  
  (* Create output directory if it doesn't exist *)
  (try Unix.mkdir output_dir 0o755 with Unix.Unix_error(Unix.EEXIST, _, _) -> ());
  
  (* Create CSV file for results *)
  let csv_path = Filename.concat output_dir "alignment_results.csv" in
  let csv = open_out csv_path in
  fprintf csv "Image,Correlation,MeanDiff,StdDev,MaxDiff\n";
  
  (* Track overall stats *)
  let correlation_sum = ref 0.0 in
  let count = ref 0 in
  
  (* Process each image *)
  List.iter (fun img ->
    printf "Checking alignment of %s...\n" (Filename.basename img);
    flush stdout;
    
    let (corr, mean_diff, stddev, max_diff) = check_alignment reference_image img output_dir in
    
    if corr >= 0.0 then begin
      correlation_sum := !correlation_sum +. corr;
      incr count;
      
      (* Write to CSV *)
      fprintf csv "%s,%.6f,%.2f,%.2f,%.2f\n" 
        (Filename.basename img) corr mean_diff stddev max_diff;
      flush csv;
    end
  ) aligned_images;
  
  close_out csv;
  
  (* Print summary *)
  if !count > 0 then begin
    let avg_corr = !correlation_sum /. float_of_int !count in
    
    printf "\nAlignment Results Summary:\n";
    printf "  Images analyzed: %d\n" !count;
    printf "  Average correlation: %.6f\n" avg_corr;
    
    if avg_corr < 0.8 then
      printf "  WARNING: Average correlation is poor (<0.8). Alignment may have failed.\n"
    else
      printf "  Average correlation is good (>= 0.8).\n";
    
    printf "\nResults saved to %s\n" csv_path;
  end else
    printf "\nNo valid images were processed.\n";
  
  flush stdout
