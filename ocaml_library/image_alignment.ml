(* image_alignment.ml - Module for aligning and stacking astronomical images *)

open Types
open Fits
open Printf
open Fft_alignment

(* Default parameters for star detection *)
let default_detection_params = {
  threshold = 5.0;
  min_separation = 10;
  max_stars = 1000;
}

(* Improved alignment code to insert into image_alignment.ml *)

(* Enhanced star matching with RANSAC for robust alignment *)
let match_star_patterns ref_stars target_stars max_iterations =
  if List.length ref_stars < 3 || List.length target_stars < 3 then
    None
  else
    (* Use brightest stars for initial matching *)
    let ref_sorted = List.sort (fun s1 s2 -> compare s2.flux s1.flux) ref_stars in
    let target_sorted = List.sort (fun s1 s2 -> compare s2.flux s1.flux) target_stars in
    
    (* Take top N stars *)
    let n_bright = min 30 (min (List.length ref_sorted) (List.length target_sorted)) in
    let ref_bright = Array.of_list (List.take n_bright ref_sorted) in
    let target_bright = Array.of_list (List.take n_bright target_sorted) in
    
    (* RANSAC algorithm for robust model estimation *)
    let best_model = ref None in
    let best_inliers = ref 0 in
    let best_error = ref infinity in
    
    for _ = 1 to max_iterations do
      (* Randomly select 3 pairs of stars *)
      let idx1 = Random.int n_bright in
      let idx2 = Random.int n_bright in
      let idx3 = Random.int n_bright in
      
      (* Ensure we have 3 different indices *)
      if idx1 <> idx2 && idx2 <> idx3 && idx1 <> idx3 then
        (* Get star coordinates *)
        let r1 = ref_bright.(idx1) in
        let r2 = ref_bright.(idx2) in
        let r3 = ref_bright.(idx3) in
        
        let t1 = target_bright.(idx1) in
        let t2 = target_bright.(idx2) in
        let t3 = target_bright.(idx3) in
        
        (* Calculate centroid *)
        let rx_sum = r1.x +. r2.x +. r3.x in
        let ry_sum = r1.y +. r2.y +. r3.y in
        let tx_sum = t1.x +. t2.x +. t3.x in
        let ty_sum = t1.y +. t2.y +. t3.y in
        
        let rx_center = rx_sum /. 3.0 in
        let ry_center = ry_sum /. 3.0 in
        let tx_center = tx_sum /. 3.0 in
        let ty_center = ty_sum /. 3.0 in
        
        (* Shift to center for rotation/scale calculation *)
        let r1x = r1.x -. rx_center in
        let r1y = r1.y -. ry_center in
        let r2x = r2.x -. rx_center in
        let r2y = r2.y -. ry_center in
        let r3x = r3.x -. rx_center in
        let r3y = r3.y -. ry_center in
        
        let t1x = t1.x -. tx_center in
        let t1y = t1.y -. ty_center in
        let t2x = t2.x -. tx_center in
        let t2y = t2.y -. ty_center in
        let t3x = t3.x -. tx_center in
        let t3y = t3.y -. ty_center in
        
        (* Calculate rotation and scale using least squares *)
        (* First compute the scaled rotation matrix elements *)
        let a = r1x *. t1x +. r2x *. t2x +. r3x *. t3x in
        let b = r1x *. t1y +. r2x *. t2y +. r3x *. t3y in
        let c = r1y *. t1x +. r2y *. t2x +. r3y *. t3x in
        let d = r1y *. t1y +. r2y *. t2y +. r3y *. t3y in
        
        let denominator = r1x *. r1x +. r1y *. r1y +. 
                         r2x *. r2x +. r2y *. r2y +. 
                         r3x *. r3x +. r3y *. r3y in
        
        if denominator > 0.0 then
          (* Compute rotation and scale *)
          let s_cos = (a +. d) /. denominator in
          let s_sin = (c -. b) /. denominator in
          
          let scale = sqrt (s_cos *. s_cos +. s_sin *. s_sin) in
          let rotation = atan2 s_sin s_cos in
          
          (* Compute translation *)
          let dx = tx_center -. (scale *. (rx_center *. cos rotation -. ry_center *. sin rotation)) in
          let dy = ty_center -. (scale *. (rx_center *. sin rotation +. ry_center *. cos rotation)) in
          
          (* Create transformation model *)
          let model = { dx; dy; rotation; scale } in
          
          (* Count inliers *)
          let inliers = ref 0 in
          let total_error = ref 0.0 in
          
          for i = 0 to n_bright - 1 do
            let rx = ref_bright.(i).x in
            let ry = ref_bright.(i).y in
            
            (* Apply transform *)
            let tx_rot = scale *. (rx *. cos rotation -. ry *. sin rotation) +. dx in
            let ty_rot = scale *. (rx *. sin rotation +. ry *. cos rotation) +. dy in
            
            (* Compare with actual target position *)
            let tx_actual = target_bright.(i).x in
            let ty_actual = target_bright.(i).y in
            
            let dist_sq = (tx_rot -. tx_actual) ** 2.0 +. (ty_rot -. ty_actual) ** 2.0 in
            
            (* Threshold for inliers *)
            if dist_sq < 10.0 ** 2.0 then begin
              incr inliers;
              total_error := !total_error +. sqrt dist_sq;
            end
          done;
          
          (* Update best model if better *)
          let avg_error = if !inliers > 0 then !total_error /. float_of_int !inliers else infinity in
          if !inliers > !best_inliers || (!inliers = !best_inliers && avg_error < !best_error) then begin
            best_model := Some model;
            best_inliers := !inliers;
            best_error := avg_error;
          end
    done;
    
    (* Return best model found *)
    !best_model

(* Background estimation using sigma clipping *)
let estimate_background_stats data width height =
  (* Sample a fraction of pixels for efficiency *)
  let samples = ref [] in
  let sample_rate = max 1 (min width height / 100) in
  
  for y = 0 to height - 1 do
    if y mod sample_rate = 0 then
      for x = 0 to width - 1 do
        if x mod sample_rate = 0 then
          samples := data.(y).(x) :: !samples
      done
  done;
  
  let (sorted: int array) = Array.of_list (List.sort compare !samples) in
  let n = Array.length sorted in
  
  if n = 0 then (0.0, 0.0)  (* Handle empty case *)
  else begin
    (* Initial median and MAD estimation *)
    let median = float_of_int sorted.(n/2) in
    let mad_array = Array.map (fun x -> abs_float (float_of_int x -. median)) sorted in
    Array.sort compare mad_array;
    let mad = mad_array.(n/2) in
    
    (* Sigma clip at 3σ *)
    let sigma = 1.4826 *. mad in
    let clipped = ref [] in
    Array.iter (fun x ->
        if abs_float (float_of_int x -. median) < 3.0 *. sigma then
            clipped := float_of_int x :: !clipped
    ) sorted;
    
    (* Calculate stats on clipped data *)
    let n_clipped = List.length !clipped in
    if n_clipped = 0 then (median, sigma)
    else begin
      let mean = List.fold_left (+.) 0.0 !clipped /. float_of_int n_clipped in
      let variance = List.fold_left (fun acc x ->
          let diff = x -. mean in
          acc +. (diff *. diff)
      ) 0.0 !clipped /. float_of_int (n_clipped - 1) in
      
      (mean, sqrt variance)
    end
  end

(* Simple efficient star detection *)
let detect_stars data width height params =
  (* First estimate background level and noise *)
  let bg_mean, bg_stddev = estimate_background_stats data width height in
  let detection_threshold = bg_mean +. params.threshold *. bg_stddev in
  
  printf "Background mean: %.2f, stddev: %.2f, threshold: %.2f\n" 
    bg_mean bg_stddev detection_threshold;
  
  (* Find local maxima above threshold *)
  let candidates = ref [] in
  
  (* Skip border to avoid edge effects *)
  for y = 5 to height - 6 do
    for x = 5 to width - 6 do
      let value = float_of_int data.(y).(x) in
      if value > detection_threshold then begin
        (* Check if it's a local maximum in 7x7 window *)
        let is_maximum = ref true in
        for dy = -3 to 3 do
          for dx = -3 to 3 do
            if (dx <> 0 || dy <> 0) && 
               value < float_of_int data.(y + dy).(x + dx) then
              is_maximum := false
          done
        done;
        
        if !is_maximum then
          candidates := (x, y, value) :: !candidates
      end
    done
  done;
  
  (* Sort by brightness (descending) *)
  let sorted_candidates = List.sort (fun (_, _, v1) (_, _, v2) -> 
    compare v2 v1  (* Reverse order for descending *)
  ) !candidates in
  
  (* Apply minimum separation to avoid dupicate detections *)
  let selected = ref [] in
  let used = Array.make_matrix height width false in
  
  List.iter (fun (x, y, value) ->
    (* Check if this position is already used *)
    if not used.(y).(x) then begin
      (* Mark region around this star as used *)
      for dy = -params.min_separation to params.min_separation do
        for dx = -params.min_separation to params.min_separation do
          let nx = x + dx in
          let ny = y + dy in
          if nx >= 0 && nx < width && ny >= 0 && ny < height then
            used.(ny).(nx) <- true
        done
      done;
      
      (* Estimate FWHM by measuring width at half max *)
      let half_max = (value -. bg_mean) /. 2.0 +. bg_mean in
      let radius = ref 0.0 in
      
      (* Try to measure in 4 directions *)
      let directions = [(1, 0); (0, 1); (-1, 0); (0, -1)] in
      List.iter (fun (dx, dy) ->
        let r = ref 0 in
        let nx = ref (x + dx) in
        let ny = ref (y + dy) in
        
        while !nx >= 0 && !nx < width && !ny >= 0 && !ny < height && 
              float_of_int data.(!ny).(!nx) > half_max do
          r := !r + 1;
          nx := !nx + dx;
          ny := !ny + dy;
        done;
        
        radius := !radius +. float_of_int !r
      ) directions;
      
      let fwhm = (!radius /. 2.0) *. 2.0 in  (* Average the 4 directions * 2 for diameter *)
      
      (* Add the star to our list *)
      selected := { x = float_of_int x; y = float_of_int y; flux = value -. bg_mean; fwhm = fwhm } :: !selected;
      
      (* Stop if we've found enough stars *)
      if List.length !selected >= params.max_stars then
        raise Exit
    end
  ) sorted_candidates;
  
  (* Return the stars *)
  List.rev !selected  (* Preserve brightness order *)

(* Improved alignment function that considers rotation and scale *)
let align_with_stars ref_data img_data width height =
  (* Detect stars in both images *)
  let ref_stars = detect_stars ref_data width height default_detection_params in
  let img_stars = detect_stars img_data width height default_detection_params in
  
  Printf.printf "  Detected %d stars in reference, %d stars in target\n" 
    (List.length ref_stars) (List.length img_stars);
  flush stdout;
  
  if List.length ref_stars < 3 || List.length img_stars < 3 then begin
    (* Not enough stars for robust alignment, fall back to FFT *)
    Printf.printf "  Not enough stars for robust alignment, using FFT instead\n";
    flush stdout;
    align_with_fft ref_data img_data width height
  end else begin
    (* Use RANSAC to find transformation *)
    match match_star_patterns ref_stars img_stars 200 with
    | Some params ->
        Printf.printf "  Star alignment: dx=%.2f, dy=%.2f, rotation=%.4f°, scale=%.4f\n" 
          params.dx params.dy (params.rotation *. 180.0 /. Float.pi) params.scale;
        flush stdout;
        params
    | None ->
        (* Fall back to FFT if star matching fails *)
        Printf.printf "  Star matching failed, using FFT instead\n";
        flush stdout;
        align_with_fft ref_data img_data width height
  end

(* Enhanced align_image function supporting rotation and scaling *)
let align_image src_data width height params =
  (* Create output image buffer *)
  let dest_data = Array.make_matrix height width 0 in
  
  (* Extract transformation parameters *)
  let dx = params.dx in
  let dy = params.dy in
  let rotation = params.rotation in
  let scale = params.scale in
  
  (* Check if rotation and scale are significant *)
  let simple_translation = 
    abs_float rotation < 0.001 && abs_float (scale -. 1.0) < 0.001
  in
  
  if simple_translation then begin
    (* Fast path for pure translation *)
    Printf.printf "  Using fast path for pure translation\n";
    flush stdout;
    
    (* Integer pixel shifts for translation only *)
    let dx_int = int_of_float (Float.round dx) in
    let dy_int = int_of_float (Float.round dy) in
    
    (* Fill with zeros initially *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        dest_data.(y).(x) <- 0
      done
    done;
    
    (* Copy pixels with bounds checking *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let src_x = x - dx_int in
        let src_y = y - dy_int in
        
        if src_x >= 0 && src_x < width && 
           src_y >= 0 && src_y < height then
          dest_data.(y).(x) <- src_data.(src_y).(src_x)
      done
    done
  end else begin
    (* Full transformation with rotation and scaling *)
    Printf.printf "  Using full transformation with rotation and scaling\n";
    flush stdout;
    
    (* Cache trigonometric values *)
    let cos_angle = cos (-.rotation) in
    let sin_angle = sin (-.rotation) in
    let inv_scale = 1.0 /. scale in
    
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        (* Apply inverse transformation to get source coordinates *)
        let tx = float_of_int x -. dx in
        let ty = float_of_int y -. dy in
        
        (* Apply rotation and scaling *)
        let src_x = (tx *. cos_angle -. ty *. sin_angle) *. inv_scale in
        let src_y = (tx *. sin_angle +. ty *. cos_angle) *. inv_scale in
        
        (* Bilinear interpolation *)
        let src_x_floor = floor src_x in
        let src_y_floor = floor src_y in
        let src_x_int = int_of_float src_x_floor in
        let src_y_int = int_of_float src_y_floor in
        
        let x_frac = src_x -. src_x_floor in
        let y_frac = src_y -. src_y_floor in
        
        if src_x_int >= 0 && src_x_int + 1 < width && 
           src_y_int >= 0 && src_y_int + 1 < height then begin
          (* Get the four surrounding pixels *)
          let p00 = float_of_int src_data.(src_y_int).(src_x_int) in
          let p10 = float_of_int src_data.(src_y_int).(src_x_int + 1) in
          let p01 = float_of_int src_data.(src_y_int + 1).(src_x_int) in
          let p11 = float_of_int src_data.(src_y_int + 1).(src_x_int + 1) in
          
          (* Interpolate *)
          let value = 
            p00 *. (1.0 -. x_frac) *. (1.0 -. y_frac) +.
            p10 *. x_frac *. (1.0 -. y_frac) +.
            p01 *. (1.0 -. x_frac) *. y_frac +.
            p11 *. x_frac *. y_frac
          in
          
          dest_data.(y).(x) <- int_of_float (Float.round value)
        end
      done
    done
  end;
  
  dest_data

(* Write stacked image to FITS file with updated header *)
and write_stacked_image output_path ref_hdrh data aligned_files stacking_method =
  try
    (* Create a copy of the reference header *)
    let header = Hashtbl.copy ref_hdrh in
    
    (* Update header with stacking information *)
    let method_str = match stacking_method with
      | Average -> "AVERAGE"
      | Median -> "MEDIAN"
      | SigmaClip sigma -> Printf.sprintf "SIGCLIP-%.1f" sigma
      | Kappa k -> Printf.sprintf "KAPPA-%.1f" k
      | WeightedAverage -> "WEIGHTED" 
    in
    
    Hashtbl.replace header "IMAGETYP=" "'STACKED'           / Stacked image";
    Hashtbl.replace header "NCOMBINE=" (Printf.sprintf " = %d / Number of combined frames" (Array.length aligned_files));
    Hashtbl.replace header "STACKMTD=" (Printf.sprintf "'%s'        / Stacking method" method_str);
    
    (* Add list of input files to header *)
    if false then for i = 0 to min 9 (Array.length aligned_files - 1) do
      let key = Printf.sprintf "IMGSRC%d=" i in
      let value = Printf.sprintf "'%s'" (Filename.basename aligned_files.(i)) in
      let comment = if i = 0 then " / Source images (up to 10 listed)" else "" in
      Hashtbl.replace header key (Printf.sprintf " = %s%s" value comment);
    done;
    
    if Array.length aligned_files > 10 then
      Hashtbl.replace header "NIMGSRC=" (Printf.sprintf " = %d / Total number of source images" (Array.length aligned_files));
    
    (* Get dimensions *)
    let width = parse_int header "NAXIS1" in
    let height = parse_int header "NAXIS2" in
    
    (* Open output file *)
    let oc = open_out_bin output_path in
    
    (* Write FITS header *)
    ignore (write_fits_header oc header);
    
    (* Write image data *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        (* FITS uses big-endian *)
        let value = data.(y).(x) in
        output_byte oc (value lsr 8);
        output_byte oc (value land 0xFF);
      done
    done;
    
    (* Pad data to multiple of 2880 bytes *)
    let data_size = width * height * 2 in
    let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
    output_string oc (String.make padding_size '\000');
    
    close_out oc;
    printf "Stacked image saved to %s\n" output_path;
    true
  with e ->
    printf "Error writing stacked image: %s\n" (Printexc.to_string e);
    false

(* Improved stacking function with better alignment methods *)
let stack_with_robust_alignment files reference_idx stacking_method output_path =
  Printf.printf "Stacking images with robust alignment...\n";
  flush stdout;
  
  if Array.length files = 0 then
    None  (* Return empty arrays *)
  else begin
    Printf.printf "Using %s as reference image\n" files.(reference_idx);
    flush stdout;
    
    (* Read reference image *)
    let ref_img = read_image files.(reference_idx) in
    let ref_hdrh, ref_contents = find_header_end files.(reference_idx) ref_img in
    let width = parse_int ref_hdrh "NAXIS1" in
    let height = parse_int ref_hdrh "NAXIS2" in
    
    Printf.printf "Reference image dimensions: %dx%d\n" width height;
    flush stdout;
    
    (* Read FITS data *)
    let ref_data = read_fits_data ref_contents width height in
    
    (* Array to store alignment parameters for each image *)
    let alignment_params = Array.make (Array.length files) None in
    
    (* Reference image has identity transform *)
    alignment_params.(reference_idx) <- Some identity_transform;
    
    (* Process each image *)
    for i = 0 to Array.length files - 1 do
      if i <> reference_idx then begin
        Printf.printf "Processing %s (%d/%d)...\n" 
          (Filename.basename files.(i)) (i+1) (Array.length files);
        flush stdout;
        
        try
          (* Read target image *)
          let img = read_image files.(i) in
          let hdrh, contents = find_header_end files.(i) img in
          let img_width = parse_int hdrh "NAXIS1" in
          let img_height = parse_int hdrh "NAXIS2" in
          
          (* Check dimensions match *)
          if img_width <> width || img_height <> height then begin
            Printf.printf "  Warning: Dimensions don't match reference (%dx%d vs %dx%d)\n" 
              img_width img_height width height;
            flush stdout;
            alignment_params.(i) <- None
          end else begin
            (* Read data *)
            let data = read_fits_data contents width height in
            
            (* Try both star-based and FFT alignment methods *)
            let params = align_with_stars ref_data data width height in
            
            Printf.printf "  Alignment parameters: dx=%.2f, dy=%.2f, rotation=%.4f°, scale=%.4f\n"
              params.dx params.dy (params.rotation *. 180.0 /. Float.pi) params.scale;
            flush stdout;
            alignment_params.(i) <- Some params
          end
        with e ->
          Printf.printf "  Error processing image: %s\n" (Printexc.to_string e);
          flush stdout;
          alignment_params.(i) <- None
      end
    done;
    
    (* Separate successful and failed alignments *)
    let aligned_images = ref [] in
    let aligned_data = ref [] in
    let failed_images = ref [] in
    
    for i = 0 to Array.length files - 1 do
      match alignment_params.(i) with
      | Some params -> 
          aligned_images := files.(i) :: !aligned_images;
          
          (* Align the image *)
          let img = read_image files.(i) in
          let _, contents = find_header_end files.(i) img in
          let data = read_fits_data contents width height in
          let aligned = align_image data width height params in
          aligned_data := aligned :: !aligned_data
          
      | None -> 
          if i <> reference_idx then 
            failed_images := files.(i) :: !failed_images
    done;
    
    (* Reverse to maintain original order *)
    let aligned_images = Array.of_list (List.rev !aligned_images) in
    let aligned_data = Array.of_list (List.rev !aligned_data) in
    let failed_images = Array.of_list (List.rev !failed_images) in
    
    (* Save individual aligned images for inspection *)
    let dir = Filename.dirname output_path in
    let aligned_dir = Filename.concat dir "aligned" in
    (try Unix.mkdir aligned_dir 0o755 with Unix.Unix_error(Unix.EEXIST, _, _) -> ());
    
    Printf.printf "Saving aligned images to %s\n" aligned_dir;
    flush stdout;
    
    Array.iteri (fun i img_data ->
      let basename = Filename.basename aligned_images.(i) in
      let aligned_path = Filename.concat aligned_dir ("aligned_" ^ basename) in
      
      (* Create a copy of the reference header *)
      let out_fd = open_out_bin aligned_path in
      ignore (write_fits_header out_fd ref_hdrh);
      
      (* Write data *)
      for y = 0 to height - 1 do
        for x = 0 to width - 1 do
          (* FITS uses big-endian *)
          let value = img_data.(y).(x) in
          output_byte out_fd (value lsr 8);
          output_byte out_fd (value land 0xFF);
        done
      done;
      
      (* Pad data to multiple of 2880 bytes *)
      let data_size = width * height * 2 in
      let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
      output_string out_fd (String.make padding_size '\000');
      
      close_out out_fd;
      
      Printf.printf "  Saved aligned image: %s\n" aligned_path;
      flush stdout;
    ) aligned_data;
    
    (* Stack the aligned images *)
    Printf.printf "Stacking %d aligned images...\n" (Array.length aligned_data);
    flush stdout;
    
    let output_data = Array.make_matrix height width 0 in
    
    (* Apply stacking method to each pixel *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        (* Extract values for this pixel from all images *)
        let values = Array.map (fun img -> img.(y).(x)) aligned_data in
        
        (* Apply stacking method *)
        let stacked_value = match stacking_method with
          | Average ->
              (* Calculate mean *)
              let sum = Array.fold_left (+) 0 values in
              sum / Array.length values
              
          | Median ->
              (* Calculate median *)
              let sorted = Array.copy values in
              Array.sort compare sorted;
              sorted.(Array.length sorted / 2)
              
          | SigmaClip sigma ->
              (* Mean with sigma clipping *)
              let sum = Array.fold_left (+) 0 values in
              let mean = float_of_int sum /. float_of_int (Array.length values) in
              
              (* Calculate standard deviation *)
              let variance = Array.fold_left (fun acc v ->
                  let diff = float_of_int v -. mean in
                  acc +. diff *. diff
                ) 0.0 values /. float_of_int (Array.length values) in
              let stddev = sqrt variance in
              
              (* Filter values and recalculate mean *)
              let threshold = sigma *. stddev in
              let filtered_sum = ref 0 in
              let filtered_count = ref 0 in
              Array.iter (fun v ->
                  if abs_float (float_of_int v -. mean) <= threshold then begin
                    filtered_sum := !filtered_sum + v;
                    incr filtered_count
                  end
                ) values;
              
              if !filtered_count > 0 then !filtered_sum / !filtered_count else int_of_float mean
              
          | Kappa k ->
              (* Similar to sigma clip but with robust statistics *)
              let sorted = Array.copy values in
              Array.sort compare sorted;
              
              let median = sorted.(Array.length sorted / 2) in
              let mad = Array.map (fun v -> abs (v - median)) sorted in
              Array.sort compare mad;
              let k_sigma = float_of_int (mad.(Array.length mad / 2)) *. 1.4826 *. k in
              
              let filtered_sum = ref 0 in
              let filtered_count = ref 0 in
              Array.iter (fun v ->
                  if abs (v - median) <= int_of_float k_sigma then begin
                    filtered_sum := !filtered_sum + v;
                    incr filtered_count
                  end
                ) values;
              
              if !filtered_count > 0 then !filtered_sum / !filtered_count else median
              
          | WeightedAverage ->
              (* Enhanced weighted average based on image quality *)
              (* In a real implementation, weights would be based on metrics like FWHM *)
              (* For now, just use unweighted mean *)
              let sum = Array.fold_left (+) 0 values in
              sum / Array.length values
        in
        
        output_data.(y).(x) <- stacked_value
      done;
      
      (* Print progress for large images *)
      if height > 1000 && y mod 100 = 0 then begin
        Printf.printf "  Stacking progress: %.1f%%\n" (float_of_int y *. 100.0 /. float_of_int height);
        flush stdout;
      end
    done;
    
    (* Write output FITS file *)
    let success = write_stacked_image output_path ref_hdrh output_data 
                    aligned_images stacking_method in
    
    if success then
      Some {
        reference_image = files.(reference_idx);
        aligned_images;
        failed_images;
        stacking_method;
        output_file = output_path;
      }
    else
      None
  end

(* Calculate alignment parameters between two sets of star positions with improved robustness *)
let calculate_alignment_parameters reference_stars target_stars =
  (* We need at least 3 matching stars to determine transformation reliably *)
  if List.length reference_stars < 3 || List.length target_stars < 3 then
    None
  else begin
    (* For simplicity, we'll use the brightest stars for matching *)
    let ref_sorted = List.sort (fun s1 s2 -> compare s2.flux s1.flux) reference_stars in
    let target_sorted = List.sort (fun s1 s2 -> compare s2.flux s1.flux) target_stars in
    
    (* Take the top N stars from each list *)
    let n_match = min 20 (min (List.length ref_sorted) (List.length target_sorted)) in
    let ref_top = Array.of_list (List.take n_match ref_sorted) in
    let target_top = Array.of_list (List.take n_match target_sorted) in
    
    (* Calculate centroids *)
    let ref_x_sum = ref 0.0 in
    let ref_y_sum = ref 0.0 in
    let target_x_sum = ref 0.0 in
    let target_y_sum = ref 0.0 in
    
    for i = 0 to n_match - 1 do
      ref_x_sum := !ref_x_sum +. ref_top.(i).x;
      ref_y_sum := !ref_y_sum +. ref_top.(i).y;
      target_x_sum := !target_x_sum +. target_top.(i).x;
      target_y_sum := !target_y_sum +. target_top.(i).y;
    done;
    
    let ref_centroid_x = !ref_x_sum /. float_of_int n_match in
    let ref_centroid_y = !ref_y_sum /. float_of_int n_match in
    let target_centroid_x = !target_x_sum /. float_of_int n_match in
    let target_centroid_y = !target_y_sum /. float_of_int n_match in
    
    (* Calculate dx and dy more precisely using least squares estimation *)
    let dx = target_centroid_x -. ref_centroid_x in
    let dy = target_centroid_y -. ref_centroid_y in
    
    (* Refine the estimate by minimizing distance between matched stars *)
    let refine_translation () =
      let dx_sum = ref 0.0 in
      let dy_sum = ref 0.0 in
      let count = ref 0 in
      
      (* For each reference star, find the closest target star after initial translation *)
      for i = 0 to n_match - 1 do
        let ref_x = ref_top.(i).x in
        let ref_y = ref_top.(i).y in
        
        let min_dist = ref infinity in
        let best_match = ref (-1) in
        
        (* Find closest target star *)
        for j = 0 to n_match - 1 do
          let target_x = target_top.(j).x -. dx in (* Apply initial translation *)
          let target_y = target_top.(j).y -. dy in
          
          let dist_sq = (target_x -. ref_x) ** 2.0 +. (target_y -. ref_y) ** 2.0 in
          if dist_sq < !min_dist then begin
            min_dist := dist_sq;
            best_match := j;
          end
        done;
        
        (* If good match found (distance less than threshold) *)
        if !min_dist < 100.0 && !best_match >= 0 then begin
          let target_x = target_top.(!best_match).x in
          let target_y = target_top.(!best_match).y in
          
          dx_sum := !dx_sum +. (target_x -. ref_x);
          dy_sum := !dy_sum +. (target_y -. ref_y);
          incr count;
        end
      done;
      
      (* Return refined displacement if we have enough matches *)
      if !count >= 3 then
        (!dx_sum /. float_of_int !count, !dy_sum /. float_of_int !count)
      else
        (dx, dy) (* Fall back to centroid-based estimate *)
    in
    
    let final_dx, final_dy = refine_translation () in
    
    Some { dx = final_dx; dy = final_dy; rotation = 0.0; scale = 1.0 }
  end

(* Apply alignment transform to coordinates *)
let transform_coordinates x y params =
  (* Shift to origin for rotation and scaling *)
  let x0 = x in
  let y0 = y in
  
  (* Apply rotation and scaling *)
  let cos_angle = cos params.rotation in
  let sin_angle = sin params.rotation in
  let x1 = (x0 *. cos_angle -. y0 *. sin_angle) *. params.scale in
  let y1 = (x0 *. sin_angle +. y0 *. cos_angle) *. params.scale in
  
  (* Apply translation *)
  let x2 = x1 +. params.dx in
  let y2 = y1 +. params.dy in
  
  (x2, y2)
(* Apply alignment to an image with improved interpolation *)
let align_image src_data width height params =
  (* Create output image buffer *)
  let dest_data = Array.make_matrix height width 0 in
  
  (* Extract transformation parameters *)
  let dx = params.dx in
  let dy = params.dy in
  let rotation = params.rotation in
  let scale = params.scale in
  
  (* Fast path for pure translation (no rotation or scaling) *)
  if rotation = 0.0 && scale = 1.0 then begin
    (* Integer pixel shifts for translation only *)
    let dx_int = int_of_float (Float.round dx) in
    let dy_int = int_of_float (Float.round dy) in
    
    (* Fill with zeros initially *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        dest_data.(y).(x) <- 0
      done
    done;
    
    (* Copy pixels with bounds checking *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let src_x = x - dx_int in
        let src_y = y - dy_int in
        
        if src_x >= 0 && src_x < width && 
           src_y >= 0 && src_y < height then
          dest_data.(y).(x) <- src_data.(src_y).(src_x)
      done
    done
  end else begin
    (* Full transformation with rotation and scaling *)
    let cos_angle = cos (-.rotation) in
    let sin_angle = sin (-.rotation) in
    let inv_scale = 1.0 /. scale in
    
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        (* Apply inverse transformation to get source coordinates *)
        let tx = float_of_int x -. dx in
        let ty = float_of_int y -. dy in
        
        (* Apply rotation and scaling *)
        let src_x = (tx *. cos_angle -. ty *. sin_angle) *. inv_scale in
        let src_y = (tx *. sin_angle +. ty *. cos_angle) *. inv_scale in
        
        (* Bilinear interpolation *)
        let src_x_floor = floor src_x in
        let src_y_floor = floor src_y in
        let src_x_int = int_of_float src_x_floor in
        let src_y_int = int_of_float src_y_floor in
        
        let x_frac = src_x -. src_x_floor in
        let y_frac = src_y -. src_y_floor in
        
        if src_x_int >= 0 && src_x_int + 1 < width && 
           src_y_int >= 0 && src_y_int + 1 < height then
          (* Get the four surrounding pixels *)
          let p00 = float_of_int src_data.(src_y_int).(src_x_int) in
          let p10 = float_of_int src_data.(src_y_int).(src_x_int + 1) in
          let p01 = float_of_int src_data.(src_y_int + 1).(src_x_int) in
          let p11 = float_of_int src_data.(src_y_int + 1).(src_x_int + 1) in
          
          (* Interpolate *)
          let value = 
            p00 *. (1.0 -. x_frac) *. (1.0 -. y_frac) +.
            p10 *. x_frac *. (1.0 -. y_frac) +.
            p01 *. (1.0 -. x_frac) *. y_frac +.
            p11 *. x_frac *. y_frac
          in
          
          dest_data.(y).(x) <- int_of_float (Float.round value)
      done
    done
  end;
  
  dest_data

(* Align all images to a reference image *)
let align_images_by_star files reference_idx detection_params =
  if Array.length files = 0 then
    [||], [||]  (* Return empty arrays *)
  else begin
    printf "Using %s as reference image\n" files.(reference_idx);
    
    (* Read reference image *)
    let ref_img = read_image files.(reference_idx) in
    let ref_hdrh, ref_contents = find_header_end files.(reference_idx) ref_img in
    let width = parse_int ref_hdrh "NAXIS1" in
    let height = parse_int ref_hdrh "NAXIS2" in
    
    printf "Reference image dimensions: %dx%d\n" width height;
    
    (* Read FITS data *)
    let ref_data = read_fits_data ref_contents width height in
    
    (* Detect stars in reference image *)
    let ref_stars = detect_stars ref_data width height detection_params in
    
    printf "Detected %d stars in reference image\n" (List.length ref_stars);
    
    (* Array to store alignment parameters for each image *)
    let alignment_params = Array.make (Array.length files) None in
    
    (* Reference image has identity transform *)
    alignment_params.(reference_idx) <- Some identity_transform;
    
    (* Process each image *)
    for i = 0 to Array.length files - 1 do
      if i <> reference_idx then begin
        printf "Processing %s (%d/%d)...\n" 
          (Filename.basename files.(i)) (i+1) (Array.length files);
        
        try
          (* Read target image *)
          let img = read_image files.(i) in
          let hdrh, contents = find_header_end files.(i) img in
          let img_width = parse_int hdrh "NAXIS1" in
          let img_height = parse_int hdrh "NAXIS2" in
          
          (* Check dimensions match *)
          if img_width <> width || img_height <> height then begin
            printf "  Warning: Dimensions don't match reference (%dx%d vs %dx%d)\n" 
              img_width img_height width height;
            alignment_params.(i) <- None
          end else begin
            (* Read data and detect stars *)
            let data = read_fits_data contents width height in
            let stars = detect_stars data width height detection_params in
            
            printf "  Detected %d stars\n" (List.length stars);
            
            (* Calculate alignment parameters *)
            match calculate_alignment_parameters ref_stars stars with
            | Some params ->
                printf "  Alignment parameters: dx=%.2f, dy=%.2f, rotation=%.4f, scale=%.4f\n"
                  params.dx params.dy params.rotation params.scale;
                alignment_params.(i) <- Some params
            | None ->
                printf "  Failed to calculate alignment parameters\n";
                alignment_params.(i) <- None
          end
        with e ->
          printf "  Error processing image: %s\n" (Printexc.to_string e);
          alignment_params.(i) <- None
      end
    done;
    
    (* Separate successful and failed alignments *)
    let aligned_images = ref [] in
    let failed_images = ref [] in
    
    for i = 0 to Array.length files - 1 do
      match alignment_params.(i) with
      | Some _ -> aligned_images := files.(i) :: !aligned_images
      | None -> if i <> reference_idx then failed_images := files.(i) :: !failed_images
    done;
    
    (* Return as arrays *)
    Array.of_list (List.rev !aligned_images), 
    Array.of_list (List.rev !failed_images)
  end

let align_images = Lacaml_only_alignment.align_images

(* Stack images using given method *)
let stack_images files reference_idx stacking_method output_path =
  if Array.length files = 0 then
    None
  else begin
    (* First align all images *)
    let aligned_files, failed_files = 
      align_images files reference_idx default_detection_params in
    
    if Array.length aligned_files = 0 then begin
      printf "No images could be aligned successfully\n";
      None
    end else begin
      printf "Successfully aligned %d images, %d failed\n" 
        (Array.length aligned_files) (Array.length failed_files);
      
      (* Read reference image to get dimensions and header *)
      let ref_img = read_image files.(reference_idx) in
      let ref_hdrh, ref_contents = find_header_end files.(reference_idx) ref_img in
      let width = parse_int ref_hdrh "NAXIS1" in
      let height = parse_int ref_hdrh "NAXIS2" in
      
      printf "Image dimensions: %dx%d\n" width height;
      
      (* Create buffer for accumulating results *)
      let stack_buffer = Array.make_matrix height width [||] in
      
      (* Initialize the buffer with arrays *)
      for y = 0 to height - 1 do
        for x = 0 to width - 1 do
          stack_buffer.(y).(x) <- Array.make (Array.length aligned_files) 0
        done
      done;
      
      (* Process each aligned image *)
      for i = 0 to Array.length aligned_files - 1 do
        printf "Processing %s for stacking (%d/%d)...\n" 
          (Filename.basename aligned_files.(i)) (i+1) (Array.length aligned_files);
        
        try
          (* Read image data *)
          let img = read_image aligned_files.(i) in
          let hdrh, contents = find_header_end aligned_files.(i) img in
          let data = read_fits_data contents width height in
          
          (* Add to stack buffer *)
          for y = 0 to height - 1 do
            for x = 0 to width - 1 do
              stack_buffer.(y).(x).(i) <- data.(y).(x)
            done
          done
        with e ->
          printf "  Error reading image for stacking: %s\n" (Printexc.to_string e);
      done;
      
      (* Create output image based on stacking method *)
      let output_data = Array.make_matrix height width 0 in
      
      let apply_method values =
        match stacking_method with
        | Average ->
            (* Calculate mean *)
            let sum = ref 0 in
            let count = ref 0 in
            Array.iter (fun v -> 
              sum := !sum + v;
              incr count
            ) values;
            if !count > 0 then !sum / !count else 0
            
        | Median ->
            (* Calculate median *)
            let sorted = Array.copy values in
            Array.sort compare sorted;
            sorted.(Array.length sorted / 2)
            
        | SigmaClip sigma ->
            (* Mean with sigma clipping *)
            let sum = ref 0 in
            let count = ref 0 in
            Array.iter (fun v -> 
              sum := !sum + v;
              incr count
            ) values;
            let mean = if !count > 0 then float_of_int !sum /. float_of_int !count else 0.0 in
            
            (* Calculate standard deviation *)
            let variance = ref 0.0 in
            Array.iter (fun v ->
              let diff = float_of_int v -. mean in
              variance := !variance +. diff *. diff
            ) values;
            let stddev = sqrt (!variance /. float_of_int !count) in
            
            (* Filter values and recalculate mean *)
            let threshold = sigma *. stddev in
            let filtered_sum = ref 0 in
            let filtered_count = ref 0 in
            Array.iter (fun v ->
              if abs_float (float_of_int v -. mean) <= threshold then begin
                filtered_sum := !filtered_sum + v;
                incr filtered_count
              end
            ) values;
            
            if !filtered_count > 0 then !filtered_sum / !filtered_count else 0
            
        | Kappa k ->
            (* Similar to sigma clip but with different threshold calculation *)
            (* For now, implement same as sigma clip *)
            let sum = ref 0 in
            let count = ref 0 in
            Array.iter (fun v -> 
              sum := !sum + v;
              incr count
            ) values;
            let mean = if !count > 0 then float_of_int !sum /. float_of_int !count else 0.0 in
            
            (* Calculate standard deviation *)
            let variance = ref 0.0 in
            Array.iter (fun v ->
              let diff = float_of_int v -. mean in
              variance := !variance +. diff *. diff
            ) values;
            let stddev = sqrt (!variance /. float_of_int !count) in
            
            (* Filter values and recalculate mean *)
            let threshold = k *. stddev in
            let filtered_sum = ref 0 in
            let filtered_count = ref 0 in
            Array.iter (fun v ->
              if abs_float (float_of_int v -. mean) <= threshold then begin
                filtered_sum := !filtered_sum + v;
                incr filtered_count
              end
            ) values;
            
            if !filtered_count > 0 then !filtered_sum / !filtered_count else 0
            
        | WeightedAverage ->
            (* Simple implementation - all weights equal for now *)
            (* A real implementation would weight based on image quality metrics *)
            let sum = ref 0 in
            let count = ref 0 in
            Array.iter (fun v -> 
              sum := !sum + v;
              incr count
            ) values;
            if !count > 0 then !sum / !count else 0
      in
      
      (* Apply stacking method to each pixel *)
      for y = 0 to height - 1 do
        for x = 0 to width - 1 do
          output_data.(y).(x) <- apply_method stack_buffer.(y).(x)
        done
      done;
      
      (* Write output FITS file *)
      let success = write_stacked_image output_path ref_hdrh output_data 
                      aligned_files stacking_method in
      
      if success then
        Some {
          reference_image = files.(reference_idx);
          aligned_images = aligned_files;
          failed_images = failed_files;
          stacking_method = stacking_method;
          output_file = output_path;
        }
      else
        None
    end
  end

(* Specialized function to write RGB FITS file with careful handling of the 3 planes *)
let write_rgb_fits output_path hdrh rgb_data aligned_files stacking_method =
  try
    (* Create a copy of the reference header *)
    let header = Hashtbl.copy hdrh in
    
    (* Get image dimensions *)
    let height = Array.length rgb_data in
    let width = if height > 0 then Array.length rgb_data.(0) else 0 in
    
    printf "Writing RGB FITS with dimensions: %dx%d, 3 planes\n" width height;
    
    (* Update header for RGB FITS format *)
    Hashtbl.replace header "NAXIS" (sprintf " = 3 / Number of data axes");
    Hashtbl.replace header "NAXIS1" (sprintf " = %d / Width in pixels" width);
    Hashtbl.replace header "NAXIS2" (sprintf " = %d / Height in pixels" height);
    Hashtbl.replace header "NAXIS3" (sprintf " = 3 / Number of color planes (RGB)");
    Hashtbl.replace header "BITPIX" (sprintf " = 16 / 16-bit integers");
    Hashtbl.replace header "BZERO" (sprintf " = 32768 / Offset to unsigned short range");
    Hashtbl.replace header "BSCALE" (sprintf " = 1 / Default scaling factor");
    
    (* Make sure we have a simpler header without any content that's too large *)
    let clean_header = Hashtbl.create (Hashtbl.length header) in
    Hashtbl.iter (fun key value ->
      (* Only keep values that aren't excessively long *)
      if String.length value < 70 then
        Hashtbl.add clean_header key value
    ) header;
    
    (* Add stacking information *)
    let method_str = match stacking_method with
      | Average -> "AVERAGE"
      | Median -> "MEDIAN"
      | SigmaClip sigma -> Printf.sprintf "SIGCLIP-%.1f" sigma
      | Kappa k -> Printf.sprintf "KAPPA-%.1f" k
      | WeightedAverage -> "WEIGHTED" 
    in
    
    Hashtbl.replace clean_header "IMAGETYP=" (Printf.sprintf "'STACKED'           / Stacked image");
    Hashtbl.replace clean_header "NCOMBINE=" (Printf.sprintf " = %d / Number of combined frames" (Array.length aligned_files));
    Hashtbl.replace clean_header "STACKMTD=" (Printf.sprintf "'%s'        / Stacking method" method_str);
    
    (* List source files - just a few to avoid overly large headers *)
    let max_files = min 5 (Array.length aligned_files) in
    for i = 0 to max_files - 1 do
      let key = Printf.sprintf "IMGSRC%d=" i in
      let basename = Filename.basename aligned_files.(i) in
      let max_len = min (String.length basename) 50 in 
      let short_name = String.sub basename 0 max_len in
      Hashtbl.replace clean_header key (Printf.sprintf " = '%s'" short_name);
    done;
    
    if Array.length aligned_files > max_files then
      Hashtbl.replace clean_header "NIMGSRC=" (Printf.sprintf " = %d / Total number of source images" (Array.length aligned_files));
    
    (* Open output file *)
    let out_fd = open_out_bin output_path in
    
    (* Write the header *)
    let header_size = write_fits_header out_fd clean_header in
    printf "Wrote header of size %d bytes\n" header_size;
    
    (* Function to write a single plane with robust error handling *)
    let write_plane get_value =
      (* Use a fixed buffer size for safety *)
      let buffer_size = width * 2 in
      let buffer = Bytes.create buffer_size in
      
      for y = 0 to height - 1 do
        (* Reset buffer for each row *)
        Bytes.fill buffer 0 buffer_size '\000';
        
        for x = 0 to width - 1 do
          let value = get_value rgb_data.(y).(x) in
          let value = min 65535 (max 0 value) in
          
          (* Set bytes with bounds checking *)
          if x * 2 + 1 < buffer_size then begin
            Bytes.set buffer (x * 2) (char_of_int (value lsr 8));
            Bytes.set buffer (x * 2 + 1) (char_of_int (value land 0xFF));
          end
        done;
        
        output out_fd buffer 0 (min buffer_size (width * 2));
      done
    in
    
    (* Write each plane separately with robust error handling *)
    printf "Writing red plane...\n";
    (try write_plane (fun (r, _, _) -> r) with e -> 
       printf "Error writing red plane: %s\n" (Printexc.to_string e));
    
    printf "Writing green plane...\n";
    (try write_plane (fun (_, g, _) -> g) with e -> 
       printf "Error writing green plane: %s\n" (Printexc.to_string e));
    
    printf "Writing blue plane...\n";
    (try write_plane (fun (_, _, b) -> b) with e -> 
       printf "Error writing blue plane: %s\n" (Printexc.to_string e));
    
    (* Pad data to multiple of 2880 bytes *)
    let data_size = width * height * 2 * 3 in
    let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
    if padding_size > 0 then begin
      printf "Adding %d bytes of padding\n" padding_size;
      let padding = Bytes.create (min padding_size 2880) in
      Bytes.fill padding 0 (Bytes.length padding) '\000';
      output out_fd padding 0 (Bytes.length padding);
    end;
    
    close_out out_fd;
    printf "Successfully wrote RGB FITS to %s\n" output_path;
    true
  with e ->
    printf "Error writing RGB FITS: %s\n" (Printexc.to_string e);
    close_out_noerr (try open_out_bin output_path with _ -> stdout);
    false

(* Helper function to create RGB data array from monochrome images *)
let create_rgb_from_mono r_data g_data b_data width height =
  let rgb_data = Array.make_matrix height width (0, 0, 0) in
  
  (* Debug statistics for the mono channels *)
  let r_sum = ref 0 in
  let g_sum = ref 0 in
  let b_sum = ref 0 in
  let count = width * height in
  
  (* Fill the RGB array *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let r = r_data.(y).(x) in
      let g = g_data.(y).(x) in
      let b = b_data.(y).(x) in
      rgb_data.(y).(x) <- (r, g, b);
      r_sum := !r_sum + r;
      g_sum := !g_sum + g;
      b_sum := !b_sum + b;
    done
  done;
  
  (* Print channel statistics *)
  let r_avg = !r_sum / count in
  let g_avg = !g_sum / count in
  let b_avg = !b_sum / count in
  printf "Channel averages: R=%d, G=%d, B=%d\n" r_avg g_avg b_avg;
  
  rgb_data

(* Stack RGB image sequence (three grayscale images per frame) *)
let stack_rgb_images r_files g_files b_files reference_idx stacking_method output_path =
  if Array.length r_files = 0 || 
     Array.length g_files = 0 || 
     Array.length b_files = 0 then begin
    printf "Missing files for one or more color channels\n";
    None
  end else if Array.length r_files <> Array.length g_files || 
             Array.length r_files <> Array.length b_files then begin
    printf "Mismatch in number of files per channel: R=%d, G=%d, B=%d\n"
           (Array.length r_files) (Array.length g_files) (Array.length b_files);
    None
  end else begin
    (* Stack each channel separately *)
    printf "Stacking red channel...\n";
    let r_result = stack_images r_files reference_idx stacking_method (output_path ^ ".r.fits") in
    
    printf "Stacking green channel...\n";
    let g_result = stack_images g_files reference_idx stacking_method (output_path ^ ".g.fits") in
    
    printf "Stacking blue channel...\n";
    let b_result = stack_images b_files reference_idx stacking_method (output_path ^ ".b.fits") in
    
    match r_result, g_result, b_result with
    | Some r, Some g, Some b ->
        (* Read the stacked channel images *)
        printf "Combining RGB channels...\n";
        
        (* Read red channel data *)
        let r_img = read_image (output_path ^ ".r.fits") in
        let r_hdrh, r_contents = find_header_end (output_path ^ ".r.fits") r_img in
        let width = parse_int r_hdrh "NAXIS1" in
        let height = parse_int r_hdrh "NAXIS2" in
        let r_data = read_fits_data r_contents width height in
        
        (* Read green channel data *)
        let g_img = read_image (output_path ^ ".g.fits") in
        let g_hdrh, g_contents = find_header_end (output_path ^ ".g.fits") g_img in
        let g_data = read_fits_data g_contents width height in
        
        (* Read blue channel data *)
        let b_img = read_image (output_path ^ ".b.fits") in
        let b_hdrh, b_contents = find_header_end (output_path ^ ".b.fits") b_img in
        let b_data = read_fits_data b_contents width height in
        
        (* Debug information for channel data *)
        printf "Channel data information:\n";
        printf "  Red channel - min: %d, max: %d\n" 
          (Array.fold_left (fun min_val row -> 
            Array.fold_left (fun m v -> if v < m then v else m) min_val row) 65535 r_data)
          (Array.fold_left (fun max_val row -> 
            Array.fold_left (fun m v -> if v > m then v else m) max_val row) 0 r_data);
        printf "  Green channel - min: %d, max: %d\n" 
          (Array.fold_left (fun min_val row -> 
            Array.fold_left (fun m v -> if v < m then v else m) min_val row) 65535 g_data)
          (Array.fold_left (fun max_val row -> 
            Array.fold_left (fun m v -> if v > m then v else m) max_val row) 0 g_data);
        printf "  Blue channel - min: %d, max: %d\n" 
          (Array.fold_left (fun min_val row -> 
            Array.fold_left (fun m v -> if v < m then v else m) min_val row) 65535 b_data)
          (Array.fold_left (fun max_val row -> 
            Array.fold_left (fun m v -> if v > m then v else m) max_val row) 0 b_data);
        
        (* Combine into RGB array *)
        let rgb_data = create_rgb_from_mono r_data g_data b_data width height in
        
        (* Debug sample of RGB pixels *)
        printf "RGB combined data samples (first 5x5 pixels):\n";
        for y = 0 to min 4 (height - 1) do
          printf "  Row %d:" y;
          for x = 0 to min 4 (width - 1) do
            let (r, g, b) = rgb_data.(y).(x) in
            printf " (%d,%d,%d)" r g b;
          done;
          printf "\n";
        done;
        
        (* Write combined RGB FITS - enhanced version *)
        let success = write_rgb_fits output_path r_hdrh rgb_data r.aligned_images stacking_method in
        
        if success then
          Some {
            reference_image = r.reference_image;
            aligned_images = r.aligned_images;
            failed_images = Array.append r.failed_images 
                             (Array.append g.failed_images b.failed_images);
            stacking_method = stacking_method;
            output_file = output_path;
          }
        else
          None
    | _ ->
        printf "One or more color channels failed to stack\n";
        None
  end

let debug_rgb_data width height rgb_data =
        
        (* Debug RGB data *)
        printf "Bayer debayering produced RGB data\n";
        let r_sum = ref 0 in
        let g_sum = ref 0 in
        let b_sum = ref 0 in
        let count = width * height / 4 in  (* 2x2 binning reduces dimensions *)
        
        for y = 0 to Array.length rgb_data - 1 do
          for x = 0 to Array.length rgb_data.(0) - 1 do
            let (r, g, b) = rgb_data.(y).(x) in
            r_sum := !r_sum + r;
            g_sum := !g_sum + g;
            b_sum := !b_sum + b;
          done
        done;
        
        if count > 0 then
          printf "RGB averages: R=%d, G=%d, B=%d\n" 
            (!r_sum / count) (!g_sum / count) (!b_sum / count);
        
        (* Sample a few pixels *)
        let rgb_height = Array.length rgb_data in
        let rgb_width = Array.length rgb_data.(0) in
        printf "Debayered dimensions: %dx%d\n" rgb_width rgb_height;
        
        if rgb_height > 0 && rgb_width > 0 then begin
          printf "Sample pixels:\n";
          for y = 0 to min 2 (rgb_height - 1) do
            for x = 0 to min 2 (rgb_width - 1) do
              let (r, g, b) = rgb_data.(y).(x) in
              printf "  (%d,%d): R=%d, G=%d, B=%d\n" x y r g b;
            done
          done
        end        

(* Stack aligned frames from a Bayer RGB image sequence *)
let stack_bayer_images files reference_idx pattern stacking_method output_path =
  if Array.length files = 0 then
    None
  else begin
    (* First align all images *)
    let aligned_files, failed_files = 
      align_images files reference_idx default_detection_params in
    
    if Array.length aligned_files = 0 then begin
      printf "No images could be aligned successfully\n";
      None
    end else begin
      printf "Successfully aligned %d images, %d failed\n" 
        (Array.length aligned_files) (Array.length failed_files);
      
      (* Read reference image to get dimensions and header *)
      let ref_img = read_image files.(reference_idx) in
      let ref_hdrh, ref_contents = find_header_end files.(reference_idx) ref_img in
      let width = parse_int ref_hdrh "NAXIS1" in
      let height = parse_int ref_hdrh "NAXIS2" in
      
      printf "Image dimensions: %dx%d\n" width height;
      
      (* Create buffer for accumulating results *)
      let stack_buffer = Array.make_matrix height width [||] in
      
      (* Initialize the buffer with arrays *)
      for y = 0 to height - 1 do
        for x = 0 to width - 1 do
          stack_buffer.(y).(x) <- Array.make (Array.length aligned_files) 0
        done
      done;
      
      (* Process each aligned image *)
      for i = 0 to Array.length aligned_files - 1 do
        printf "Processing %s for stacking (%d/%d)...\n" 
          (Filename.basename aligned_files.(i)) (i+1) (Array.length aligned_files);
        
        try
          (* Read image data *)
          let img = read_image aligned_files.(i) in
          let hdrh, contents = find_header_end aligned_files.(i) img in
          let data = read_fits_data contents width height in
          
          (* Add to stack buffer *)
          for y = 0 to height - 1 do
            for x = 0 to width - 1 do
              stack_buffer.(y).(x).(i) <- data.(y).(x)
            done
          done
        with e ->
          printf "  Error reading image for stacking: %s\n" (Printexc.to_string e);
      done;
      
      (* Create output monochrome image based on stacking method *)
      let output_data = Array.make_matrix height width 0 in
      
      let apply_method values =
        match stacking_method with
        | Average ->
            (* Calculate mean *)
            let sum = ref 0 in
            let count = ref 0 in
            Array.iter (fun v -> 
              sum := !sum + v;
              incr count
            ) values;
            if !count > 0 then !sum / !count else 0
            
        | Median ->
            (* Calculate median *)
            let sorted = Array.copy values in
            Array.sort compare sorted;
            sorted.(Array.length sorted / 2)
            
        | SigmaClip sigma ->
            (* Mean with sigma clipping *)
            let sum = ref 0 in
            let count = ref 0 in
            Array.iter (fun v -> 
              sum := !sum + v;
              incr count
            ) values;
            let mean = if !count > 0 then float_of_int !sum /. float_of_int !count else 0.0 in
            
            (* Calculate standard deviation *)
            let variance = ref 0.0 in
            Array.iter (fun v ->
              let diff = float_of_int v -. mean in
              variance := !variance +. diff *. diff
            ) values;
            let stddev = sqrt (!variance /. float_of_int !count) in
            
            (* Filter values and recalculate mean *)
            let threshold = sigma *. stddev in
            let filtered_sum = ref 0 in
            let filtered_count = ref 0 in
            Array.iter (fun v ->
              if abs_float (float_of_int v -. mean) <= threshold then begin
                filtered_sum := !filtered_sum + v;
                incr filtered_count
              end
            ) values;
            
            if !filtered_count > 0 then !filtered_sum / !filtered_count else 0
            
        | Kappa k -> 
            (* Similar to sigma clip for now *)
            let sorted = Array.copy values in
            Array.sort compare sorted;
            
            let median = sorted.(Array.length sorted / 2) in
            let mad = Array.map (fun v -> abs (v - median)) sorted in
            Array.sort compare mad;
            let k_sigma = float_of_int (mad.(Array.length mad / 2)) *. 1.4826 *. k in
            
            let filtered_sum = ref 0 in
            let filtered_count = ref 0 in
            Array.iter (fun v ->
              if abs (v - median) <= int_of_float k_sigma then begin
                filtered_sum := !filtered_sum + v;
                incr filtered_count
              end
            ) values;
            
            if !filtered_count > 0 then !filtered_sum / !filtered_count else median
            
        | WeightedAverage ->
            (* Simple implementation - all weights equal for now *)
            let sum = ref 0 in
            let count = ref 0 in
            Array.iter (fun v -> 
              sum := !sum + v;
              incr count
            ) values;
            if !count > 0 then !sum / !count else 0
      in
      
      (* Apply stacking stacking_method to each pixel *)
      for y = 0 to height - 1 do
        for x = 0 to width - 1 do
          output_data.(y).(x) <- apply_method stack_buffer.(y).(x)
        done
      done;
      
      (* First save the stacked monochrome image *)
      let mono_output_path = Filename.remove_extension output_path ^ "_mono.fits" in
      let mono_success = write_stacked_image mono_output_path ref_hdrh output_data 
                      aligned_files stacking_method in
      
      (* Now debayer the stacked image to get RGB *)
      if mono_success then begin
        printf "Creating RGB image from debayered stacked image...\n";
        
        let rgb_data = (function
          | Some pattern ->
              printf "Using %s Bayer pattern\n" (Debayer_integration.describe_bayer_pattern pattern);
              Debayer_integration.bin_bayer_pattern output_data width height (Some pattern)
          | None ->
              (* Default to RGGB if not specified *)
              printf "No Bayer pattern specified, defaulting to RGGB\n";
              Debayer_integration.bin_bayer_pattern output_data width height (Some `RGGB)) pattern
        in
        
        (* Write the RGB FITS *)
        if write_rgb_fits output_path ref_hdrh rgb_data aligned_files stacking_method then
          Some {
            reference_image = files.(reference_idx);
            aligned_images = aligned_files;
            failed_images = failed_files;
            stacking_method = stacking_method;
            output_file = output_path;
          }
        else
          None
      end else begin
        printf "Failed to write monochrome stacked image\n";
        None
      end
    end
  end

(* Main stacking function with auto-detection of image type *)
let stack_auto files reference_idx stacking_method output_path =
  if Array.length files = 0 then begin
    printf "No files to stack\n";
    None
  end else begin
    (* Read the first image to determine type *)
    let img = read_image files.(0) in
    let hdrh, contents = find_header_end files.(0) img in
    
    (* Check for Bayer pattern *)
    let bayer_pattern = Debayer_integration.get_bayer_pattern hdrh in
    
    (* Check for color planes - enhanced detection *)
    let naxis = parse_int hdrh "NAXIS" in
    printf "Image type detection: NAXIS=%d\n" naxis;
    
    (* Look for NAXIS3 if NAXIS=3 - this confirms it's color *)
    let naxis3 = if naxis = 3 then 
                   try parse_int hdrh "NAXIS3" 
                   with _ -> 1
                 else 1 in
    printf "Image type detection: NAXIS3=%d\n" naxis3;
    
    (* More robust check for color images *)
    let is_color = (naxis = 3 && naxis3 = 3) || 
                   (try Hashtbl.mem hdrh "COLORIMG=" with _ -> false) in
    
    (* Debug print image metadata *)
    printf "First image metadata:\n";
    printf "  Filename: %s\n" (Filename.basename files.(0));
    printf "  Dimensions: %dx%d\n" (parse_int hdrh "NAXIS1") (parse_int hdrh "NAXIS2");
    printf "  NAXIS: %d\n" naxis;
    printf "  Is color: %b\n" is_color;
    printf "  Bayer pattern: %s\n" 
      (match bayer_pattern with 
       | Some p -> Debayer_integration.describe_bayer_pattern p 
       | None -> "None");
    
    if is_color then begin
      printf "Detected color image (NAXIS=%d, NAXIS3=%d), stacking as RGB\n" naxis naxis3;
      printf "Using specialized RGB stacking path\n";
      
      (* Read dimensions from first file *)
      let width = parse_int hdrh "NAXIS1" in
      let height = parse_int hdrh "NAXIS2" in
      
      (* We need to handle RGB stacking differently - extract each plane *)
      printf "Separating RGB planes for %d files...\n" (Array.length files);
      
      (* Process each file to extract RGB planes *)
      let r_planes = ref [] in
      let g_planes = ref [] in
      let b_planes = ref [] in
      
      Array.iter (fun file ->
        printf "Processing %s\n" (Filename.basename file);
        try
          let img = read_image file in
          let hdrh, contents = find_header_end file img in
          let width = parse_int hdrh "NAXIS1" in
          let height = parse_int hdrh "NAXIS2" in
          
          (* Determine data offset and plane size *)
          let data_offset = String.length contents - (width * height * 2 * 3) in
          if data_offset < 0 then
            printf "  Warning: Unexpected data size in %s\n" (Filename.basename file)
          else begin
            (* Extract each plane *)
            let r_data = Array.make_matrix height width 0 in
            let g_data = Array.make_matrix height width 0 in
            let b_data = Array.make_matrix height width 0 in
            
            (* Read red plane *)
            let plane_size = width * height * 2 in
            for y = 0 to height - 1 do
              for x = 0 to width - 1 do
                let offset = data_offset + (y * width + x) * 2 in
                if offset + 1 < String.length contents then
                  r_data.(y).(x) <- (int_of_char contents.[offset] lsl 8) lor 
                                    (int_of_char contents.[offset + 1])
              done
            done;
            
            (* Read green plane *)
            for y = 0 to height - 1 do
              for x = 0 to width - 1 do
                let offset = data_offset + plane_size + (y * width + x) * 2 in
                if offset + 1 < String.length contents then
                  g_data.(y).(x) <- (int_of_char contents.[offset] lsl 8) lor 
                                    (int_of_char contents.[offset + 1])
              done
            done;
            
            (* Read blue plane *)
            for y = 0 to height - 1 do
              for x = 0 to width - 1 do
                let offset = data_offset + plane_size * 2 + (y * width + x) * 2 in
                if offset + 1 < String.length contents then
                  b_data.(y).(x) <- (int_of_char contents.[offset] lsl 8) lor 
                                    (int_of_char contents.[offset + 1])
              done
            done;
            
            (* Add to plane lists *)
            r_planes := (file, r_data) :: !r_planes;
            g_planes := (file, g_data) :: !g_planes;
            b_planes := (file, b_data) :: !b_planes;
            
            (* Debug - print some stats *)
            let r_sum = ref 0 in
            let g_sum = ref 0 in
            let b_sum = ref 0 in
            for i = 0 to min 10 (width * height - 1) do
              let y = i / width in
              let x = i mod width in
              r_sum := !r_sum + r_data.(y).(x);
              g_sum := !g_sum + g_data.(y).(x);
              b_sum := !b_sum + b_data.(y).(x);
            done;
            printf "  Sample averages (first 10 pixels): R=%d, G=%d, B=%d\n"
              (!r_sum / 10) (!g_sum / 10) (!b_sum / 10);
          end
        with e ->
          printf "  Error processing %s: %s\n" 
            (Filename.basename file) (Printexc.to_string e)
      ) files;
      
      (* If we successfully extracted planes, stack each separately *)
      if List.length !r_planes > 0 && 
         List.length !g_planes > 0 && 
         List.length !b_planes > 0 then begin
        printf "Extracted %d sets of RGB planes\n" (List.length !r_planes);
        
        (* Create temporary files for each plane *)
        let tmp_dir = Filename.get_temp_dir_name () in
        let base_name = Filename.remove_extension (Filename.basename output_path) in
        
        (* Function to write a plane to a temporary FITS file *)
        let write_plane_to_file plane_data filename =
          let out_fd = open_out_bin filename in
          
          (* Copy header from original but modify for single plane *)
          let new_hdrh = Hashtbl.copy hdrh in
          Hashtbl.replace new_hdrh "NAXIS" " = 2 / Number of data axes";
          Hashtbl.remove new_hdrh "NAXIS3";
          
          (* Write header *)
          ignore (write_fits_header out_fd new_hdrh);
          
          (* Write data *)
          for y = 0 to height - 1 do
            for x = 0 to width - 1 do
              let value = plane_data.(y).(x) in
              output_byte out_fd (value lsr 8);
              output_byte out_fd (value land 0xFF);
            done
          done;
          
          close_out out_fd;
          filename
        in
        
        (* Write each plane to temporary files *)
        let r_files = List.mapi (fun i (file, data) ->
          let tmp_file = Printf.sprintf "%s/%s_r_%03d.fits" tmp_dir base_name i in
          write_plane_to_file data tmp_file
        ) !r_planes in
        
        let g_files = List.mapi (fun i (file, data) ->
          let tmp_file = Printf.sprintf "%s/%s_g_%03d.fits" tmp_dir base_name i in
          write_plane_to_file data tmp_file
        ) !g_planes in
        
        let b_files = List.mapi (fun i (file, data) ->
          let tmp_file = Printf.sprintf "%s/%s_b_%03d.fits" tmp_dir base_name i in
          write_plane_to_file data tmp_file
        ) !b_planes in
        
        (* Now stack each channel separately and combine *)
        printf "Stacking individual color planes...\n";
        let r_output = Printf.sprintf "%s/%s_r_stacked.fits" tmp_dir base_name in
        let g_output = Printf.sprintf "%s/%s_g_stacked.fits" tmp_dir base_name in
        let b_output = Printf.sprintf "%s/%s_b_stacked.fits" tmp_dir base_name in
        
        let r_result = stack_images (Array.of_list r_files) reference_idx stacking_method r_output in
        let g_result = stack_images (Array.of_list g_files) reference_idx stacking_method g_output in
        let b_result = stack_images (Array.of_list b_files) reference_idx stacking_method b_output in
        
        match r_result, g_result, b_result with
        | Some r, Some g, Some b ->
            (* Read the stacked color planes *)
            printf "Combining stacked color planes...\n";
            
            let r_img = read_image r_output in
            let r_hdrh, r_contents = find_header_end r_output r_img in
            let r_data = read_fits_data r_contents width height in
            
            let g_img = read_image g_output in
            let g_hdrh, g_contents = find_header_end g_output g_img in
            let g_data = read_fits_data g_contents width height in
            
            let b_img = read_image b_output in
            let b_hdrh, b_contents = find_header_end b_output b_img in
            let b_data = read_fits_data b_contents width height in
            
            (* Combine the color planes *)
            let rgb_data = create_rgb_from_mono r_data g_data b_data width height in
            
            (* Write the combined RGB FITS *)
            if write_rgb_fits output_path r_hdrh rgb_data (Array.of_list r_files) stacking_method then begin
              printf "Successfully stacked and combined RGB planes to %s\n" output_path;
              Some {
                reference_image = files.(reference_idx);
                aligned_images = files;
                failed_images = [||];
                stacking_method = stacking_method;
                output_file = output_path;
              }
            end else
              None
        | _ ->
            printf "Failed to stack one or more color planes\n";
            None
      end else begin
        printf "Failed to extract color planes from input files\n";
        printf "Falling back to standard stacking method\n";
        stack_images files reference_idx stacking_method output_path
      end
    end else if bayer_pattern <> None then begin
      printf "Detected Bayer pattern: %s, stacking with debayering\n"
        (Debayer_integration.describe_bayer_pattern (Option.get bayer_pattern));
      stack_bayer_images files reference_idx bayer_pattern stacking_method output_path
    end else begin
      printf "Detected monochrome image, stacking directly\n";
      stack_images files reference_idx stacking_method output_path
    end
  end

(* Find the best reference image in a set based on image quality metrics *)
let find_best_reference_image files =
  if Array.length files = 0 then
    0  (* Default to first image if list is empty *)
  else begin
    (* This is a simplified implementation *)
    (* A real implementation would analyze each image for:
       1. Star count and sharpness
       2. Background noise levels
       3. Overall contrast
       For now we'll just count stars using our detector 
    *)
    
    let max_stars = ref 0 in
    let best_index = ref 0 in
    
    for i = 0 to Array.length files - 1 do
      printf "Analyzing quality of %s (%d/%d)...\n" 
        (Filename.basename files.(i)) (i+1) (Array.length files);
      
      try
        (* Read image *)
        let img = read_image files.(i) in
        let hdrh, contents = find_header_end files.(i) img in
        let width = parse_int hdrh "NAXIS1" in
        let height = parse_int hdrh "NAXIS2" in
        let data = read_fits_data contents width height in
        
        (* Detect stars *)
        let detection_params = {
          threshold = 5.0;
          min_separation = 10;
          max_stars = 1000;  (* Large value to count all stars *)
        } in
        
        let stars = detect_stars data width height detection_params in
        let star_count = List.length stars in
        
        printf "  Detected %d stars\n" star_count;
        
        (* If this image has more stars than our current best, update the best *)
        if star_count > !max_stars then begin
          max_stars := star_count;
          best_index := i;
        end
      with e ->
        printf "  Error analyzing image: %s\n" (Printexc.to_string e)
    done;
    
    printf "Best reference image: %s with %d stars\n" 
      (Filename.basename files.(!best_index)) !max_stars;
    
    !best_index
  end
