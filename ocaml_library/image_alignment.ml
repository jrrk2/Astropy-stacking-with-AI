(* image_alignment.ml - Module for aligning and stacking astronomical images *)

open Types
open Fits
open Printf

(* Types for star detection and alignment *)
type star_point = {
  x: float;          (* X coordinate in pixels *)
  y: float;          (* Y coordinate in pixels *)
  flux: float;       (* Integrated flux (brightness) *)
  fwhm: float;       (* Full-width half-maximum (star size) *)
}

type alignment_parameters = {
  dx: float;         (* X translation *)
  dy: float;         (* Y translation *)
  rotation: float;   (* Rotation angle in radians *)
  scale: float;      (* Scale factor *)
}

type stacking_method = 
  | Average           (* Simple mean stacking *)
  | Median            (* Median stacking - good for cosmic ray rejection *)
  | SigmaClip of float (* Sigma-clipped mean with given sigma threshold *)
  | Kappa of float    (* Kappa-sigma clipping with rejection threshold *)
  | WeightedAverage   (* Weighted by image quality *)

(* Star detection parameters *)
type detection_params = {
  threshold: float;   (* Detection threshold in sigma above background *)
  min_separation: int; (* Minimum separation between stars in pixels *)
  max_stars: int;     (* Maximum number of stars to use for alignment *)
}

(* Result of the stacking operation *)
type stacking_result = {
  reference_image: string;         (* Filename of reference image *)
  aligned_images: string array;    (* Filenames of successfully aligned images *)
  failed_images: string array;     (* Filenames of images that failed to align *)
  stacking_method: stacking_method; (* Method used for stacking *)
  output_file: string;            (* Path to output stacked image *)
}

(* Default parameters for star detection *)
let default_detection_params = {
  threshold = 5.0;
  min_separation = 10;
  max_stars = 100;
}

(* Create identity transformation parameters (no change) *)
let identity_transform = {
  dx = 0.0;
  dy = 0.0;
  rotation = 0.0;
  scale = 1.0;
}

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

(* Calculate similarity transformation parameters between two sets of star positions *)
let calculate_alignment_parameters reference_stars target_stars =
  (* We need at least 3 matching stars to determine transformation reliably *)
  if List.length reference_stars < 3 || List.length target_stars < 3 then
    None
  else begin
    (* For simplicity, we'll use the brightest stars for matching *)
    let ref_sorted = List.sort (fun s1 s2 -> compare s2.flux s1.flux) reference_stars in
    let target_sorted = List.sort (fun s1 s2 -> compare s2.flux s1.flux) target_stars in
    
    (* Take the top N stars from each list *)
    let n_match = min 10 (min (List.length ref_sorted) (List.length target_sorted)) in
    let ref_top = Array.of_list (List.take n_match ref_sorted) in
    let target_top = Array.of_list (List.take n_match target_sorted) in
    
    (* Match stars based on triangle patterns - simplified algorithm *)
    (* For now we'll just use a naive approach matching brightest stars *)
    (* A real implementation would use triangle pattern matching *)
    
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
    
    (* Use centroids to calculate translation *)
    let dx = target_centroid_x -. ref_centroid_x in
    let dy = target_centroid_y -. ref_centroid_y in
    
    (* For now, simplest alignment is just translation *)
    (* A full solution would calculate rotation and scale too *)
    Some { dx; dy; rotation = 0.0; scale = 1.0 }
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

(* Apply alignment to an image *)
let align_image src_data width height params =
  (* Create output image buffer *)
  let dest_data = Array.make_matrix height width 0 in
  
  (* Loop through destination pixels and sample from source using inverse transform *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      (* Apply inverse transformation to get source coordinates *)
      let src_x, src_y = transform_coordinates 
        (float_of_int x -. params.dx) 
        (float_of_int y -. params.dy) 
        { dx = 0.0; dy = 0.0; 
          rotation = -.params.rotation; 
          scale = 1.0 /. params.scale } in
      
      (* Convert to integers and check bounds *)
      let src_x_int = int_of_float (src_x +. 0.5) in
      let src_y_int = int_of_float (src_y +. 0.5) in
      
      (* Sample source pixel - with bounds checking *)
      if src_x_int >= 0 && src_x_int < width && 
         src_y_int >= 0 && src_y_int < height then
        dest_data.(y).(x) <- src_data.(src_y_int).(src_x_int)
      (* else leave as 0 - more sophisticated approach would be interpolation *)
    done
  done;
  
  dest_data

(* Align all images to a reference image *)
let align_images files reference_idx detection_params =
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

(* Stack images using given method *)
let rec stack_images files reference_idx stacking_method output_path =
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
    for i = 0 to min 9 (Array.length aligned_files - 1) do
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

(* Write stacked RGB image *)
let write_stacked_rgb_image output_path ref_hdrh rgb_data aligned_files stacking_method =
  try
    (* Create a copy of the reference header *)
    let header = Hashtbl.copy ref_hdrh in
    
    (* Update header for RGB FITS format *)
    Hashtbl.replace header "NAXIS" (sprintf " = 3 / Number of data axes");
    Hashtbl.replace header "NAXIS3" (sprintf " = 3 / Number of color planes (RGB)");
    
    (* Add stacking information *)
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
    
    (* List source files *)
    for i = 0 to min 9 (Array.length aligned_files - 1) do
      let key = Printf.sprintf "IMGSRC%d=" i in
      let value = Printf.sprintf "'%s'" (Filename.basename aligned_files.(i)) in
      let comment = if i = 0 then " / Source images (up to 10 listed)" else "" in
      Hashtbl.replace header key (Printf.sprintf " = %s%s" value comment);
    done;
    
    (* Use the RGB helper function to write the data *)
    Fits.write_rgb_data_to_fits output_path header rgb_data
  with e ->
    printf "Error writing stacked RGB image: %s\n" (Printexc.to_string e);
    false

(* Helper function to create RGB data array from monochrome images *)
let create_rgb_from_mono r_data g_data b_data width height =
  let rgb_data = Array.make_matrix height width (0, 0, 0) in
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      rgb_data.(y).(x) <- (r_data.(y).(x), g_data.(y).(x), b_data.(y).(x))
    done
  done;
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
        
        (* Combine into RGB array *)
        let rgb_data = create_rgb_from_mono r_data g_data b_data width height in
        
        (* Write combined RGB FITS *)
        if write_stacked_rgb_image output_path r_hdrh rgb_data r.aligned_images stacking_method then
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
      
      (* Apply stacking method to each pixel *)
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
        if write_stacked_rgb_image output_path ref_hdrh rgb_data aligned_files stacking_method then
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
    let hdrh, _ = find_header_end files.(0) img in
    
    (* Check for Bayer pattern *)
    let bayer_pattern = Debayer_integration.get_bayer_pattern hdrh in
    
    (* Check for color planes *)
    let naxis = parse_int hdrh "NAXIS" in
    let is_color = naxis = 3 || naxis = 4 in
    
    if is_color then begin
      printf "Detected color image (NAXIS=%d), stacking as RGB\n" naxis;
      stack_images files reference_idx stacking_method output_path
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
