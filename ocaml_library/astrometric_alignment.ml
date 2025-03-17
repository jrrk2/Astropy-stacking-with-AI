(* astrometric_alignment.ml - Functions for alignment based on plate solving data *)
open Types
open Fits

(* WCS parameters from plate solved FITS headers *)
type wcs_params = {
  crpix1: float;     (* X reference pixel *)
  crpix2: float;     (* Y reference pixel *)
  crval1: float;     (* RA at reference pixel (degrees) *)
  crval2: float;     (* DEC at reference pixel (degrees) *)
  cd1_1: float;      (* Transformation matrix element *)
  cd1_2: float;      (* Transformation matrix element *)
  cd2_1: float;      (* Transformation matrix element *)
  cd2_2: float;      (* Transformation matrix element *)
  equinox: float;    (* Equinox of coordinates *)
}

let print_memory_usage label =
  let stat = Gc.stat () in
  print_endline (Printf.sprintf "%s: Heap words: %d, Live words: %d, Free words: %d\n" 
    label stat.heap_words stat.live_words stat.free_words)

(* Extract WCS information from FITS header *)
let extract_wcs_params header =
  try
    let crpix1 = parse_float header "CRPIX1" in
    let crpix2 = parse_float header "CRPIX2" in
    let crval1 = parse_float header "CRVAL1" in
    let crval2 = parse_float header "CRVAL2" in
    
    (* CD matrix values - handles both CD and CDELT formats *)
    let (cd1_1, cd1_2, cd2_1, cd2_2) =
      try
        (* Try CD matrix format first *)
        (parse_float header "CD1_1",
         parse_float header "CD1_2",
         parse_float header "CD2_1",
         parse_float header "CD2_2")
      with Not_found ->
        (* Fall back to CDELT format *)
        let cdelt1 = parse_float header "CDELT1" in
        let cdelt2 = parse_float header "CDELT2" in
        let crota = 
          try parse_float header "CROTA2"
          with Not_found -> 0.0 
        in
        let cos_rot = cos (crota *. Float.pi /. 180.0) in
        let sin_rot = sin (crota *. Float.pi /. 180.0) in
        (cdelt1 *. cos_rot, (-. cdelt1) *. sin_rot,
         cdelt2 *. sin_rot, cdelt2 *. cos_rot)
    in
    
    (* Get equinox or default to 2000.0 *)
    let equinox = 
      try parse_float header "EQUINOX"
      with Not_found -> 
        try parse_float header "EPOCH"
        with Not_found -> 2000.0
    in
    
    Some {
      crpix1; crpix2;
      crval1; crval2;
      cd1_1; cd1_2;
      cd2_1; cd2_2;
      equinox;
    }
  with Not_found ->
    Printf.printf "Warning: Missing required WCS keys in FITS header\n";
    flush stdout;
    None

(* Convert pixel coordinates to sky coordinates (RA/Dec) *)
let pixel_to_sky wcs x y =
  (* Convert pixel coordinates to 0-based *)
  let x_pix = x +. 1.0 -. wcs.crpix1 in
  let y_pix = y +. 1.0 -. wcs.crpix2 in
  
  (* Apply CD matrix transformation *)
  let ra_offset = wcs.cd1_1 *. x_pix +. wcs.cd1_2 *. y_pix in
  let dec_offset = wcs.cd2_1 *. x_pix +. wcs.cd2_2 *. y_pix in
  
  (* Add offsets to reference values *)
  let ra = wcs.crval1 +. ra_offset in
  let dec = wcs.crval2 +. dec_offset in
  
  (ra, dec)

(* Convert sky coordinates (RA/Dec) to pixel coordinates *)
let sky_to_pixel wcs ra dec =
  (* Calculate offsets from reference point *)
  let ra_offset = ra -. wcs.crval1 in
  let dec_offset = dec -. wcs.crval2 in
  
  (* Compute determinant of CD matrix for inverse *)
  let det = wcs.cd1_1 *. wcs.cd2_2 -. wcs.cd1_2 *. wcs.cd2_1 in
  
  if abs_float det < 1e-10 then
    failwith "Singular CD matrix in WCS parameters";
  
  (* Compute inverse of CD matrix *)
  let cd1_1_inv = wcs.cd2_2 /. det in
  let cd1_2_inv = -.wcs.cd1_2 /. det in
  let cd2_1_inv = -.wcs.cd2_1 /. det in
  let cd2_2_inv = wcs.cd1_1 /. det in
  
  (* Apply inverse transformation *)
  let x_pix = cd1_1_inv *. ra_offset +. cd1_2_inv *. dec_offset in
  let y_pix = cd2_1_inv *. ra_offset +. cd2_2_inv *. dec_offset in
  
  (* Convert to FITS 1-based coordinates *)
  let x = x_pix +. wcs.crpix1 -. 1.0 in
  let y = y_pix +. wcs.crpix2 -. 1.0 in
  
  (x, y)

(* Create a transformation function from one WCS to another *)
let create_wcs_transform src_wcs dst_wcs =
  (fun x y ->
    (* Convert source pixel to sky coordinates *)
    let ra, dec = pixel_to_sky src_wcs x y in
    
    (* Convert sky coordinates to destination pixel coordinates *)
    sky_to_pixel dst_wcs ra dec)

(* Add this function for direct image-to-image alignment *)
let align_direct ref_data ref_width ref_height src_data src_width src_height src_wcs ref_wcs =
  let dst_data = Array.make_matrix ref_height ref_width 0 in
  
  (* Find 4 reference points for a projective transform *)
  let ref_points = [
    (ref_width / 4, ref_height / 4);
    (3 * ref_width / 4, ref_height / 4);
    (ref_width / 4, 3 * ref_height / 4);
    (3 * ref_width / 4, 3 * ref_height / 4)
  ] in
  
  (* Map reference points to source image using WCS *)
  let src_points = List.map (fun (x, y) ->
    let (ra, dec) = pixel_to_sky ref_wcs (float_of_int x) (float_of_int y) in
    let (src_x, src_y) = sky_to_pixel src_wcs ra dec in
    (src_x, src_y)
  ) ref_points in
  
  (* For debugging, print the mappings *)
  List.iter2 (fun (rx, ry) (sx, sy) ->
    Printf.printf "Mapping ref(%d,%d) -> src(%.1f,%.1f)\n" rx ry sx sy
  ) ref_points src_points;
  
  (* Simple bilinear mapping using the four corner points *)
  let transform x y =
    (* Calculate normalized coordinates in reference image (0-1) *)
    let nx = float_of_int x /. float_of_int ref_width in
    let ny = float_of_int y /. float_of_int ref_height in
    
    (* Interpolate between the source points *)
    let p00 = List.nth src_points 0 in
    let p10 = List.nth src_points 1 in
    let p01 = List.nth src_points 2 in
    let p11 = List.nth src_points 3 in
    
    let sx = (1.0 -. nx) *. (1.0 -. ny) *. (fst p00) +.
             nx *. (1.0 -. ny) *. (fst p10) +.
             (1.0 -. nx) *. ny *. (fst p01) +.
             nx *. ny *. (fst p11) in
    
    let sy = (1.0 -. nx) *. (1.0 -. ny) *. (snd p00) +.
             nx *. (1.0 -. ny) *. (snd p10) +.
             (1.0 -. nx) *. ny *. (snd p01) +.
             nx *. ny *. (snd p11) in
    
    (sx, sy)
  in
  
  (* Fill destination image using the transform *)
  for y = 0 to ref_height - 1 do
    for x = 0 to ref_width - 1 do
      let (src_x, src_y) = transform x y in
      
      (* Check bounds and do bilinear interpolation *)
      if src_x >= 0.0 && src_x < float_of_int src_width -. 1.0 &&
         src_y >= 0.0 && src_y < float_of_int src_height -. 1.0 then begin
        
        let src_x_floor = floor src_x in
        let src_y_floor = floor src_y in
        let src_x_int = int_of_float src_x_floor in
        let src_y_int = int_of_float src_y_floor in
        
        let x_frac = src_x -. src_x_floor in
        let y_frac = src_y -. src_y_floor in
        
        (* Bilinear interpolation *)
        let p00 = float_of_int src_data.(src_y_int).(src_x_int) in
        let p10 = float_of_int src_data.(src_y_int).(src_x_int + 1) in
        let p01 = float_of_int src_data.(src_y_int + 1).(src_x_int) in
        let p11 = float_of_int src_data.(src_y_int + 1).(src_x_int + 1) in
        
        let value = 
          p00 *. (1.0 -. x_frac) *. (1.0 -. y_frac) +.
          p10 *. x_frac *. (1.0 -. y_frac) +.
          p01 *. (1.0 -. x_frac) *. y_frac +.
          p11 *. x_frac *. y_frac
        in
        
        dst_data.(y).(x) <- int_of_float (Float.round value)
      end
    done
  done;
  
  dst_data

let align_image_wcs src_data src_width src_height src_wcs dst_wcs dst_width dst_height =
  (* Create output image buffer *)
  let dst_data = Array.make_matrix dst_height dst_width 0 in
  
  (* Debug the transformation to understand its characteristics *)
  Printf.printf "WCS transformation analysis:\n";
  Printf.printf "  Source: CRPIX=(%.2f, %.2f), CRVAL=(%.6f, %.6f)\n" 
    src_wcs.crpix1 src_wcs.crpix2 src_wcs.crval1 src_wcs.crval2;
  Printf.printf "  Destination: CRPIX=(%.2f, %.2f), CRVAL=(%.6f, %.6f)\n" 
    dst_wcs.crpix1 dst_wcs.crpix2 dst_wcs.crval1 dst_wcs.crval2;
  
(* In align_image_wcs, replace the transform function with this more precise version *)
let transform x y =
  (* Convert destination pixel to sky coordinates with high precision *)
  let x_pix = x +. 1.0 -. dst_wcs.crpix1 in
  let y_pix = y +. 1.0 -. dst_wcs.crpix2 in
  
  (* Apply CD matrix with double precision *)
  let ra_offset = dst_wcs.cd1_1 *. x_pix +. dst_wcs.cd1_2 *. y_pix in
  let dec_offset = dst_wcs.cd2_1 *. x_pix +. dst_wcs.cd2_2 *. y_pix in
  
  (* Calculate precise RA/Dec for this pixel *)
  let ra = dst_wcs.crval1 +. ra_offset in
  let dec = dst_wcs.crval2 +. dec_offset in
  
  (* Convert to source pixel coordinates with high precision inverse matrix *)
  let det = src_wcs.cd1_1 *. src_wcs.cd2_2 -. src_wcs.cd1_2 *. src_wcs.cd2_1 in
  
  if abs_float det < 1e-10 then
    failwith "Singular CD matrix in source WCS parameters";
  
  let cd1_1_inv = src_wcs.cd2_2 /. det in
  let cd1_2_inv = -.src_wcs.cd1_2 /. det in
  let cd2_1_inv = -.src_wcs.cd2_1 /. det in
  let cd2_2_inv = src_wcs.cd1_1 /. det in
  
  (* Calculate RA/Dec offsets from source reference point *)
  let ra_offset_src = ra -. src_wcs.crval1 in
  let dec_offset_src = dec -. src_wcs.crval2 in
  
  (* Apply inverse transformation with double precision *)
  let x_pix_src = cd1_1_inv *. ra_offset_src +. cd1_2_inv *. dec_offset_src in
  let y_pix_src = cd2_1_inv *. ra_offset_src +. cd2_2_inv *. dec_offset_src in
  
  (* Convert to source pixel coordinates (1-based to 0-based) *)
  let x_src = x_pix_src +. src_wcs.crpix1 -. 1.0 in
  let y_src = y_pix_src +. src_wcs.crpix2 -. 1.0 in
  
  (x_src, y_src)
in
  (* Add test points to verify transformation *)
  let test_points = [(dst_width/2, dst_height/2); (0, 0); (dst_width-1, dst_height-1)] in
  Printf.printf "Transformation test points:\n";
  List.iter (fun (x, y) ->
    let src_x, src_y = transform (float_of_int x) (float_of_int y) in
    Printf.printf "  Dst (%d, %d) -> Src (%.1f, %.1f)\n" x y src_x src_y;
    
    (* Check if this falls within src image *)
    let in_bounds = 
      src_x >= 0.0 && src_x < float_of_int src_width &&
      src_y >= 0.0 && src_y < float_of_int src_height
    in
    Printf.printf "    In bounds: %b\n" in_bounds;
  ) test_points;
  
  (* If this is the first image being aligned to itself, add extra debug *)
  if src_width = dst_width && src_height = dst_height then begin
    let non_zero_count = ref 0 in
    
    (* Process center region of image *)
    for y = dst_height/4 to 3*dst_height/4 - 1 do
      for x = dst_width/4 to 3*dst_width/4 - 1 do
        let src_x, src_y = transform (float_of_int x) (float_of_int y) in
        
        if src_x >= 0.0 && src_x < float_of_int src_width -. 1.0 &&
           src_y >= 0.0 && src_y < float_of_int src_height -. 1.0 then begin
          
          let src_x_int = int_of_float src_x in
          let src_y_int = int_of_float src_y in
          
          let value = src_data.(src_y_int).(src_x_int) in
          if value > 0 then incr non_zero_count;
          
          dst_data.(y).(x) <- value;
        end
      done
    done;
    
    Printf.printf "Self-alignment test: %d non-zero pixels in center region\n" !non_zero_count;
  end;
  
  (* Align the full image - iterate through each pixel in destination image *)
  for y = 0 to dst_height - 1 do
    for x = 0 to dst_width - 1 do
      (* Calculate source coordinates *)
      let src_x, src_y = transform (float_of_int x) (float_of_int y) in
      
      (* Check if source coordinates are within bounds *)
      if src_x >= 0.0 && src_x < float_of_int src_width -. 1.0 &&
         src_y >= 0.0 && src_y < float_of_int src_height -. 1.0 then begin
        
        (* Bilinear interpolation *)
        let src_x_floor = floor src_x in
        let src_y_floor = floor src_y in
        let src_x_int = int_of_float src_x_floor in
        let src_y_int = int_of_float src_y_floor in
        
        let x_frac = src_x -. src_x_floor in
        let y_frac = src_y -. src_y_floor in
        
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
        
        dst_data.(y).(x) <- int_of_float (Float.round value)
      end
    done
  done;
  
  dst_data

let calculate_stack_dimensions files =
  (* Extract WCS and dimensions from all images *)
  let image_params = ref [] in
  
  List.iter (fun file ->
    try
      let header, contents = find_header_end file (read_image file) in
      match extract_wcs_params header with
      | Some wcs -> 
          let width = parse_int header "NAXIS1" in
          let height = parse_int header "NAXIS2" in
          image_params := (file, wcs, width, height) :: !image_params
      | None -> 
          Printf.printf "Warning: No WCS information found in %s\n" file
    with e -> 
      Printf.printf "Error processing %s: %s\n" file (Printexc.to_string e)
  ) files;
  
  if List.length !image_params = 0 then
    failwith "No valid plate-solved images found";
  
  (* Choose the middle image as reference *)
  let (ref_file, ref_wcs, ref_width, ref_height) = 
    List.nth !image_params (List.length !image_params / 2)
  in
  
  Printf.printf "Reference image: %s (%dx%d)\n" ref_file ref_width ref_height;
  
  (* Calculate corners of reference image *)
  let corners = [
    pixel_to_sky ref_wcs 0.0 0.0;
    pixel_to_sky ref_wcs (float_of_int (ref_width-1)) 0.0;
    pixel_to_sky ref_wcs 0.0 (float_of_int (ref_height-1));
    pixel_to_sky ref_wcs (float_of_int (ref_width-1)) (float_of_int (ref_height-1))
  ] in
  
  (* For each image, calculate its mapping onto the reference image *)
  List.iter (fun (file, wcs, width, height) ->
    if file <> ref_file then begin
      let center_px = (float_of_int width /. 2.0, float_of_int height /. 2.0) in
      let (ra, dec) = pixel_to_sky wcs (fst center_px) (snd center_px) in
      let (ref_x, ref_y) = sky_to_pixel ref_wcs ra dec in
      Printf.printf "  Image %s center maps to ref position (%.1f, %.1f)\n" 
        (Filename.basename file) ref_x ref_y
    end
  ) !image_params;
  
  (* Use the reference image dimensions and add margins *)
  let margin = 100 in
  let width_px = ref_width + 2 * margin in
  let height_px = ref_height + 2 * margin in
  
  (* Calculate the plate scale from reference image *)
  let scale_x = sqrt (ref_wcs.cd1_1 *. ref_wcs.cd1_1 +. ref_wcs.cd2_1 *. ref_wcs.cd2_1) *. 3600.0 in
  let scale_y = sqrt (ref_wcs.cd1_2 *. ref_wcs.cd1_2 +. ref_wcs.cd2_2 *. ref_wcs.cd2_2) *. 3600.0 in
  let avg_scale = (scale_x +. scale_y) /. 2.0 in
  
  Printf.printf "Average plate scale: %.2f arcsec/pixel\n" avg_scale;
  
  (* Create output WCS based on reference image but shifted to include margins *)
  let output_wcs = {
    crpix1 = ref_wcs.crpix1 +. float_of_int margin;
    crpix2 = ref_wcs.crpix2 +. float_of_int margin;
    crval1 = ref_wcs.crval1;
    crval2 = ref_wcs.crval2;
    cd1_1 = ref_wcs.cd1_1;
    cd1_2 = ref_wcs.cd1_2;
    cd2_1 = ref_wcs.cd2_1;
    cd2_2 = ref_wcs.cd2_2;
    equinox = ref_wcs.equinox;
  } in
  
  (output_wcs, width_px, height_px, avg_scale, !image_params)

(* Stack a list of plate-solved FITS images *)

let apply_stacking_method values stacking_method =
      if Array.length values > 0 then
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
              (* Simple implementation - all weights equal for now *)
              let sum = Array.fold_left (+) 0 values in
              sum / Array.length values
        in        
        stacked_value
    else
        0

let stack_astrometric files reference_idx stacking_method output_path =
  Printf.printf "Stacking %d images using astrometric alignment...\n" (List.length files);
  flush stdout;
  
  (* Calculate the output dimensions and WCS parameters *)
  let (output_wcs, width, height, avg_scale, image_params) = calculate_stack_dimensions files in
  let (ref_file, ref_wcs, ref_width, ref_height) = List.nth image_params (reference_idx) in
  let _, ref_contents = find_header_end ref_file (read_image ref_file) in
  let ref_data = read_fits_data ref_contents ref_width ref_height in

  (* First check if the input files are RGB (NAXIS=3) *)
  let is_rgb = 
    try
      let first_hdr = Fits.just_header (List.hd files) in
      let naxis = parse_int first_hdr "NAXIS" in
      let naxis3 = if naxis = 3 then parse_int first_hdr "NAXIS3" else 1 in
      naxis = 3 && naxis3 = 3
    with _ -> false
  in
  
  Printf.printf "Detected %s images\n" (if is_rgb then "RGB" else "monochrome");
  
  if is_rgb then begin
    (* Handle RGB stacking *)
    (* Create arrays for each color plane *)
    let stacked_r = Array.make_matrix height width [] in
    let stacked_g = Array.make_matrix height width [] in
    let stacked_b = Array.make_matrix height width [] in
    
    (* Process each image *)
    List.iter (fun (file, wcs, img_width, img_height) ->
      Printf.printf "Processing %s...\n" (Filename.basename file);
      flush stdout;
      
      (* Read the image data (all 3 planes) *)
      let _, contents = Fits.find_header_end file (Fits.read_image file) in
      let img_data = read_fits_data contents img_width img_height in
      
      (* Calculate plane size and offsets *)
      let plane_size = img_width * img_height * 2 in (* 16-bit = 2 bytes per pixel *)
      
      (* Extract and align each color plane *)
      for plane = 0 to 2 do
        let plane_offset = plane * plane_size in
        
        (* Extract the plane data *)
        let plane_data = Array.make_matrix img_height img_width 0 in
        for y = 0 to img_height - 1 do
          for x = 0 to img_width - 1 do
            let offset = plane_offset + (y * img_width + x) * 2 in
            if offset + 1 < String.length contents then
              plane_data.(y).(x) <- (int_of_char contents.[offset] lsl 8) lor 
                                  (int_of_char contents.[offset + 1])
          done
        done;
        
        (* Align this plane *)
	let aligned_data = 
	  if file = ref_file then 
	    Array.map Array.copy ref_data
          else
	      align_direct ref_data ref_width ref_height img_data img_width img_height wcs ref_wcs in

        (* Add to stacked data for this plane *)
        let stacked_plane = match plane with
          | 0 -> stacked_r
          | 1 -> stacked_g
          | _ -> stacked_b
        in
        
        for y = 0 to ref_height - 1 do
          for x = 0 to ref_width - 1 do
            stacked_plane.(y).(x) <- aligned_data.(y).(x) :: stacked_plane.(y).(x)
          done
        done;
        
        Printf.printf "  Added %s plane %d to stack\n" (Filename.basename file) plane;
      done;
    ) image_params;
    
    (* Apply stacking method to each pixel in each plane *)
    let output_r = Array.make_matrix height width 0 in
    let output_g = Array.make_matrix height width 0 in
    let output_b = Array.make_matrix height width 0 in
    
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        (* Process each color plane separately *)
        for plane = 0 to 2 do
          let values = match plane with
            | 0 -> Array.of_list stacked_r.(y).(x)
            | 1 -> Array.of_list stacked_g.(y).(x)
            | _ -> Array.of_list stacked_b.(y).(x)
          in
          
          let stacked_value = apply_stacking_method values stacking_method in
          
          match plane with
            | 0 -> output_r.(y).(x) <- stacked_value
            | 1 -> output_g.(y).(x) <- stacked_value
            | _ -> output_b.(y).(x) <- stacked_value
        done
      done;
      
      (* Print progress for large images *)
      if height > 1000 && y mod 100 = 0 then begin
        Printf.printf "  Stacking progress: %.1f%%\n" (float_of_int y *. 100.0 /. float_of_int height);
        flush stdout
      end
    done;
    
    (* Create FITS header for output file - make sure to set NAXIS=3 *)
    let header = Hashtbl.create 50 in
    
    (* Get header from reference file *)
    let ref_hdr = Fits.just_header (List.nth files reference_idx) in
    
    (* Copy important keywords but enforce RGB structure *)
    List.iter (fun key ->
      if key <> "NAXIS" && key <> "NAXIS1" && key <> "NAXIS2" && key <> "NAXIS3" then
        match Hashtbl.find_opt ref_hdr key with
        | Some value -> Hashtbl.add header key value
        | None -> ()
    ) ["SIMPLE"; "BITPIX"; "BZERO"; "BSCALE"; "DATE-OBS"; "INSTRUME"; "EXPOSURE"; "FOCAL"; "PIXSZ"; "EXTEND"];
    
    (* Set RGB dimensions *)
    Hashtbl.add header "SIMPLE" " = T / FITS standard";
    Hashtbl.add header "BITPIX" " = 16 / 16-bit signed integers";
    Hashtbl.add header "NAXIS" " = 3 / Number of data axes";
    Hashtbl.add header "NAXIS1" (Printf.sprintf " = %d / Width in pixels" width);
    Hashtbl.add header "NAXIS2" (Printf.sprintf " = %d / Height in pixels" height);
    Hashtbl.add header "NAXIS3" " = 3 / Number of color planes (RGB)";
    Hashtbl.add header "EXTEND" " = T / Extensions may be present";
    Hashtbl.add header "BZERO" " = 32768 / Offset to unsigned short range";
    Hashtbl.add header "BSCALE" " = 1 / Default scaling factor";
    
    (* Add metadata about stacking *)
    Hashtbl.add header "HISTORY" " Stacked with OCaml Astrometric Alignment";
    Hashtbl.add header "HISTORY" (Printf.sprintf " Stacking method: %s" 
      (match stacking_method with
       | Average -> "Average"
       | Median -> "Median"
       | SigmaClip sigma -> Printf.sprintf "SigmaClip (%.1f)" sigma
       | Kappa k -> Printf.sprintf "Kappa (%.1f)" k
       | WeightedAverage -> "WeightedAverage"));
    Hashtbl.add header "HISTORY" (Printf.sprintf " Number of frames: %d" (List.length image_params));
    
    (* Write the stacked RGB image *)
    let oc = open_out_bin output_path in
    
    (* Write header *)
    ignore (Fits.write_fits_header oc header);
    
    (* Write red plane *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        output_byte oc (output_r.(y).(x) lsr 8);
        output_byte oc (output_r.(y).(x) land 0xFF);
      done
    done;
    
    (* Write green plane *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        output_byte oc (output_g.(y).(x) lsr 8);
        output_byte oc (output_g.(y).(x) land 0xFF);
      done
    done;
    
    (* Write blue plane *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        output_byte oc (output_b.(y).(x) lsr 8);
        output_byte oc (output_b.(y).(x) land 0xFF);
      done
    done;
    
    (* Pad data to multiple of 2880 bytes *)
    let data_size = width * height * 2 * 3 in
    let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
    output_string oc (String.make padding_size '\000');
    
    close_out oc;
    
    Printf.printf "Stacked RGB image saved to %s\n" output_path;
    true
  end
  else begin
  
  (* Create an array to accumulate pixel values and weights *)
  let stacked_data = Array.make_matrix height width [] in
  
  (* Process each image *)
  List.iter (fun (file, wcs, img_width, img_height) ->
    Printf.printf "Processing %s...\n" (Filename.basename file);
    flush stdout;
    
    (* Read the image data *)
    let _, contents = find_header_end file (read_image file) in
    let data = read_fits_data contents img_width img_height in
    
    (* Align to the output WCS frame *)
    let aligned_data = align_image_wcs data img_width img_height wcs output_wcs width height in
    
    (* Add to stacked data *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let value = aligned_data.(y).(x) in
        if value > 0 then
          stacked_data.(y).(x) <- value :: stacked_data.(y).(x)
      done
    done;
    
    Printf.printf "  Added %s to stack\n" (Filename.basename file);
    flush stdout
  ) image_params;
  
  (* Apply stacking method to each pixel *)
  let output_data = Array.make_matrix height width 0 in

  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let values = Array.of_list (stacked_data.(y).(x)) in
      let stacked_value = apply_stacking_method values stacking_method in             
      output_data.(y).(x) <- stacked_value
    done;
    
    (* Print progress for large images *)
    if height > 1000 && y mod 100 = 0 then begin
      Printf.printf "  Stacking progress: %.1f%%\n" (float_of_int y *. 100.0 /. float_of_int height);
      flush stdout
    end
  done;
  
  (* Create FITS header for output file *)
  let header = Hashtbl.create 50 in
  
  (* Standard FITS keywords *)
  Hashtbl.add header "SIMPLE" " = T / FITS standard";
  Hashtbl.add header "BITPIX" " = 16 / 16-bit signed integers";
  Hashtbl.add header "NAXIS" " = 2 / Number of axes";
  Hashtbl.add header "NAXIS1" (Printf.sprintf " = %d / Width in pixels" width);
  Hashtbl.add header "NAXIS2" (Printf.sprintf " = %d / Height in pixels" height);
  Hashtbl.add header "EXTEND" " = T / Extensions may be present";
  
  (* WCS keywords *)
  Hashtbl.add header "CTYPE1" " = 'RA---TAN' / Right ascension, tangent projection";
  Hashtbl.add header "CTYPE2" " = 'DEC--TAN' / Declination, tangent projection";
  Hashtbl.add header "CRPIX1" (Printf.sprintf " = %.6f / X reference pixel" output_wcs.crpix1);
  Hashtbl.add header "CRPIX2" (Printf.sprintf " = %.6f / Y reference pixel" output_wcs.crpix2);
  Hashtbl.add header "CRVAL1" (Printf.sprintf " = %.10f / RA at reference pixel (deg)" output_wcs.crval1);
  Hashtbl.add header "CRVAL2" (Printf.sprintf " = %.10f / Dec at reference pixel (deg)" output_wcs.crval2);
  Hashtbl.add header "CD1_1" (Printf.sprintf " = %.10e / Transformation matrix element" output_wcs.cd1_1);
  Hashtbl.add header "CD1_2" (Printf.sprintf " = %.10e / Transformation matrix element" output_wcs.cd1_2);
  Hashtbl.add header "CD2_1" (Printf.sprintf " = %.10e / Transformation matrix element" output_wcs.cd2_1);
  Hashtbl.add header "CD2_2" (Printf.sprintf " = %.10e / Transformation matrix element" output_wcs.cd2_2);
  Hashtbl.add header "EQUINOX" (Printf.sprintf " = %.1f / Equinox of coordinates" output_wcs.equinox);
  
  (* Add metadata about stacking *)
  Hashtbl.add header "HISTORY" " Stacked with OCaml Astrometric Alignment";
  Hashtbl.add header "HISTORY" (Printf.sprintf " Stacking method: %s" 
    (match stacking_method with
     | Average -> "Average"
     | Median -> "Median"
     | SigmaClip sigma -> Printf.sprintf "SigmaClip (%.1f)" sigma
     | Kappa k -> Printf.sprintf "Kappa (%.1f)" k
     | WeightedAverage -> "WeightedAverage"));
  Hashtbl.add header "HISTORY" (Printf.sprintf " Number of frames: %d" (List.length image_params));
  Hashtbl.add header "HIERARCH ASTRO SCALE" (Printf.sprintf " = %.6f / Plate scale (arcsec/pixel)" avg_scale);
  
  (* List input files *)
  List.iteri (fun i (file, _, _, _) ->
    Hashtbl.add header (Printf.sprintf "FRAME%03d" (i+1)) (Printf.sprintf " = '%s'" (Filename.basename file))
  ) image_params;
  
  (* Write the stacked image *)
  let oc = open_out_bin output_path in
  
  (* Write header *)
  ignore (write_fits_header oc header);
  
  (* Write data *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      (* FITS uses big-endian *)
      let value = output_data.(y).(x) in
      output_byte oc (value lsr 8);
      output_byte oc (value land 0xFF);
    done
  done;
  
  (* Pad data to multiple of 2880 bytes *)
  let data_size = width * height * 2 in
  let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
  output_string oc (String.make padding_size '\000');
  
  close_out oc;
  
  Printf.printf "Stacked image saved to %s\n" output_path;
  flush stdout;
  
  true
  end

(* Standalone tool for stacking astrometric images *)
let astrometric_stack_cli args =
  (* Command line arguments *)
  let output_file = ref "stacked_astrometric.fits" in
  let stack_method = ref "average" in
  let sigma_value = ref 3.0 in
  let kappa_value = ref 3.0 in
  let input_files = ref [] in
  
  (* Parse command line *)
  let i = ref 1 in
  while !i < Array.length args do
    (match args.(!i) with
    | "-o" -> 
        incr i;
        if !i < Array.length args then
          output_file := args.(!i)
        else
          failwith "Missing output file name after -o"
    | "-method" -> 
        incr i;
        if !i < Array.length args then
          stack_method := args.(!i)
        else
          failwith "Missing method name after -method"
    | "-sigma" ->
        incr i;
        if !i < Array.length args then
          sigma_value := float_of_string args.(!i)
        else
          failwith "Missing value after -sigma"
    | "-kappa" ->
        incr i;
        if !i < Array.length args then
          kappa_value := float_of_string args.(!i)
        else
          failwith "Missing value after -kappa"
    | arg when Filename.check_suffix arg ".fits" || Filename.check_suffix arg ".fit" ->
        input_files := arg :: !input_files
    | _ -> ()
    );
    incr i
  done;
  
  (* Convert stacking method string to type *)
  let method_type = match !stack_method with
    | "average" -> Average
    | "median" -> Median
    | "sigmaclip" -> SigmaClip !sigma_value
    | "kappa" -> Kappa !kappa_value
    | "weighted" -> WeightedAverage
    | _ -> 
        Printf.printf "Unknown stacking method '%s', using average\n" !stack_method;
        flush stdout;
        Average
  in
  
  (* Check for input files *)
  let input_files = List.rev !input_files in
  if List.length input_files = 0 then begin
    Printf.printf "Error: No input FITS files provided\n";
    Printf.printf "Usage: %s [options] file1.fits file2.fits ...\n" args.(0);
    Printf.printf "Options:\n";
    Printf.printf "  -o <file>         Output file name (default: stacked_astrometric.fits)\n";
    Printf.printf "  -method <method>  Stacking method: average, median, sigmaclip, kappa, weighted\n";
    Printf.printf "  -sigma <value>    Sigma value for sigmaclip method (default: 3.0)\n";
    Printf.printf "  -kappa <value>    Kappa value for kappa method (default: 3.0)\n";
    false
  end else begin
    Printf.printf "Astrometric Stacking\n";
    Printf.printf "===================\n";
    Printf.printf "Input files: %d\n" (List.length input_files);
    Printf.printf "Stacking method: %s\n" !stack_method;
    Printf.printf "Output file: %s\n\n" !output_file;
    flush stdout;

    (* Perform stacking *)
    print_memory_usage "Before aligned_data creation";
    stack_astrometric input_files 0 method_type !output_file
  end
