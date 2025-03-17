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
let align_direct ref_width ref_height src_width src_height src_wcs ref_wcs =
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
  let trans = Array.make_matrix ref_height ref_width (0.0,0.0) in
  for y = 0 to ref_height - 1 do
    for x = 0 to ref_width - 1 do
      trans.(y).(x) <- transform x y
    done
  done;
  trans

let fill_direct ref_width ref_height src_data src_width src_height src_wcs ref_wcs trans =
  let dst_data = Array.make_matrix ref_height ref_width 0 in
  
  (* Fill destination image using the transform *)
  for y = 0 to ref_height - 1 do
    for x = 0 to ref_width - 1 do
      let (src_x, src_y) = trans.(y).(x) in
      
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

(* In calculate_stack_dimensions, implement a better sizing approach *)
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
  
  (* Collect all corners from all images in sky coordinates *)
  let all_corners = ref [] in
  List.iter (fun (file, wcs, width, height) ->
    let corners = [
      pixel_to_sky wcs 0.0 0.0;
      pixel_to_sky wcs (float_of_int (width-1)) 0.0;
      pixel_to_sky wcs 0.0 (float_of_int (height-1));
      pixel_to_sky wcs (float_of_int (width-1)) (float_of_int (height-1))
    ] in
    all_corners := !all_corners @ corners;
  ) !image_params;
  
  (* Find the min/max RA/Dec to determine the global field size *)
  let ras = List.map fst !all_corners in
  let decs = List.map snd !all_corners in
  let ra_min = List.fold_left min max_float ras in
  let ra_max = List.fold_left max min_float ras in
  let dec_min = List.fold_left min max_float decs in
  let dec_max = List.fold_left max min_float decs in
  
  Printf.printf "Global field bounds: RA=[%.6f, %.6f], Dec=[%.6f, %.6f]\n" 
    ra_min ra_max dec_min dec_max;
  
  (* Map these corners to reference image pixels to find required output size *)
  let corners_in_ref = List.map (fun (ra, dec) ->
    sky_to_pixel ref_wcs ra dec
  ) !all_corners in
  
  let ref_xs = List.map fst corners_in_ref in
  let ref_ys = List.map snd corners_in_ref in
  let min_x = List.fold_left min max_float ref_xs in
  let max_x = List.fold_left max min_float ref_xs in
  let min_y = List.fold_left min max_float ref_ys in
  let max_y = List.fold_left max min_float ref_ys in
  
  (* Calculate required dimensions with margins *)
  let margin = 50 in (* Additional margin to ensure all data is captured *)
  let min_x_with_margin = floor (min_x -. float_of_int margin) in
  let min_y_with_margin = floor (min_y -. float_of_int margin) in
  let max_x_with_margin = ceil (max_x +. float_of_int margin) in
  let max_y_with_margin = ceil (max_y +. float_of_int margin) in
  
  let width_px = int_of_float (max_x_with_margin -. min_x_with_margin) in
  let height_px = int_of_float (max_y_with_margin -. min_y_with_margin) in
  
  Printf.printf "Calculated output dimensions: %d × %d pixels\n" width_px height_px;
  
  (* Calculate the plate scale from reference image *)
  let scale_x = sqrt (ref_wcs.cd1_1 *. ref_wcs.cd1_1 +. ref_wcs.cd2_1 *. ref_wcs.cd2_1) *. 3600.0 in
  let scale_y = sqrt (ref_wcs.cd1_2 *. ref_wcs.cd1_2 +. ref_wcs.cd2_2 *. ref_wcs.cd2_2) *. 3600.0 in
  let avg_scale = (scale_x +. scale_y) /. 2.0 in
  
  Printf.printf "Average plate scale: %.2f arcsec/pixel\n" avg_scale;
  
  (* Create output WCS based on reference image with adjusted CRPIX *)
  let output_wcs = {
    crpix1 = ref_wcs.crpix1 -. min_x_with_margin;
    crpix2 = ref_wcs.crpix2 -. min_y_with_margin;
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
        (max 0 (min stacked_value 32767))
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

  (* In stack_astrometric function, ensure the RGB handling is correct *)
  if is_rgb then begin
    (* Handle RGB stacking *)
    (* Create arrays for each color plane *)
    let stacked_r = Array.make_matrix height width [] in
    let stacked_g = Array.make_matrix height width [] in
    let stacked_b = Array.make_matrix height width [] in

    (* Pre-process each image *)
    let minx = ref 0.0 and miny = ref 0.0 and maxx = ref 0.0 and maxy = ref 0.0 in
    let transforms = List.map (fun (file, wcs, img_width, img_height) ->
      Printf.printf "Processing %s...\n" (Filename.basename file);
      flush stdout;

      (* Calculate transform bounds *)
      let trans = align_direct ref_width ref_height img_width img_height wcs ref_wcs in
      for y = 0 to ref_height - 1 do
	for x = 0 to ref_width - 1 do
	  let (src_x, src_y) = trans.(y).(x) in
          if !minx > src_x then minx := src_x;
          if !miny > src_y then miny := src_y;
          if !maxx < src_x then maxx := src_x;
          if !maxy < src_x then maxy := src_y;
	done
      done;
      trans
    ) image_params in

    (* Process each image *)
    List.iter2 (fun (file, wcs, img_width, img_height) trans ->
      Printf.printf "Processing %s...\n" (Filename.basename file);
      flush stdout;
      let fit = try bool_of_string (Sys.getenv "FIT") with _ -> false in
      let scale = try bool_of_string (Sys.getenv "SCALE") with _ -> false in
      let marginx = try float_of_string (Sys.getenv "MARGINX") with _ -> 50.0 in
      let marginy = try float_of_string (Sys.getenv "MARGINY") with _ -> 50.0 in

      if fit then for y = 0 to ref_height - 1 do
	for x = 0 to ref_width - 1 do
	  let (src_x, src_y) = trans.(y).(x) in
	  trans.(y).(x) <- ((src_x +. marginx -. !minx) *. (if scale then float_of_int ref_width /. (!maxx -. !minx) else 1.0),
                            (src_y +. marginy -. !miny) *. (if scale then float_of_int ref_height /. (!maxy -. !miny) else 1.0));
	done
      done;

      (* Read the image data (all 3 planes) *)
      let _, contents = Fits.find_header_end file (Fits.read_image file) in

      (* Calculate plane size and offsets *)
      let plane_size = img_width * img_height * 2 in (* 16-bit = 2 bytes per pixel *)

      for plane = 0 to 2 do
	let plane_offset = plane * plane_size in

	(* Extract the plane data *)
	let plane_data = Array.make_matrix img_height img_width 0 in
	for y = 0 to img_height - 1 do
	  for x = 0 to img_width - 1 do
	    let off = plane_offset + (y * img_width + x) * 2 in
	    if off + 1 < String.length contents then
	      plane_data.(y).(x) <- (int_of_char contents.[off] lsl 8) lor 
				  (int_of_char contents.[off + 1])
	  done
	done;

	(* Align this plane *)
	let aligned_data = 
	  if file = ref_file then 
	    plane_data
	  else
            begin
            fill_direct ref_width ref_height plane_data img_width img_height wcs ref_wcs trans
            end in

	(* Add to stacked data for this plane *)
	let stacked_plane = match plane with
	  | 0 -> stacked_r
	  | 1 -> stacked_g
	  | _ -> stacked_b
	in

	for y = 0 to height - 1 do
	  for x = 0 to width - 1 do
	    if y < Array.length aligned_data && x < Array.length aligned_data.(0) then
	      stacked_plane.(y).(x) <- aligned_data.(y).(x) :: stacked_plane.(y).(x)
	  done
	done;

	Printf.printf "  Added %s plane %d to stack\n" (Filename.basename file) plane;
      done;
    ) image_params transforms;

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

    (* Write the stacked RGB image *)
    let oc = open_out_bin output_path in

    (* Write header *)
    ignore (Fits.write_fits_header oc header);

    (* Debug info *)
    Printf.printf "Writing RGB planes: R:%dx%d, G:%dx%d, B:%dx%d\n"
      (Array.length output_r) (if Array.length output_r > 0 then Array.length output_r.(0) else 0)
      (Array.length output_g) (if Array.length output_g > 0 then Array.length output_g.(0) else 0)
      (Array.length output_b) (if Array.length output_b > 0 then Array.length output_b.(0) else 0);

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
    Image_stretching.stretch_rgb_fits_file output_path "stretched.fits" Image_stretching.CustomStretch
  end
  else begin
  Printf.printf "Stacked monochrome image skipped\n";
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
