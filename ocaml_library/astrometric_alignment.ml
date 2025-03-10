(* astrometric_alignment.ml - Functions for alignment based on plate solving data *)
open Types
open Fits
open Printf

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

(* Align image using WCS information *)
let align_image_wcs src_data src_width src_height src_wcs dst_wcs dst_width dst_height =
  (* Create output image buffer *)
  let dst_data = Array.make_matrix dst_height dst_width 0 in
  
  (* Create transformation function *)
  let transform = create_wcs_transform dst_wcs src_wcs in
  
  (* Iterate through each pixel in destination image *)
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
    done;
  done;
  
  dst_data

(* Calculate the bounding box that encompasses all images *)
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
          Printf.printf "Warning: No WCS information found in %s\n" file;
          flush stdout
    with e -> 
      Printf.printf "Error processing %s: %s\n" file (Printexc.to_string e);
      flush stdout
  ) files;
  
  if List.length !image_params = 0 then
    failwith "No valid plate-solved images found";
  
  (* Calculate the corners of each image in sky coordinates *)
  let corners = List.map (fun (file, wcs, width, height) ->
    let ra_dec_corners = [
      pixel_to_sky wcs 0.0 0.0;  (* bottom-left *)
      pixel_to_sky wcs (float_of_int (width-1)) 0.0;  (* bottom-right *)
      pixel_to_sky wcs 0.0 (float_of_int (height-1));  (* top-left *)
      pixel_to_sky wcs (float_of_int (width-1)) (float_of_int (height-1))  (* top-right *)
    ] in
    (file, wcs, width, height, ra_dec_corners)
  ) !image_params in
  
  (* Find the min/max RA and Dec across all corners *)
  let ra_min = ref 360.0 in
  let ra_max = ref 0.0 in
  let dec_min = ref 90.0 in
  let dec_max = ref (-90.0) in
  
  List.iter (fun (_, _, _, _, corners) ->
    List.iter (fun (ra, dec) ->
      (* Handle RA wrap around at 0/360 degrees *)
      let ra_norm = 
        if List.exists (fun (_, _, _, _, c) -> 
            List.exists (fun (r, _) -> abs_float (r -. ra) > 180.0) c
          ) corners 
        then
          (* We have images that cross the 0/360 boundary *)
          if ra > 180.0 then ra else ra +. 360.0
        else
          ra
      in
      
      ra_min := min !ra_min ra_norm;
      ra_max := max !ra_max ra_norm;
      dec_min := min !dec_min dec;
      dec_max := max !dec_max dec
    ) corners
  ) corners;
  
  (* Normalize RA back to 0-360 range *)
  let ra_min = if !ra_min >= 360.0 then !ra_min -. 360.0 else !ra_min in
  let ra_max = if !ra_max >= 360.0 then !ra_max -. 360.0 else !ra_max in
  
  (* Select a reference image (middle of the set) *)
  let (ref_file, ref_wcs, ref_width, ref_height, _) = 
    List.nth corners (List.length corners / 2) 
  in
  
  (* Calculate the plate scale (arcsec per pixel) *)
  let scale_x = sqrt (ref_wcs.cd1_1 *. ref_wcs.cd1_1 +. ref_wcs.cd2_1 *. ref_wcs.cd2_1) *. 3600.0 in
  let scale_y = sqrt (ref_wcs.cd1_2 *. ref_wcs.cd1_2 +. ref_wcs.cd2_2 *. ref_wcs.cd2_2) *. 3600.0 in
  let avg_scale = (scale_x +. scale_y) /. 2.0 in
  
  Printf.printf "Average plate scale: %.2f arcsec/pixel\n" avg_scale;
  
  (* Calculate output image size *)
  let ra_span = if ra_max > ra_min then ra_max -. ra_min else (ra_max +. 360.0) -. ra_min in
  let dec_span = !dec_max -. !dec_min in
  
  (* Convert to pixels using the average plate scale *)
  let width_px = int_of_float (Float.ceil (ra_span *. 3600.0 /. avg_scale)) in
  let height_px = int_of_float (Float.ceil (dec_span *. 3600.0 /. avg_scale)) in
  
  Printf.printf "Output dimensions: %d x %d pixels\n" width_px height_px;
  Printf.printf "RA range: %.6f to %.6f (%.6f°)\n" ra_min ra_max ra_span;
  Printf.printf "Dec range: %.6f to %.6f (%.6f°)\n" !dec_min !dec_max dec_span;
  
  (* Create a new WCS for the output image *)
  let output_wcs = {
    crpix1 = 1.0;  (* Reference pixel at the bottom-left *)
    crpix2 = 1.0;
    crval1 = ra_min;
    crval2 = !dec_min;
    cd1_1 = avg_scale /. 3600.0;  (* Convert arcsec/px to deg/px *)
    cd1_2 = 0.0;
    cd2_1 = 0.0;
    cd2_2 = avg_scale /. 3600.0;
    equinox = ref_wcs.equinox;
  } in
  
  (output_wcs, width_px, height_px, avg_scale, !image_params)

(* Stack a list of plate-solved FITS images *)
let stack_astrometric files stacking_method output_path =
  Printf.printf "Stacking %d images using astrometric alignment...\n" (List.length files);
  flush stdout;
  
  (* Calculate the output dimensions and WCS parameters *)
  let (output_wcs, width, height, avg_scale, image_params) = calculate_stack_dimensions files in
  
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

(* Standalone tool for stacking astrometric images *)
let astrometric_stack_cli args =
  (* Command line arguments *)
  let output_file = ref "stacked_astrometric.fits" in
  let stack_method = ref "average" in
  let sigma_value = ref 3.0 in
  let kappa_value = ref 3.0 in
  let input_files = ref [] in
  
  (* Parse command line *)
  let i = ref 0 in
  while !i < Array.length args do
    match args.(!i) with
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
    ;
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
    exit 1
  end;
  
  Printf.printf "Astrometric Stacking\n";
  Printf.printf "===================\n";
  Printf.printf "Input files: %d\n" (List.length input_files);
  Printf.printf "Stacking method: %s\n" !stack_method;
  Printf.printf "Output file: %s\n\n" !output_file;
  flush stdout;
  
  (* Perform stacking *)
  stack_astrometric input_files method_type !output_file
