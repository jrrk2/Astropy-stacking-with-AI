(* astrometric_stack_cli.ml - Command line tool for astrometric stacking 
 * Enhanced to support both plate-solved FITS keywords and live stacking matrices
 *)
open Types
open Fits
open Stack_debug
open Astrometric_alignment

type alignment_source = 
  | PlateSolved   (* Use WCS keywords from plate solving *)
  | LiveStacking  (* Use transformation matrices provided by live stacking *)
  | Auto          (* Automatically use what's available *)

(* Updated structure to handle both alignment types *)
type alignment_info = 
  | PlateInfo of wcs_params_solved
  | LiveInfo of matrix_data

type interp_params = {
  src_x: float;
  src_y: float;
  src_x_int: int;
  src_y_int: int;
  x_frac: float;
  y_frac: float;
  in_bounds: bool;
}

let usage = "Usage: astrometric_stack_cli [options] file1.fits file2.fits ...\n\
            \n\
            Options:\n\
            \  -o <file>         Output file name (default: stacked_astrometric.fits)\n\
            \  -method <method>  Stacking method: average, median, sigmaclip, kappa, weighted\n\
            \  -sigma <value>    Sigma value for sigmaclip method (default: 3.0)\n\
            \  -kappa <value>    Kappa value for kappa method (default: 3.0)\n\
            \  -list <file>      Read input file list from file\n\
            \  -verbose          Enable verbose output\n\
            \  -align <source>   Alignment source: plate-solved, live-stack, auto (default: auto)\n\
            \  -ref <index>      Reference frame index (default: 0)\n\
            \n\
            Examples:\n\
            \  astrometric_stack_cli -o result.fits -method median image1.fits image2.fits image3.fits\n\
            \  astrometric_stack_cli -list images.txt -method sigmaclip -sigma 2.5\n\
            \  astrometric_stack_cli -align live-stack -method average *.fits\n"

(* Check if header contains live stacking keywords *)
let has_live_stack_matrix header =
  try
    let _ = parse_float header "CD1_1M" in
    let _ = parse_float header "CD1_2M" in
    let _ = parse_float header "CD1_3M" in
    let _ = parse_float header "CD2_1M" in
    let _ = parse_float header "CD2_2M" in
    let _ = parse_float header "CD2_3M" in
    true
  with _ -> false

(* Check if header contains plate solving WCS keywords *)
let has_plate_solving_wcs header =
  try
    let _ = parse_float header "CD1_1" in
    let _ = parse_float header "CD1_2" in
    let _ = parse_float header "CD2_1" in
    let _ = parse_float header "CD2_2" in
    let _ = parse_float header "CRPIX1" in
    let _ = parse_float header "CRPIX2" in
    let _ = parse_float header "CRVAL1" in
    let _ = parse_float header "CRVAL2" in
    true
  with _ -> false

(* Get the appropriate alignment information based on source preference *)
let get_alignment_info filename source =
  let header = just_header filename in
  
  match source with
  | PlateSolved ->
      (match extract_wcs_params header with
       | Some wcs -> Some (PlateInfo wcs)
       | None -> None)
  | LiveStacking ->
      (match extract_live_stack_matrix header filename false with
       | Some matrix -> Some (LiveInfo matrix)
       | None -> None)
  | Auto ->
      (* Try plate solved first, then live stacking *)
      if has_plate_solving_wcs header then
        match extract_wcs_params header with
        | Some wcs -> Some (PlateInfo wcs)
        | None -> None
      else if has_live_stack_matrix header then
        match extract_live_stack_matrix header filename false with
        | Some matrix -> Some (LiveInfo matrix)
        | None -> None
      else
        None

(* Align image using the live stacking matrix directly *)
let align_with_live_stack src_data src_width src_height src_matrix dst_matrix dst_width dst_height =
  (* Create output image buffer *)
  let dst_data = Array.make_matrix dst_height dst_width 0 in
  
  (* Each matrix already represents the full transformation for its image.
     To transform from dst to src coordinates, we need:
     1. Apply dst_matrix inverse (to get back to "reference" space)
     2. Apply src_matrix (to get to source image space) *)
  
  (* Calculate dst matrix inverse *)
  let det_dst = dst_matrix.tel_m11 *. dst_matrix.tel_m22 -. dst_matrix.tel_m12 *. dst_matrix.tel_m21 in
  
  if abs_float det_dst < 1e-10 then
    failwith "Singular destination matrix in live stacking transform";
  
  (* Compute inverse of destination matrix *)
  let dst_m11_inv = dst_matrix.tel_m22 /. det_dst in
  let dst_m12_inv = -. dst_matrix.tel_m12 /. det_dst in
  let dst_m21_inv = -. dst_matrix.tel_m21 /. det_dst in
  let dst_m22_inv = dst_matrix.tel_m11 /. det_dst in
  
  (* Calculate translation part of inverse *)
  let dst_m13_inv = -. (dst_m11_inv *. dst_matrix.tel_m13 +. dst_m12_inv *. dst_matrix.tel_m23) in
  let dst_m23_inv = -. (dst_m21_inv *. dst_matrix.tel_m13 +. dst_m22_inv *. dst_matrix.tel_m23) in
  
  (* Define transformation function that applies dst_inv then src *)
  let transform x y =
    (* First apply dst_matrix inverse to get to "reference" space *)
    let ref_x = dst_m11_inv *. x +. dst_m12_inv *. y +. dst_m13_inv in
    let ref_y = dst_m21_inv *. x +. dst_m22_inv *. y +. dst_m23_inv in
    
    (* Then apply src_matrix to get to source image space *)
    let src_x = src_matrix.tel_m11 *. ref_x +. src_matrix.tel_m12 *. ref_y +. src_matrix.tel_m13 in
    let src_y = src_matrix.tel_m21 *. ref_x +. src_matrix.tel_m22 *. ref_y +. src_matrix.tel_m23 in
    
    (src_x, src_y)
  in
  
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

let dump_rgb_fits output_path ref_hdr output_r output_g output_b ref_width ref_height stacking_method =

  (* Create FITS header for output file *)
  let header = Hashtbl.create 50 in

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
  Hashtbl.add header "NAXIS1" (Printf.sprintf " = %d / Width in pixels" ref_width);
  Hashtbl.add header "NAXIS2" (Printf.sprintf " = %d / Height in pixels" ref_height);
  Hashtbl.add header "NAXIS3" " = 3 / Number of color planes (RGB)";
  Hashtbl.add header "EXTEND" " = T / Extensions may be present";
  Hashtbl.add header "BZERO" " = 32768 / Offset to unsigned short range";
  Hashtbl.add header "BSCALE" " = 1 / Default scaling factor";

  (* Add metadata about stacking *)
  Hashtbl.add header "HISTORY" " Stacked with OCaml Live Stack Alignment";
  Hashtbl.add header "HISTORY" (Printf.sprintf " Stacking method: %s" 
    (match stacking_method with
     | Average -> "Average"
     | Median -> "Median"
     | SigmaClip sigma -> Printf.sprintf "SigmaClip (%.1f)" sigma
     | Kappa k -> Printf.sprintf "Kappa (%.1f)" k
     | WeightedAverage -> "WeightedAverage"));

  (* Write the stacked RGB image *)
  let oc = open_out_bin output_path in

  (* Write header *)
  ignore (write_fits_header oc header);

  (* Write red plane *)
  for y = 0 to ref_height - 1 do
    for x = 0 to ref_width - 1 do
      output_byte oc (output_r.(y).(x) lsr 8);
      output_byte oc (output_r.(y).(x) land 0xFF);
    done
  done;

  (* Write green plane *)
  for y = 0 to ref_height - 1 do
    for x = 0 to ref_width - 1 do
      output_byte oc (output_g.(y).(x) lsr 8);
      output_byte oc (output_g.(y).(x) land 0xFF);
    done
  done;

  (* Write blue plane *)
  for y = 0 to ref_height - 1 do
    for x = 0 to ref_width - 1 do
      output_byte oc (output_b.(y).(x) lsr 8);
      output_byte oc (output_b.(y).(x) land 0xFF);
    done
  done;

  (* Pad data to multiple of 2880 bytes *)
  let data_size = ref_width * ref_height * 2 * 3 in
  let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
  output_string oc (String.make padding_size '\000');

  close_out oc

(* Enhanced stack function that can use either alignment source *)
let stack_images files ref_idx stacking_method output_path alignment_source =
  let reference_file = List.nth files ref_idx in
  Printf.printf "Using reference frame: %s\n" reference_file;

  (* Add this call for all files being processed *)
  List.iter (fun file ->
    if file <> reference_file then
      dump_transformation_comparison reference_file file
  ) files;

  (* Determine what alignment source to use and get reference information *)
  let (actual_source, ref_info) =
    match get_alignment_info reference_file alignment_source with
    | Some (PlateInfo wcs) -> 
        Printf.printf "Using plate-solved WCS keywords for alignment\n";
        (PlateSolved, PlateInfo wcs)
    | Some (LiveInfo matrix) -> 
        Printf.printf "Using live stacking matrices for alignment\n";
        (LiveStacking, LiveInfo matrix)
    | None ->
        (* If we can't get alignment info using the preferred method, try auto *)
        if alignment_source <> Auto then
          match get_alignment_info reference_file Auto with
          | Some (PlateInfo wcs) -> 
              Printf.printf "Falling back to plate-solved WCS keywords\n";
              (PlateSolved, PlateInfo wcs)
          | Some (LiveInfo matrix) -> 
              Printf.printf "Falling back to live stacking matrices\n";
              (LiveStacking, LiveInfo matrix)
          | None ->
              failwith (Printf.sprintf "No alignment information found in %s" reference_file)
        else
          failwith (Printf.sprintf "No alignment information found in %s" reference_file)
  in
  
  (* Branch based on the alignment source *)
  match actual_source with
  | Auto -> failwith "not supported: auto"
  | PlateSolved ->
      (* If using plate-solved, call the existing function *)
      (match ref_info with
      | PlateInfo _ -> stack_astrometric files ref_idx stacking_method output_path
      | _ -> failwith "Invalid reference info type for plate solving")
      
  | LiveStacking ->
      (* Extract the reference matrix *)
      let ref_matrix = match ref_info with
        | LiveInfo matrix -> matrix
        | _ -> failwith "Invalid reference info type for live stacking"
      in
      
      (* Get reference frame properties *)
      let ref_width = ref_matrix.width in
      let ref_height = ref_matrix.height in
      Printf.printf "Reference dimensions: %d x %d\n" ref_width ref_height;
      
      (* Check if images are RGB or monochrome *)
      let is_rgb = 
        try
          let header = just_header reference_file in
          let naxis = parse_int header "NAXIS" in
          naxis > 2
        with _ -> false
      in
      
      if is_rgb then begin
        (* Handle RGB stacking *)
        Printf.printf "Processing RGB images...\n";
        
        (* Create arrays for each color plane *)
        let stacked_r = Array.make_matrix ref_height ref_width [] in
        let stacked_g = Array.make_matrix ref_height ref_width [] in
        let stacked_b = Array.make_matrix ref_height ref_width [] in
        
        (* Process each image file *)
        List.iter (fun file ->
          (* Skip files that don't have matrix data *)
          match get_alignment_info file LiveStacking with
          | Some (LiveInfo matrix) -> begin
              Printf.printf "Processing %s...\n" (Filename.basename file);
              
              (* Read the image data (all 3 planes) *)
              let header, contents = find_header_end file (read_image file) in
              
              (* Get dimensions and check consistency *)
              let width = parse_int header "NAXIS1" in
              let height = parse_int header "NAXIS2" in
              
              (* Calculate plane size and offsets *)
              let plane_size = width * height * 2 in (* 16-bit = 2 bytes per pixel *)
              
              (* Extract and align each color plane *)
              for plane = 0 to 2 do
                let plane_offset = plane * plane_size in
                
                (* Extract the plane data *)
                let plane_data = Array.make_matrix height width 0 in
                for y = 0 to height - 1 do
                  for x = 0 to width - 1 do
                    let offset = plane_offset + (y * width + x) * 2 in
                    if offset + 1 < String.length contents then
                      plane_data.(y).(x) <- (int_of_char contents.[offset] lsl 8) lor 
                                          (int_of_char contents.[offset + 1])
                  done
                done;
                
                (* Align this plane using live stacking matrix *)
                let aligned_data = align_with_live_stack plane_data width height matrix ref_matrix ref_width ref_height in
                
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
            end
          | _ -> Printf.printf "Warning: No live stacking matrix found in %s. Skipping.\n" file
        ) files;
        
        (* Apply stacking method to each pixel in each plane *)
        let output_r = Array.make_matrix ref_height ref_width 0 in
        let output_g = Array.make_matrix ref_height ref_width 0 in
        let output_b = Array.make_matrix ref_height ref_width 0 in
        
        for y = 0 to ref_height - 1 do
          for x = 0 to ref_width - 1 do
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
          if ref_height > 1000 && y mod 100 = 0 then begin
            Printf.printf "  Stacking progress: %.1f%%\n" (float_of_int y *. 100.0 /. float_of_int ref_height);
            flush stdout
          end
        done;

	(* Get header from reference file *)
	let ref_hdr = just_header reference_file in
        dump_rgb_fits output_path ref_hdr output_r output_g output_b ref_width ref_height stacking_method;

        Printf.printf "Stacked RGB image saved to %s\n" output_path;
        true
      end
      else begin
        (* Handle monochrome stacking *)
        Printf.printf "Processing monochrome images...\n";
        
        (* Create an array to accumulate pixel values *)
        let stacked_data = Array.make_matrix ref_height ref_width [] in
        
        (* Process each image *)
        List.iter (fun file ->
          (* Skip files that don't have matrix data *)
          match get_alignment_info file LiveStacking with
          | Some (LiveInfo matrix) -> begin
              Printf.printf "Processing %s...\n" (Filename.basename file);
              
              (* Read the image data *)
              let header, contents = find_header_end file (read_image file) in
              
              (* Get dimensions *)
              let width = parse_int header "NAXIS1" in
              let height = parse_int header "NAXIS2" in
              
              (* Read the image data into array *)
              let data = read_fits_data contents width height in
              
              (* Align to the reference frame using live stacking matrix *)
              let aligned_data = align_with_live_stack data width height matrix ref_matrix ref_width ref_height in
              
              (* Add to stacked data *)
              for y = 0 to ref_height - 1 do
                for x = 0 to ref_width - 1 do
                  let value = aligned_data.(y).(x) in
                  if value > 0 then
                    stacked_data.(y).(x) <- value :: stacked_data.(y).(x)
                done
              done;
              
              Printf.printf "  Added %s to stack\n" (Filename.basename file);
            end
          | _ -> Printf.printf "Warning: No live stacking matrix found in %s. Skipping.\n" file
        ) files;
        
        (* Apply stacking method to each pixel *)
        let output_data = Array.make_matrix ref_height ref_width 0 in
        
        for y = 0 to ref_height - 1 do
          for x = 0 to ref_width - 1 do
            let values = Array.of_list (stacked_data.(y).(x)) in
            let stacked_value = apply_stacking_method values stacking_method in
            output_data.(y).(x) <- stacked_value
          done;
          
          (* Print progress for large images *)
          if ref_height > 1000 && y mod 100 = 0 then begin
            Printf.printf "  Stacking progress: %.1f%%\n" (float_of_int y *. 100.0 /. float_of_int ref_height);
            flush stdout
          end
        done;
        
        (* Create FITS header for output file *)
        let header = Hashtbl.create 50 in
        
        (* Get header from reference file *)
        let ref_hdr = just_header reference_file in
        
        (* Copy important keywords *)
        List.iter (fun key ->
          if key <> "NAXIS" && key <> "NAXIS1" && key <> "NAXIS2" then
            match Hashtbl.find_opt ref_hdr key with
            | Some value -> Hashtbl.add header key value
            | None -> ()
        ) ["SIMPLE"; "BITPIX"; "BZERO"; "BSCALE"; "DATE-OBS"; "INSTRUME"; "EXPOSURE"; "FOCAL"; "PIXSZ"; "EXTEND"];
        
        (* Set dimensions *)
        Hashtbl.add header "SIMPLE" " = T / FITS standard";
        Hashtbl.add header "BITPIX" " = 16 / 16-bit signed integers";
        Hashtbl.add header "NAXIS" " = 2 / Number of axes";
        Hashtbl.add header "NAXIS1" (Printf.sprintf " = %d / Width in pixels" ref_width);
        Hashtbl.add header "NAXIS2" (Printf.sprintf " = %d / Height in pixels" ref_height);
        Hashtbl.add header "EXTEND" " = T / Extensions may be present";
        
        (* Add metadata about stacking *)
        Hashtbl.add header "HISTORY" " Stacked with OCaml Live Stack Alignment";
        Hashtbl.add header "HISTORY" (Printf.sprintf " Stacking method: %s" 
          (match stacking_method with
           | Average -> "Average"
           | Median -> "Median"
           | SigmaClip sigma -> Printf.sprintf "SigmaClip (%.1f)" sigma
           | Kappa k -> Printf.sprintf "Kappa (%.1f)" k
           | WeightedAverage -> "WeightedAverage"));
        
        (* Write the stacked image *)
        let oc = open_out_bin output_path in
        
        (* Write header *)
        ignore (write_fits_header oc header);
        
        (* Write data *)
        for y = 0 to ref_height - 1 do
          for x = 0 to ref_width - 1 do
            let value = output_data.(y).(x) in
            output_byte oc (value lsr 8);
            output_byte oc (value land 0xFF);
          done
        done;
        
        (* Pad data to multiple of 2880 bytes *)
        let data_size = ref_width * ref_height * 2 in
        let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
        output_string oc (String.make padding_size '\000');
        
        close_out oc;
        
        Printf.printf "Stacked image saved to %s\n" output_path;
        true
      end

(* Add this function to your astrometric_stack_cli.ml file to enable unified stacking *)

let stack_with_unified_wcs files ref_idx stacking_method output_path =
  let reference_file = List.nth files ref_idx in
  Printf.printf "Using reference frame: %s\n" reference_file;
  
  (* Add this call for all files being processed - debug output *)
  List.iter (fun file ->
    if file <> reference_file then
      dump_transformation_comparison reference_file file
  ) files;
  
  (* Process reference file *)
  let ref_hdr = Fits.just_header reference_file in
  let ref_wcs_opt = extract_wcs_params ref_hdr in
  let ref_live_opt = extract_live_stack_matrix ref_hdr reference_file true in
  
  (* Get reference WCS - either directly from WCS headers or converted from matrix *)
  let ref_wcs = match ref_wcs_opt, ref_live_opt with
    | Some wcs, _ -> 
        Printf.printf "Using WCS parameters from reference frame\n";
        wcs
    | None, Some matrix ->
        Printf.printf "Converting live matrix to WCS for reference frame\n";
        convert_live_matrix_to_wcs matrix
    | None, None ->
        failwith "No transformation information found in reference frame"
  in
  
  (* Create RGB arrays for different stacking methods *)
  let is_rgb = 
    try
      let naxis = parse_int ref_hdr "NAXIS" in
      naxis = 3
    with _ -> false
  in
  
  if is_rgb then
    Printf.printf "Processing RGB images\n"
  else
    Printf.printf "Processing monochrome images\n";
  
  (* Set up stacking arrays based on image type *)
  let width = parse_int ref_hdr "NAXIS1" in
  let height = parse_int ref_hdr "NAXIS2" in
  
  (* Initialize stacking arrays *)
  let stacked_data = 
    if is_rgb then
      (Array.make_matrix height width [],  (* R *)
       Array.make_matrix height width [],  (* G *)
       Array.make_matrix height width [])  (* B *)
    else
      (Array.make_matrix height width [], (* Single channel *)
       [||], [||])
  in
  
  (* Process each image file *)
  List.iter (fun file ->
    Printf.printf "Processing %s...\n" (Filename.basename file);
    
    (* Get WCS info for this image - either from WCS headers or converted from matrix *)
    let file_hdr = Fits.just_header file in
    let file_wcs_opt = extract_wcs_params file_hdr in
    let file_live_opt = extract_live_stack_matrix file_hdr file false in
    
    let file_wcs = match file_wcs_opt, file_live_opt with
      | Some wcs, _ -> 
          Printf.printf "  Using WCS parameters\n";
          wcs
      | None, Some matrix ->
          Printf.printf "  Converting live matrix to WCS\n";
          convert_live_matrix_to_wcs matrix
      | None, None ->
          Printf.printf "  Warning: No transformation found, skipping %s\n" file;
          raise (Failure "No transformation information")
    in
    
    (* Create transformation function *)
    let transform = create_wcs_transform file_wcs ref_wcs in
    
    (* Read image data *)
    let header, contents = find_header_end file (read_image file) in
    
    (* Process based on image type *)
    if is_rgb then begin
      (* RGB image processing *)
      let naxis3 = parse_int header "NAXIS3" in
      if naxis3 < 3 then
        Printf.printf "  Warning: Expected RGB but found only %d channels\n" naxis3
      else begin
        let (r_stack, g_stack, b_stack) = stacked_data in
        
        (* Calculate plane size and offsets *)
        let plane_size = width * height * 2 in  (* 16-bit pixels *)

	(* Outside the plane loop, pre-calculate ALL interpolation parameters *)

	(* Create the interpolation parameter matrix *)
	let interp_matrix = Array.make_matrix height width 
	  {src_x=0.0; src_y=0.0; src_x_int=0; src_y_int=0; x_frac=0.0; y_frac=0.0; in_bounds=false} in

	(* Pre-calculate all transformations and interpolation parameters *)
	for dst_y = 0 to height - 1 do
	  for dst_x = 0 to width - 1 do
	    (* Apply the transformation *)
	    let src_x, src_y = transform (float_of_int dst_x) (float_of_int dst_y) in

	    (* Calculate interpolation parameters *)
	    let src_x_floor = floor src_x in
	    let src_y_floor = floor src_y in
	    let src_x_int = int_of_float src_x_floor in
	    let src_y_int = int_of_float src_y_floor in
	    let x_frac = src_x -. src_x_floor in
	    let y_frac = src_y -. src_y_floor in

	    (* Check bounds *)
	    let in_bounds = 
	      src_x >= 0.0 && src_x < float_of_int width -. 1.0 &&
	      src_y >= 0.0 && src_y < float_of_int height -. 1.0
	    in

	    (* Store all parameters *)
	    interp_matrix.(dst_y).(dst_x) <- {
	      src_x; src_y; src_x_int; src_y_int; x_frac; y_frac; in_bounds
	    };
	  done
	done;

	(* Now process each plane using the exact same interpolation parameters *)
	for plane = 0 to 2 do
	  let plane_data = Array.make_matrix height width 0 in
	  let plane_size = width * height * 2 in  (* 16-bit pixels = 2 bytes *)
	  let plane_offset = plane * plane_size in

	  (* Extract plane data *)
	  for y = 0 to height - 1 do
	    for x = 0 to width - 1 do
	      let offset = plane_offset + (y * width + x) * 2 in
	      if offset + 1 < String.length contents then
		plane_data.(y).(x) <- (int_of_char contents.[offset] lsl 8) lor 
				    (int_of_char contents.[offset + 1])
	    done
	  done;

	  for dst_y = 0 to height - 1 do
	    for dst_x = 0 to width - 1 do
	      let params = interp_matrix.(dst_y).(dst_x) in

	      if params.in_bounds then begin
		(* Get the four surrounding pixels using identical coordinates *)
		let p00 = plane_data.(params.src_y_int).(params.src_x_int) in
		let p10 = plane_data.(params.src_y_int).(params.src_x_int + 1) in
		let p01 = plane_data.(params.src_y_int + 1).(params.src_x_int) in
		let p11 = plane_data.(params.src_y_int + 1).(params.src_x_int + 1) in

		(* Interpolate using identical fractions *)
		let value = 
		  float_of_int p00 *. (1.0 -. params.x_frac) *. (1.0 -. params.y_frac) +.
		  float_of_int p10 *. params.x_frac *. (1.0 -. params.y_frac) +.
		  float_of_int p01 *. (1.0 -. params.x_frac) *. params.y_frac +.
		  float_of_int p11 *. params.x_frac *. params.y_frac
		in

		(* Store the interpolated value for this plane *)
		let interp_value = int_of_float (Float.round value) in
		match plane with
		| 0 -> r_stack.(dst_y).(dst_x) <- interp_value :: r_stack.(dst_y).(dst_x)
		| 1 -> g_stack.(dst_y).(dst_x) <- interp_value :: g_stack.(dst_y).(dst_x)
		| _ -> b_stack.(dst_y).(dst_x) <- interp_value :: b_stack.(dst_y).(dst_x)
		end
	    done
	  done;
          Printf.printf "  Added %s plane %d to stack\n" (Filename.basename file) plane;
        done
      end
    end else begin
      (* Monochrome image processing *)
      let (mono_stack, _, _) = stacked_data in
      let data = read_fits_data contents width height in
      
      (* Apply transformation and add to stack *)
      for dst_y = 0 to height - 1 do
        for dst_x = 0 to width - 1 do
          (* Apply the transformation *)
          let src_x, src_y = transform (float_of_int dst_x) (float_of_int dst_y) in
          
          (* Check if source coordinates are within bounds *)
          if src_x >= 0.0 && src_x < float_of_int width -. 1.0 &&
             src_y >= 0.0 && src_y < float_of_int height -. 1.0 then begin
            
            (* Bilinear interpolation *)
            let src_x_floor = floor src_x in
            let src_y_floor = floor src_y in
            let src_x_int = int_of_float src_x_floor in
            let src_y_int = int_of_float src_y_floor in
            
            let x_frac = src_x -. src_x_floor in
            let y_frac = src_y -. src_y_floor in
            
            let p00 = float_of_int data.(src_y_int).(src_x_int) in
            let p10 = float_of_int data.(src_y_int).(src_x_int + 1) in
            let p01 = float_of_int data.(src_y_int + 1).(src_x_int) in
            let p11 = float_of_int data.(src_y_int + 1).(src_x_int + 1) in
            
            let value = 
              p00 *. (1.0 -. x_frac) *. (1.0 -. y_frac) +.
              p10 *. x_frac *. (1.0 -. y_frac) +.
              p01 *. (1.0 -. x_frac) *. y_frac +.
              p11 *. x_frac *. y_frac
            in
            
            mono_stack.(dst_y).(dst_x) <- int_of_float (Float.round value) :: mono_stack.(dst_y).(dst_x)
          end
        done
      done;
      
      Printf.printf "  Added %s to stack\n" (Filename.basename file);
    end
  ) files;
  
  (* Apply stacking method to create final image *)
  if is_rgb then begin
    (* RGB stacking *)
    let (r_stack, g_stack, b_stack) = stacked_data in
    let r_output = Array.make_matrix height width 0 in
    let g_output = Array.make_matrix height width 0 in
    let b_output = Array.make_matrix height width 0 in
    
    (* Apply stacking method to each pixel in each plane *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        (* Process each color plane separately *)
        let r_values = Array.of_list r_stack.(y).(x) in
        let g_values = Array.of_list g_stack.(y).(x) in
        let b_values = Array.of_list b_stack.(y).(x) in
        
        r_output.(y).(x) <- apply_stacking_method r_values stacking_method;
        g_output.(y).(x) <- apply_stacking_method g_values stacking_method;
        b_output.(y).(x) <- apply_stacking_method b_values stacking_method;
      done;
      
      (* Progress indicator *)
      if height > 1000 && y mod 100 = 0 then
        Printf.printf "  Stacking progress: %.1f%%\n" (float_of_int y /. float_of_int height *. 100.0);
    done;
    
    (* Create FITS header for output *)
    let header = Hashtbl.create 50 in
    
    (* Copy basic headers from reference *)
    List.iter (fun key ->
      match Hashtbl.find_opt ref_hdr key with
      | Some value -> Hashtbl.add header key value
      | None -> ()
    ) ["SIMPLE"; "BITPIX"; "BZERO"; "BSCALE"; "DATE-OBS"; "INSTRUME"; "EXPOSURE"; "FOCAL"; "PIXSZ"; "EXTEND"];
    
    (* Set RGB dimensions *)
    Hashtbl.add header "SIMPLE" " = T / FITS standard";
    Hashtbl.add header "BITPIX" " = 16 / 16-bit integers";
    Hashtbl.add header "NAXIS" " = 3 / Number of data axes";
    Hashtbl.add header "NAXIS1" (Printf.sprintf " = %d / Width in pixels" width);
    Hashtbl.add header "NAXIS2" (Printf.sprintf " = %d / Height in pixels" height);
    Hashtbl.add header "NAXIS3" " = 3 / Number of color planes (RGB)";
    Hashtbl.add header "EXTEND" " = T / Extensions may be present";
    Hashtbl.add header "BZERO" " = 32768 / Offset to unsigned short range";
    Hashtbl.add header "BSCALE" " = 1 / Default scaling factor";
    
    (* Add WCS parameters from reference image *)
    Hashtbl.add header "CTYPE1" " = 'RA---TAN' / Right ascension, tangent projection";
    Hashtbl.add header "CTYPE2" " = 'DEC--TAN' / Declination, tangent projection";
    Hashtbl.add header "CRPIX1" (Printf.sprintf " = %.6f / X reference pixel" ref_wcs.crpix1);
    Hashtbl.add header "CRPIX2" (Printf.sprintf " = %.6f / Y reference pixel" ref_wcs.crpix2);
    Hashtbl.add header "CRVAL1" (Printf.sprintf " = %.10f / RA at reference pixel (deg)" ref_wcs.crval1);
    Hashtbl.add header "CRVAL2" (Printf.sprintf " = %.10f / Dec at reference pixel (deg)" ref_wcs.crval2);
    Hashtbl.add header "CD1_1" (Printf.sprintf " = %.10e / Transformation matrix element" ref_wcs.cd1_1);
    Hashtbl.add header "CD1_2" (Printf.sprintf " = %.10e / Transformation matrix element" ref_wcs.cd1_2);
    Hashtbl.add header "CD2_1" (Printf.sprintf " = %.10e / Transformation matrix element" ref_wcs.cd2_1);
    Hashtbl.add header "CD2_2" (Printf.sprintf " = %.10e / Transformation matrix element" ref_wcs.cd2_2);
    Hashtbl.add header "EQUINOX" (Printf.sprintf " = %.1f / Equinox of coordinates" ref_wcs.equinox);
    
    (* Add metadata about stacking *)
    Hashtbl.add header "HISTORY" " Stacked with Unified WCS-Matrix Alignment";
    Hashtbl.add header "HISTORY" (Printf.sprintf " Stacking method: %s" 
      (match stacking_method with
       | Average -> "Average"
       | Median -> "Median"
       | SigmaClip sigma -> Printf.sprintf "SigmaClip (%.1f)" sigma
       | Kappa k -> Printf.sprintf "Kappa (%.1f)" k
       | WeightedAverage -> "WeightedAverage"));
    Hashtbl.add header "HISTORY" (Printf.sprintf " Number of frames: %d" (List.length files));
    
    (* Write the stacked RGB image *)
    dump_rgb_fits output_path header r_output g_output b_output width height stacking_method;
    
    Printf.printf "Stacked RGB image saved to %s\n" output_path;
    true
  end else begin
    (* Monochrome stacking *)
    let (mono_stack, _, _) = stacked_data in
    let output_data = Array.make_matrix height width 0 in
    
    (* Apply stacking method to each pixel *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let values = Array.of_list mono_stack.(y).(x) in
        output_data.(y).(x) <- apply_stacking_method values stacking_method;
      done;
      
      (* Progress indicator *)
      if height > 1000 && y mod 100 = 0 then
        Printf.printf "  Stacking progress: %.1f%%\n" (float_of_int y /. float_of_int height *. 100.0);
    done;
    
    (* Create header for output file *)
    let header = Hashtbl.create 50 in
    
    (* Copy basic headers from reference *)
    List.iter (fun key ->
      match Hashtbl.find_opt ref_hdr key with
      | Some value -> Hashtbl.add header key value
      | None -> ()
    ) ["SIMPLE"; "BITPIX"; "BZERO"; "BSCALE"; "DATE-OBS"; "INSTRUME"; "EXPOSURE"; "FOCAL"; "PIXSZ"; "EXTEND"];
    
    (* Set dimensions *)
    Hashtbl.add header "SIMPLE" " = T / FITS standard";
    Hashtbl.add header "BITPIX" " = 16 / 16-bit integers";
    Hashtbl.add header "NAXIS" " = 2 / Number of axes";
    Hashtbl.add header "NAXIS1" (Printf.sprintf " = %d / Width in pixels" width);
    Hashtbl.add header "NAXIS2" (Printf.sprintf " = %d / Height in pixels" height);
    Hashtbl.add header "EXTEND" " = T / Extensions may be present";
    
    (* Add WCS parameters from reference image *)
    Hashtbl.add header "CTYPE1" " = 'RA---TAN' / Right ascension, tangent projection";
    Hashtbl.add header "CTYPE2" " = 'DEC--TAN' / Declination, tangent projection";
    Hashtbl.add header "CRPIX1" (Printf.sprintf " = %.6f / X reference pixel" ref_wcs.crpix1);
    Hashtbl.add header "CRPIX2" (Printf.sprintf " = %.6f / Y reference pixel" ref_wcs.crpix2);
    Hashtbl.add header "CRVAL1" (Printf.sprintf " = %.10f / RA at reference pixel (deg)" ref_wcs.crval1);
    Hashtbl.add header "CRVAL2" (Printf.sprintf " = %.10f / Dec at reference pixel (deg)" ref_wcs.crval2);
    Hashtbl.add header "CD1_1" (Printf.sprintf " = %.10e / Transformation matrix element" ref_wcs.cd1_1);
    Hashtbl.add header "CD1_2" (Printf.sprintf " = %.10e / Transformation matrix element" ref_wcs.cd1_2);
    Hashtbl.add header "CD2_1" (Printf.sprintf " = %.10e / Transformation matrix element" ref_wcs.cd2_1);
    Hashtbl.add header "CD2_2" (Printf.sprintf " = %.10e / Transformation matrix element" ref_wcs.cd2_2);
    Hashtbl.add header "EQUINOX" (Printf.sprintf " = %.1f / Equinox of coordinates" ref_wcs.equinox);
    
    (* Add metadata about stacking *)
    Hashtbl.add header "HISTORY" " Stacked with Unified WCS-Matrix Alignment";
    Hashtbl.add header "HISTORY" (Printf.sprintf " Stacking method: %s" 
      (match stacking_method with
       | Average -> "Average"
       | Median -> "Median"
       | SigmaClip sigma -> Printf.sprintf "SigmaClip (%.1f)" sigma
       | Kappa k -> Printf.sprintf "Kappa (%.1f)" k
       | WeightedAverage -> "WeightedAverage"));
    
    (* Write the stacked image *)
    let oc = open_out_bin output_path in
    
    (* Write header *)
    ignore (Fits.write_fits_header oc header);
    
    (* Write data *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
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
    true
  end

(* Main entry point for the CLI *)
let () =
  let output_file = ref "stacked_astrometric.fits" in
  let stack_method = ref "average" in
  let sigma_value = ref 3.0 in
  let kappa_value = ref 3.0 in
  let verbose = ref false in
  let input_files = ref [] in
  let file_list = ref None in
  let alignment_source_str = ref "auto" in
  let ref_idx = ref 0 in
  let unified_wcs = ref true in

  (* Define command line arguments *)
  let specs = [
    ("-o", Arg.Set_string output_file, "Output file name");
    ("-method", Arg.Set_string stack_method, "Stacking method (average, median, sigmaclip, kappa, weighted)");
    ("-sigma", Arg.Set_float sigma_value, "Sigma value for sigmaclip method");
    ("-kappa", Arg.Set_float kappa_value, "Kappa value for kappa method");
    ("-list", Arg.String (fun f -> file_list := Some f), "File containing list of input files");
    ("-verbose", Arg.Set verbose, "Enable verbose output");
    ("-unified-wcs", Arg.Set unified_wcs, "Enable unified_wcs");
    ("-align", Arg.Set_string alignment_source_str, "Alignment source (plate-solved, live-stack, auto)");
    ("-ref", Arg.Set_int ref_idx, "Reference frame index (default: 0)");
  ] in
  
  (* Parse command line *)
  let add_file file =
    if Filename.check_suffix file ".fits" || Filename.check_suffix file ".fit" then
      input_files := file :: !input_files
    else
      Printf.printf "Skipping non-FITS file: %s\n" file
  in
  
  Arg.parse specs add_file usage;
  
  (* Read files from list if specified *)
  (match !file_list with
   | Some list_file ->
       let ic = open_in list_file in
       (try
          while true do
            let line = input_line ic in
            let trimmed = String.trim line in
            if trimmed <> "" && trimmed.[0] <> '#' then
              add_file trimmed
          done
        with End_of_file -> ());
       close_in ic
   | None -> ());
  
  (* Reverse files to maintain correct order *)
  input_files := List.rev !input_files;
  
  (* Set verbosity *)
  if !verbose then begin
    Astro_utils.verbose := true;
    Astro_utils.verbose_flag := true;
  end;
  
  (* Convert alignment source string to enum *)
  let alignment_source = match String.lowercase_ascii !alignment_source_str with
    | "plate-solved" -> PlateSolved
    | "live-stack" -> LiveStacking
    | _ -> Auto
  in
  
  (* Convert stacking method string to type *)
  let method_type = match String.lowercase_ascii !stack_method with
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
  if List.length !input_files = 0 then begin
    Printf.printf "Error: No input FITS files provided\n";
    Printf.printf "%s\n" usage;
    exit 1
  end;
  
  (* Print summary *)
  Printf.printf "Astrometric Stacking\n";
  Printf.printf "===================\n";
  Printf.printf "Input files: %d\n" (List.length !input_files);
  Printf.printf "Stacking method: %s\n" !stack_method;
  Printf.printf "Output file: %s\n" !output_file;
  Printf.printf "Alignment source: %s\n" !alignment_source_str;
  Printf.printf "Reference frame index: %d\n\n" !ref_idx;
  flush stdout;
  
  (* Perform stacking with the selected alignment source *)
  let _ = if !unified_wcs then
     stack_with_unified_wcs !input_files !ref_idx method_type !output_file
   else
     stack_images !input_files !ref_idx method_type !output_file alignment_source in ()
