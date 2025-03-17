(* image_stretching.ml - Stretching algorithms for astronomical images *)

open Bigarray
open Types
open Fits
open Fits_utils

(* Stretching types *)
type stretch_method =
  | Linear of float * float  (* min_val, max_val *)
  | LogarithmicStretch of float  (* gamma *)
  | AsinhStretch of float    (* stretch factor *)
  | HistogramEqualization
  | AutomaticStretch         (* automatic stretching *)
  | CustomStretch            (* custom arctangent-based stretching *)

(* Helper function to calculate histogram *)
let calculate_histogram data width height bins =
  let hist = Array.make bins 0 in
  let min_val = ref 65535.0 in
  let max_val = ref 0.0 in
  
  (* First pass to find min/max *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let val16 = float_of_int (get_pixel data y x) in
      min_val := min !min_val val16;
      max_val := max !max_val val16;
    done
  done;
  
  (* Second pass to build histogram *)
  let range = !max_val -. !min_val in
  if range > 0.0 then
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let val16 = float_of_int (get_pixel data y x) in
        let bin = 
          if val16 = !max_val then bins - 1
          else int_of_float ((val16 -. !min_val) /. range *. float_of_int (bins - 1))
        in
        hist.(bin) <- hist.(bin) + 1
      done
    done;
  
  (hist, !min_val, !max_val)

(* Linear stretch function *)
let apply_linear_stretch data width height min_val max_val =
  let result = Array2.create int c_layout height width in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let val16 = float_of_int (get_pixel data y x) in
      let normalized = 
        if max_val > min_val then 
          (val16 -. min_val) /. (max_val -. min_val) 
        else 0.0
      in
      let stretched = int_of_float (Float.min 32767.0 (Float.max 0.0 (normalized *. 32767.0))) in
      Array2.set result y x stretched
    done
  done;
  
  result

(* Logarithmic stretch function *)
let apply_log_stretch data width height gamma =
  let result = Array2.create int c_layout height width in
  
  (* Find max value for normalization *)
  let max_val = ref 0.0 in
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let val16 = float_of_int (get_pixel data y x) in
      max_val := max !max_val val16
    done
  done;
  
  if !max_val > 0.0 then
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let val16 = float_of_int (get_pixel data y x) in
        let normalized = val16 /. !max_val in
        
        (* Apply logarithmic transformation with gamma control *)
        let stretched = 
          if normalized > 0.0 then
            let log_val = log (1.0 +. normalized *. gamma) /. log (1.0 +. gamma) in
            int_of_float (Float.min 32767.0 (log_val *. 32767.0))
          else 0
        in
        
        Array2.set result y x stretched
      done
    done;
  
  result

(* Arcsinh stretch function - good for astronomical images *)
let apply_asinh_stretch data width height stretch_factor =
  let result = Array2.create int c_layout height width in
  
  (* Find max value for normalization *)
  let max_val = ref 0.0 in
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let val16 = float_of_int (get_pixel data y x) in
      max_val := max !max_val val16
    done
  done;
  
  if !max_val > 0.0 then
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let val16 = float_of_int (get_pixel data y x) in
        let normalized = val16 /. !max_val in
        
        (* Apply arcsinh transformation *)
        let stretched = 
          let asinh_val = (Stdlib.Float.asinh (normalized *. stretch_factor)) /. 
                          (Stdlib.Float.asinh stretch_factor) in
          int_of_float (Float.min 32767.0 (asinh_val *. 32767.0))
        in
        
        Array2.set result y x stretched
      done
    done;
  
  result

(* Histogram equalization stretch *)
let apply_histogram_equalization data width height =
  let result = Array2.create int c_layout height width in
  let bins = 65536 in  (* 16-bit precision *)
  
  (* Calculate histogram *)
  let (hist, min_val, max_val) = calculate_histogram data width height bins in
  
  (* Calculate cumulative histogram *)
  let cum_hist = Array.make bins 0 in
  let sum = ref 0 in
  for i = 0 to bins - 1 do
    sum := !sum + hist.(i);
    cum_hist.(i) <- !sum
  done;
  
  (* Total number of pixels *)
  let total_pixels = width * height in
  
  (* Apply histogram equalization *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let val16 = get_pixel data y x in
      let bin = 
        if float_of_int val16 = max_val then bins - 1
        else int_of_float ((float_of_int val16 -. min_val) /. (max_val -. min_val) *. float_of_int (bins - 1))
      in
      
      let equalized = 
        if total_pixels > 0 then
          int_of_float (float_of_int cum_hist.(bin) /. float_of_int total_pixels *. 32767.0)
        else 0
      in
      
      Array2.set result y x equalized
    done
  done;
  
  result

(* Custom stretch similar to the Python implementation *)
let apply_custom_stretch data width height =
  (* Compute image statistics *)
  let stats = compute_image_stats_big data in
  
  (* Find background and noise level *)
  let median_approx = stats.mean -. 0.5 *. stats.stddev in
  let noise_level = max 1.0 (0.3 *. stats.stddev) in
  
  (* Create output array *)
  let result = Array2.create int c_layout height width in
  
  (* Arctangent-based stretch with automatic parameters *)
  let stretch_factor = 3.0 in
  let midtone_factor = 0.5 in
  let max_val = float_of_int stats.max_value in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let val16 = float_of_int (get_pixel data y x) in
      
      (* Background subtraction with clipping at 0 *)
      let background_sub = max 0.0 (val16 -. median_approx) in
      
      (* Apply arctan stretch with normalized parameters *)
      let normalized = background_sub /. (max_val -. median_approx) in
      
      (* Handle potential NaN values *)
      let normalized = if Float.is_nan normalized then 0.0 else normalized in
      
      (* Apply arctangent stretch with midtone control *)
      let stretched = 
        (2.0 /. Float.pi) *. 
        (Float.atan (normalized /. midtone_factor *. stretch_factor)) *.
        32767.0
      in
      
      (* Ensure result is in valid range *)
      let result_val = int_of_float (Float.min 32767.0 (Float.max 0.0 stretched)) in
      Array2.set result y x result_val
    done
  done;
  
  result

(* Automatic stretching based on image statistics *)
let apply_auto_stretch data width height =
  (* Compute image statistics *)
  let stats = compute_image_stats_big data in
  
  (* Determine stretch parameters based on statistics *)
  let black_point = stats.mean -. 1.0 *. stats.stddev in
  let black_point = max 0.0 black_point in
  
  let white_point = stats.mean +. 3.0 *. stats.stddev in
  let white_point = min (float_of_int stats.max_value) white_point in
  
  (* Apply linear stretch with calculated parameters *)
  apply_linear_stretch data width height black_point white_point

(* Function to split RGB FITS image into individual planes *)
let split_rgb_planes fits_file =
  try
    (* Read FITS header and data *)
    let hdrh = just_header fits_file in
    let naxis = parse_int hdrh "NAXIS" in
    
    if naxis <> 3 then
      failwith "Not an RGB image (NAXIS != 3)";
    
    let width = parse_int hdrh "NAXIS1" in
    let height = parse_int hdrh "NAXIS2" in
    let naxis3 = parse_int hdrh "NAXIS3" in
    
    if naxis3 <> 3 then
      failwith "Not an RGB image (NAXIS3 != 3)";
    
    (* Read the image data *)
    let _, contents = find_header_end fits_file (read_image fits_file) in
    
    (* Calculate plane size and offsets *)
    let plane_size = width * height * 2 in  (* 16-bit = 2 bytes per pixel *)
    
    (* Create three separate planes *)
    let r_plane = Array2.create int c_layout height width in
    let g_plane = Array2.create int c_layout height width in
    let b_plane = Array2.create int c_layout height width in
    
    (* Extract the plane data *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        (* Red plane (first plane) *)
        let offset_r = (y * width + x) * 2 in
        if offset_r + 1 < plane_size then
          Array2.set r_plane y x ((int_of_char contents.[offset_r] lsl 8) lor 
                               (int_of_char contents.[offset_r + 1]));
        
        (* Green plane (second plane) *)
        let offset_g = plane_size + (y * width + x) * 2 in
        if offset_g + 1 < 2 * plane_size then
          Array2.set g_plane y x ((int_of_char contents.[offset_g] lsl 8) lor 
                               (int_of_char contents.[offset_g + 1]));
        
        (* Blue plane (third plane) *)
        let offset_b = 2 * plane_size + (y * width + x) * 2 in
        if offset_b + 1 < String.length contents then
          Array2.set b_plane y x ((int_of_char contents.[offset_b] lsl 8) lor 
                               (int_of_char contents.[offset_b + 1]));
      done
    done;
    
    Some (r_plane, g_plane, b_plane, width, height, hdrh)
  with e ->
    Printf.eprintf "Error splitting RGB planes: %s\n" (Printexc.to_string e);
    None

(* Main stretch function that applies the selected method *)
let stretch_image data width height method_type =
  match method_type with
  | Linear (min_val, max_val) -> 
      apply_linear_stretch data width height min_val max_val
  | LogarithmicStretch gamma -> 
      apply_log_stretch data width height gamma
  | AsinhStretch stretch_factor -> 
      apply_asinh_stretch data width height stretch_factor
  | HistogramEqualization -> 
      apply_histogram_equalization data width height
  | AutomaticStretch -> 
      apply_auto_stretch data width height
  | CustomStretch -> 
      apply_custom_stretch data width height

(* Function to apply stretching to a monochrome FITS image *)
let stretch_fits_file input_file output_file method_type =
  try
    (* Read FITS file *)
    let hdrh, data = read_fits_large input_file in
    let width = parse_int hdrh "NAXIS1" in
    let height = parse_int hdrh "NAXIS2" in
    
    Printf.printf "Stretching image %s (%dx%d)...\n" 
      (Filename.basename input_file) width height;
    
    (* Apply stretching *)
    let stretched_data = stretch_image data width height method_type in
    
    (* Create output header (copy input header) *)
    let output_header = Hashtbl.create (Hashtbl.length hdrh) in
    Hashtbl.iter (fun k v -> Hashtbl.add output_header k v) hdrh;
    
    (* Add metadata about stretching *)
    let method_desc = match method_type with
      | Linear (min_val, max_val) -> 
          Printf.sprintf "Linear stretch [%.1f, %.1f]" min_val max_val
      | LogarithmicStretch gamma -> 
          Printf.sprintf "Logarithmic stretch (gamma=%.2f)" gamma
      | AsinhStretch stretch_factor -> 
          Printf.sprintf "Arcsinh stretch (factor=%.2f)" stretch_factor
      | HistogramEqualization -> 
          "Histogram equalization"
      | AutomaticStretch -> 
          "Automatic stretch"
      | CustomStretch ->
          "Custom arctangent-based stretch"
    in
    Hashtbl.add output_header "HISTORY" (Printf.sprintf " Stretched with %s" method_desc);
    
    (* Write output file *)
    let oc = open_out_bin output_file in
    
    (* Write header *)
    ignore (write_fits_header oc output_header);
    
    (* Write data *)
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let value = Array2.get stretched_data y x in
        output_byte oc (value lsr 8);
        output_byte oc (value land 0xFF);
      done
    done;
    
    (* Pad data to multiple of 2880 bytes *)
    let data_size = width * height * 2 in
    let padding_size = (2880 - (data_size mod 2880)) mod 2880 in
    output_string oc (String.make padding_size '\000');
    
    close_out oc;
    Printf.printf "Stretched image saved to %s\n" output_file;
    true
  with e ->
    Printf.eprintf "Error stretching image: %s\n" (Printexc.to_string e);
    false

(* Function to apply stretching to an RGB FITS image *)
let stretch_rgb_fits_file input_file output_file method_type =
  match split_rgb_planes input_file with
  | Some (r_plane, g_plane, b_plane, width, height, hdrh) ->
      (* Apply stretching to each plane *)
      let r_stretched = stretch_image r_plane width height method_type in
      let g_stretched = stretch_image g_plane width height method_type in
      let b_stretched = stretch_image b_plane width height method_type in
      
      (* Convert to RGB data format *)
      let rgb_data = Array.make_matrix height width (0, 0, 0) in
      let max = ref 1 in
      for y = 0 to height - 1 do
        for x = 0 to width - 1 do
          let r = Array2.get r_stretched y x in
          let g = Array2.get g_stretched y x in
          let b = Array2.get b_stretched y x in
          rgb_data.(y).(x) <- (r,g,b);
          if !max < r then max := r;
          if !max < g then max := g;
          if !max < b then max := b;
        done
      done;

      (* Create a buffer for the image data (RGB format) *)
      let data = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout (width * height * 3) in

      (* Fill the buffer with pixel data *)
      for y = 0 to height - 1 do
	for x = 0 to width - 1 do
          let r,g,b = rgb_data.(y).(x) in
	  let index = (y * width + x) * 3 in
	  let store off pixel = Bigarray.Array1.set data (index+off) (pixel * 255 / !max) in
          store 0 r;
          store 1 g;
          store 2 b;
	done
      done;

      (* Save the image to a PNG file *)
      Stb_image_write.png (output_file^".png") ~w:width ~h:height ~c:3 data;

      (* Add metadata about stretching *)
      let method_desc = match method_type with
        | Linear (min_val, max_val) -> 
            Printf.sprintf "Linear stretch [%.1f, %.1f]" min_val max_val
        | LogarithmicStretch gamma -> 
            Printf.sprintf "Logarithmic stretch (gamma=%.2f)" gamma
        | AsinhStretch stretch_factor -> 
            Printf.sprintf "Arcsinh stretch (factor=%.2f)" stretch_factor
        | HistogramEqualization -> 
            "Histogram equalization"
        | AutomaticStretch -> 
            "Automatic stretch"
        | CustomStretch ->
            "Custom arctangent-based stretch"
      in
      Hashtbl.add hdrh "HISTORY" (Printf.sprintf " Stretched with %s" method_desc);
      
      (* Write the RGB FITS file *)
      let result = write_rgb_data_to_fits output_file hdrh rgb_data in
      if result then
        Printf.printf "Stretched RGB image saved to %s\n" output_file;
      result
      
  | None ->
      Printf.eprintf "Failed to split RGB planes from %s\n" input_file;
      false

(* Simple helper for in-memory stretching of a single-channel array *)
let stretch_array (data: (int, 'a, 'b) Array2.t) method_type =
  let height = Array2.dim1 data in
  let width = Array2.dim2 data in
  stretch_image data width height method_type
