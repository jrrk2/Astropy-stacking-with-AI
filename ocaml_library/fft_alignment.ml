(* FFT-based image alignment for astronomical images *)
open Printf
open Types
open Fits
open Fits_utils
open Bigarray

(* Find peak in cross-correlation to determine shift *)
let find_peak corr =
  let height = Array.length corr in
  let width = Array.length corr.(0) in
  let max_val = ref 0.0 in
  let max_x = ref 0 in
  let max_y = ref 0 in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      if corr.(y).(x) > !max_val then begin
        max_val := corr.(y).(x);
        max_x := x;
        max_y := y
      end
    done
  done;
  
  (* Convert to shift relative to center *)
  let shift_x = 
    if !max_x > width / 2 then !max_x - width
    else !max_x
  in
  let shift_y = 
    if !max_y > height / 2 then !max_y - height
    else !max_y
  in
  
  (shift_x, shift_y)

(* Apply FFT-based alignment to determine shift between two images *)
let align_with_fft ref_data img_data =
  (* Compute cross-correlation *)
  let corr = FFT_big.cross_correlation ref_data img_data in
  
  (* Find peak to determine shift *)
  let (dx, dy) = find_peak corr in
  
  printf "  FFT alignment detected shift: dx=%d, dy=%d\n" dx dy;
  
  { dx = float_of_int dx; dy = float_of_int dy; rotation = 0.0; scale = 1.0 }

(* Apply transformation to align image *)
let align_image (src_data: (int, 'a, 'b) Array2.t) params =
  (* Get dimensions *)
  let height = Array2.dim1 src_data in
  let width = Array2.dim2 src_data in
  
  (* Create output image buffer as Bigarray *)
  let dest_data = Array2.create int c_layout height width in
  
  (* Extract transformation parameters *)
  let dx = params.dx in
  let dy = params.dy in
  
  (* Fill with zeros initially *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      Array2.set dest_data y x 0
    done
  done;
  
  (* Compute sub-pixel translation using bilinear interpolation *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let src_x = float_of_int x -. dx in
      let src_y = float_of_int y -. dy in
      
      let src_x_floor = floor src_x in
      let src_y_floor = floor src_y in
      let src_x_int = int_of_float src_x_floor in
      let src_y_int = int_of_float src_y_floor in
      
      let x_frac = src_x -. src_x_floor in
      let y_frac = src_y -. src_y_floor in
      
      if src_x_int >= 0 && src_x_int + 1 < width && 
         src_y_int >= 0 && src_y_int + 1 < height then begin
        (* Get the four surrounding pixels *)
        let p00 = float_of_int (Array2.get src_data src_y_int src_x_int) in
        let p10 = float_of_int (Array2.get src_data src_y_int (src_x_int + 1)) in
        let p01 = float_of_int (Array2.get src_data (src_y_int + 1) src_x_int) in
        let p11 = float_of_int (Array2.get src_data (src_y_int + 1) (src_x_int + 1)) in
        
        (* Interpolate *)
        let value = 
          p00 *. (1.0 -. x_frac) *. (1.0 -. y_frac) +.
          p10 *. x_frac *. (1.0 -. y_frac) +.
          p01 *. (1.0 -. x_frac) *. y_frac +.
          p11 *. x_frac *. y_frac
        in
        
        Array2.set dest_data y x (int_of_float (Float.round value))
      end
    done
  done;
  dest_data

(* Update stack_images and related functions to use FFT alignment *)
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
    let ref_hdr, ref_data = read_fits_large ref_contents in
    
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
            (* Read data *)
            let hdrh, data = read_fits_large contents in
            
            (* Calculate alignment using FFT *)
            let params = align_with_fft ref_data data in
            
            printf "  Alignment parameters: dx=%.2f, dy=%.2f\n"
              params.dx params.dy;
            alignment_params.(i) <- Some params
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
