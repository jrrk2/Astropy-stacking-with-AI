(* FFT-based image alignment for astronomical images *)
open Printf
open Types
open Fits

(* We'll need to bring in a library for FFT - here's a simple implementation for 2D FFT *)
module FFT = struct
  (* Convert image to complex array for FFT *)
  let image_to_complex data width height =
    let complex_data = Array.make_matrix height width (Complex.zero) in
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        complex_data.(y).(x) <- Complex.{ re = float_of_int data.(y).(x); im = 0.0 }
      done
    done;
    complex_data
  
  (* Compute 1D FFT using Cooley-Tukey algorithm *)
  let rec fft_1d a =
    let n = Array.length a in
    if n <= 1 then a
    else begin
      (* Split into even and odd indices *)
      let even = Array.make (n/2) Complex.zero in
      let odd = Array.make (n/2) Complex.zero in
      for i = 0 to n/2 - 1 do
        even.(i) <- a.(2*i);
        odd.(i) <- a.(2*i+1)
      done;
      
      (* Recursively compute FFT of each half *)
      let even_fft = fft_1d even in
      let odd_fft = fft_1d odd in
      
      (* Combine results *)
      let result = Array.make n Complex.zero in
      for k = 0 to n/2 - 1 do
        let t = Complex.mul odd_fft.(k) (Complex.polar 1.0 (-2.0 *. Float.pi *. float_of_int k /. float_of_int n)) in
        result.(k) <- Complex.add even_fft.(k) t;
        result.(k + n/2) <- Complex.sub even_fft.(k) t
      done;
      result
    end
  
  (* Compute 2D FFT by applying 1D FFT to rows and columns *)
  let fft_2d data =
    let height = Array.length data in
    let width = Array.length data.(0) in
    
    (* Apply FFT to rows *)
    let row_fft = Array.make_matrix height width Complex.zero in
    for y = 0 to height - 1 do
      row_fft.(y) <- fft_1d data.(y)
    done;
    
    (* Apply FFT to columns *)
    let result = Array.make_matrix height width Complex.zero in
    for x = 0 to width - 1 do
      let col = Array.make height Complex.zero in
      for y = 0 to height - 1 do
        col.(y) <- row_fft.(y).(x)
      done;
      let col_fft = fft_1d col in
      for y = 0 to height - 1 do
        result.(y).(x) <- col_fft.(y)
      done
    done;
    
    result
  
  (* Compute inverse 2D FFT *)
  let ifft_2d data =
    let height = Array.length data in
    let width = Array.length data.(0) in
    
    (* Conjugate input *)
    let conj_data = Array.make_matrix height width Complex.zero in
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        conj_data.(y).(x) <- Complex.conj data.(y).(x)
      done
    done;
    
    (* Compute FFT of conjugated input *)
    let fft_result = fft_2d conj_data in
    
    (* Conjugate result and scale *)
    let result = Array.make_matrix height width Complex.zero in
    let scale = 1.0 /. float_of_int (width * height) in
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        let conj_val = Complex.conj fft_result.(y).(x) in
        result.(y).(x) <- Complex.{ re = conj_val.re *. scale; im = conj_val.im *. scale }
      done
    done;
    
    result
  
  (* Compute cross-correlation of two images using FFT *)
  let cross_correlation img1 img2 width height =
    (* Convert images to complex arrays *)
    let complex_img1 = image_to_complex img1 width height in
    let complex_img2 = image_to_complex img2 width height in
    
    (* Compute FFT of both images *)
    let fft_img1 = fft_2d complex_img1 in
    let fft_img2 = fft_2d complex_img2 in
    
    (* Compute conjugate of FFT of img1 *)
    let conj_fft_img1 = Array.make_matrix height width Complex.zero in
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        conj_fft_img1.(y).(x) <- Complex.conj fft_img1.(y).(x)
      done
    done;
    
    (* Multiply FFT of img2 by conjugate of FFT of img1 *)
    let cross_fft = Array.make_matrix height width Complex.zero in
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        cross_fft.(y).(x) <- Complex.mul conj_fft_img1.(y).(x) fft_img2.(y).(x)
      done
    done;
    
    (* Compute inverse FFT to get cross-correlation *)
    let cross_corr = ifft_2d cross_fft in
    
    (* Convert back to real array *)
    let result = Array.make_matrix height width 0.0 in
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        result.(y).(x) <- cross_corr.(y).(x).re
      done
    done;
    
    result
end

(* Find peak in cross-correlation to determine shift *)
let find_peak corr width height =
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
let align_with_fft ref_data img_data width height =
  (* Compute cross-correlation *)
  let corr = FFT.cross_correlation ref_data img_data width height in
  
  (* Find peak to determine shift *)
  let (dx, dy) = find_peak corr width height in
  
  printf "  FFT alignment detected shift: dx=%d, dy=%d\n" dx dy;
  
  { dx = float_of_int dx; dy = float_of_int dy; rotation = 0.0; scale = 1.0 }

(* Improved align_image function with sub-pixel accuracy *)
let align_image src_data width height params =
  (* Create output image buffer *)
  let dest_data = Array.make_matrix height width 0 in
  
  (* Extract transformation parameters *)
  let dx = params.dx in
  let dy = params.dy in
  
  (* Fast path for pure translation *)
  (* Fill with zeros initially *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      dest_data.(y).(x) <- 0
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
    let ref_data = read_fits_data ref_contents width height in
    
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
            let data = read_fits_data contents width height in
            
            (* Calculate alignment using FFT *)
            let params = align_with_fft ref_data data width height in
            
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
