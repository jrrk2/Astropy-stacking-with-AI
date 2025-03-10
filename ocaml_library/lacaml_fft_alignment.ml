(* FFT-based image alignment using Lacaml library and OCaml-fftw *)
open Printf
open Types
open Lacaml.D  (* Double precision module from Lacaml *)
open Fftw3.D   (* Double precision module from OCaml-fftw *)

(* Lacaml and FFTW-based alignment for performance *)
module FastFFT = struct
  (* Convert image to Lacaml matrix *)
  let image_to_mat data width height =
    (* Use the correct Lacaml matrix creation function *)
    let mat = Mat.create height width in
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        mat.{y+1, x+1} <- float_of_int data.(y).(x)
      done
    done;
    mat
  
  (* Convert Lacaml matrix back to OCaml array *)
  let mat_to_array mat =
    let m, n = Mat.dim1 mat, Mat.dim2 mat in
    let result = Array.make_matrix m n 0.0 in
    for y = 0 to m - 1 do
      for x = 0 to n - 1 do
        result.(y).(x) <- mat.{y+1, x+1}
      done
    done;
    result
  
  (* Helper to create complex array for FFTW *)
  let create_complex_array m n =
    Array2.create complex64 c_layout m n
  
  (* Compute cross-correlation of two images using FFT *)
  let cross_correlation img1 img2 width height =
    (* Convert images to Lacaml matrices *)
    let mat1 = image_to_mat img1 width height in
    let mat2 = image_to_mat img2 width height in
    
    (* Prepare complex arrays for FFTW *)
    let fft1 = create_complex_array height width in
    let fft2 = create_complex_array height width in
    let result = create_complex_array height width in
    
    (* Copy data to complex arrays *)
    for i = 0 to height - 1 do
      for j = 0 to width - 1 do
        fft1.{i, j} <- { re = mat1.{i+1, j+1}; im = 0.0 };
        fft2.{i, j} <- { re = mat2.{i+1, j+1}; im = 0.0 };
      done
    done;
    
    (* Create FFTW plans *)
    let plan_forward1 = dft2 Forward fft1 fft1 in
    let plan_forward2 = dft2 Forward fft2 fft2 in
    let plan_backward = dft2 Backward result result in
    
    (* Execute FFT *)
    execute plan_forward1;
    execute plan_forward2;
    
    (* Compute complex conjugate product *)
    for i = 0 to height - 1 do
      for j = 0 to width - 1 do
        let a = fft1.{i, j} in
        let b = fft2.{i, j} in
        (* conjugate of a * b *)
        result.{i, j} <- { 
          re = a.re *. b.re +. a.im *. b.im;
          im = a.re *. b.im -. a.im *. b.re;
        };
      done
    done;
    
    (* Execute inverse FFT *)
    execute plan_backward;
    
    (* Normalize and extract real part *)
    let norm = float_of_int (width * height) in
    let output_mat = Mat.create height width in
    for i = 0 to height - 1 do
      for j = 0 to width - 1 do
        output_mat.{i+1, j+1} <- result.{i, j}.re /. norm;
      done
    done;
    
    (* Clean up FFTW plans *)
    destroy_plan plan_forward1;
    destroy_plan plan_forward2;
    destroy_plan plan_backward;
    
    (* Convert back to OCaml array *)
    mat_to_array output_mat
  end

(* Find peak in cross-correlation to determine shift *)
let find_peak corr width height =
  let max_val = ref neg_infinity in
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
  (* Compute cross-correlation using Lacaml/FFTW *)
  let corr = FastFFT.cross_correlation ref_data img_data width height in
  
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
  
  (* Fast version using Lacaml for matrix operations *)
  (* First convert to float matrices *)
  let src_mat = Lacaml.D.Mat.create height width in
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      src_mat.{y+1, x+1} <- float_of_int src_data.(y).(x);
    done
  done;
  
  (* Compute sub-pixel translation using bilinear interpolation *)
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let src_x = float_of_int x - dx in
      let src_y = float_of_int y - dy in
      
      let src_x_floor = floor src_x in
      let src_y_floor = floor src_y in
      let src_x_int = int_of_float src_x_floor in
      let src_y_int = int_of_float src_y_floor in
      
      let x_frac = src_x -. src_x_floor in
      let y_frac = src_y -. src_y_floor in
      
      (* Bilinear interpolation with bounds checking *)
      if src_x_int >= 0 && src_x_int + 1 < width && 
         src_y_int >= 0 && src_y_int + 1 < height then begin
        (* Get the four surrounding pixels from Lacaml matrix *)
        let p00 = src_mat.{src_y_int+1, src_x_int+1} in
        let p10 = src_mat.{src_y_int+1, src_x_int+2} in
        let p01 = src_mat.{src_y_int+2, src_x_int+1} in
        let p11 = src_mat.{src_y_int+2, src_x_int+2} in
        
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

(* Update align_images to use Lacaml-based FFT alignment *)
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
