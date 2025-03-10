(* FFT-based alignment using Owl for speed *)

open Owl  (* The main Owl library *)

(* Convert image to Owl ndarray *)
  let image_to_ndarray data width height =
    let arr = Arr.zeros [|height; width|] in
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        Arr.set arr [|y; x|] (float_of_int data.(y).(x))
      done
    done;
    arr
  
  (* Convert Owl ndarray back to OCaml array *)
  let ndarray_to_array arr =
    let shape = Arr.shape arr in
    let height = shape.(0) in
    let width = shape.(1) in
    let result = Array.make_matrix height width 0.0 in
    for y = 0 to height - 1 do
      for x = 0 to width - 1 do
        result.(y).(x) <- Arr.get arr [|y; x|]
      done
    done;
    result
  
  (* Compute cross-correlation of two images using FFT *)
  let cross_correlation img1 img2 width height =
    (* Convert images to Owl arrays *)
    let arr1 = image_to_ndarray img1 width height in
    let arr2 = image_to_ndarray img2 width height in
    
    (* Compute FFT of both images *)
    let fft1 = Arr.fft2 arr1 in
    let fft2 = Arr.fft2 arr2 in
    
    (* Compute conjugate of FFT of img1 *)
    let conj_fft1 = Arr.conj fft1 in
    
    (* Multiply FFT of img2 by conjugate of FFT of img1 *)
    let cross_fft = Arr.mul conj_fft1 fft2 in
    
    (* Compute inverse FFT to get cross-correlation *)
    let cross_corr = Arr.ifft2 cross_fft in
    
    (* Get real part and convert back to OCaml array *)
    let real_part = Arr.re cross_corr in
    ndarray_to_array real_part
