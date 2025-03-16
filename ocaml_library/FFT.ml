
(* We'll need to bring in a library for FFT - here's a simple implementation for 2D FFT *)
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
