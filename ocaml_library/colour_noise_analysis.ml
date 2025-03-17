(* YCbCr conversion and noise analysis *)
open Types

(* RGB to YCbCr conversion *)
let rgb_to_ycbcr (r, g, b) =
    (* Convert RGB values to float in 0-1 range *)
    let r' = float_of_int r /. 65535.0 in
    let g' = float_of_int g /. 65535.0 in
    let b' = float_of_int b /. 65535.0 in
    
    (* Standard RGB -> YCbCr transformation *)
    let y  =  0.299 *. r' +. 0.587 *. g' +. 0.114 *. b' in
    let cb = -0.169 *. r' -. 0.331 *. g' +. 0.500 *. b' in
    let cr =  0.500 *. r' -. 0.419 *. g' -. 0.081 *. b' in
    
    (y, cb, cr)

(* Convert entire RGB array to YCbCr *)
let convert_array_to_ycbcr rgb_data =
    let height = Array.length rgb_data in
    let width = Array.length rgb_data.(0) in
    let ycbcr = Array.make_matrix height width (0., 0., 0.) in
    
    for y = 0 to height - 1 do
        for x = 0 to width - 1 do
            ycbcr.(y).(x) <- rgb_to_ycbcr rgb_data.(y).(x)
        done
    done;
    ycbcr

(* Calculate mean and variance for a channel *)
let calculate_stats channel_data =
    let height = Array.length channel_data in
    let width = Array.length channel_data.(0) in
    let n = float_of_int (height * width) in
    
    (* Calculate mean *)
    let sum = ref 0.0 in
    for y = 0 to height - 1 do
        for x = 0 to width - 1 do
            sum := !sum +. channel_data.(y).(x)
        done
    done;
    let mean = !sum /. n in
    
    (* Calculate variance *)
    let sum_sq_diff = ref 0.0 in
    for y = 0 to height - 1 do
        for x = 0 to width - 1 do
            let diff = channel_data.(y).(x) -. mean in
            sum_sq_diff := !sum_sq_diff +. (diff *. diff)
        done
    done;
    let variance = !sum_sq_diff /. (n -. 1.0) in
    
    (mean, sqrt variance) (* Return mean and standard deviation *)

(* Extract single channel from YCbCr data *)
let extract_channel ycbcr_data channel =
    let height = Array.length ycbcr_data in
    let width = Array.length ycbcr_data.(0) in
    let result = Array.make_matrix height width 0.0 in
    
    for y = 0 to height - 1 do
        for x = 0 to width - 1 do
            let (y', cb, cr) = ycbcr_data.(y).(x) in
            result.(y).(x) <- match channel with
                | `Y  -> y'
                | `Cb -> cb
                | `Cr -> cr
        done
    done;
    result

(* Main noise analysis function *)
let analyze_noise rgb_data =
    let ycbcr = convert_array_to_ycbcr rgb_data in
    
    (* Extract and analyze each channel *)
    let y_channel = extract_channel ycbcr `Y in
    let cb_channel = extract_channel ycbcr `Cb in
    let cr_channel = extract_channel ycbcr `Cr in
    
    let y_mean, y_stddev = calculate_stats y_channel in
    let cb_mean, cb_stddev = calculate_stats cb_channel in
    let cr_mean, cr_stddev = calculate_stats cr_channel in
    
    (* Return analysis results *)
    {
        y_stats = (y_mean, y_stddev);
        cb_stats = (cb_mean, cb_stddev);
        cr_stats = (cr_mean, cr_stddev);
        snr_y = if y_stddev > 0.0 then y_mean /. y_stddev else 0.0;
        snr_cb = if cb_stddev > 0.0 then cb_mean /. cb_stddev else 0.0;
        snr_cr = if cr_stddev > 0.0 then cr_mean /. cr_stddev else 0.0;
    }

(* Calculate local noise variance in Y channel *)
let analyze_local_noise ycbcr_data window_size =
    let height = Array.length ycbcr_data in
    let width = Array.length ycbcr_data.(0) in
    let result = Array.make_matrix height width 0.0 in
    
    for y = window_size to height - window_size - 1 do
        for x = window_size to width - window_size - 1 do
            (* Calculate local statistics in window *)
            let values = ref [] in
            for wy = y - window_size to y + window_size do
                for wx = x - window_size to x + window_size do
                    let (y', _, _) = ycbcr_data.(wy).(wx) in
                    values := y' :: !values
                done
            done;
            
            (* Calculate variance in window *)
            let n = float_of_int ((2 * window_size + 1) * (2 * window_size + 1)) in
            let mean = List.fold_left (+.) 0.0 !values /. n in
            let variance = List.fold_left (fun acc v ->
                let diff = v -. mean in
                acc +. (diff *. diff)
            ) 0.0 !values /. (n -. 1.0) in
            
            result.(y).(x) <- sqrt variance
        done
    done;
    result
