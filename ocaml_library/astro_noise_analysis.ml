(* Astronomical YCbCr noise analysis *)

type stats = {y_stats: float * float; cb_stats: float * float; cr_stats: float * float }

type noise = {
background_noise: stats;
star_fraction: float
}

(* Modified RGB to YCbCr conversion with HDR handling *)
let astro_rgb_to_ycbcr (r, g, b) =
    (* Convert to float and apply log scaling for HDR *)
    let log_scale x =
        let x' = float_of_int x /. 65535.0 in
        if x' <= 0.0 then 0.0
        else log1p(x') /. log1p(1.0) in
    
    let r' = log_scale r in
    let g' = log_scale g in
    let b' = log_scale b in
    
    (* YCbCr conversion with astronomical weightings *)
    let y  =  0.299 *. r' +. 0.587 *. g' +. 0.114 *. b' in
    let cb = -0.169 *. r' -. 0.331 *. g' +. 0.500 *. b' in
    let cr =  0.500 *. r' -. 0.419 *. g' -. 0.081 *. b' in
    
    (y, cb, cr)

(* Background estimation using sigma-clipping *)
let estimate_background channel_data =
    let height = Array.length channel_data in
    let width = Array.length channel_data.(0) in
    let values = ref [] in
    
    (* Collect all values *)
    for y = 0 to height - 1 do
        for x = 0 to width - 1 do
            values := channel_data.(y).(x) :: !values
        done
    done;
    
    (* Sort values *)
    let (sorted: float array) = Array.of_list (List.sort compare !values) in
    let n = Array.length sorted in
    
    (* Initial median and MAD estimation *)
    let median = sorted.(n/2) in
    let mad1 = Array.map (fun x -> abs_float(x -. median)) sorted in
    Array.sort compare mad1;
    let mad = mad1.(n/2) in
    
    (* Sigma clip at 3σ *)
    let sigma = 1.4826 *. mad in
    let clipped = ref [] in
    Array.iter (fun x ->
        if abs_float(x -. median) < 3.0 *. sigma then
            clipped := x :: !clipped
    ) sorted;
    
    (* Calculate stats on clipped data *)
    let n_clipped = List.length !clipped in
    let mean = List.fold_left (+.) 0.0 !clipped /. float_of_int n_clipped in
    let variance = List.fold_left (fun acc x ->
        let diff = x -. mean in
        acc +. (diff *. diff)
    ) 0.0 !clipped /. float_of_int (n_clipped - 1) in
    
    (mean, sqrt variance)

(* Star detection for noise masking *)
let detect_stars y_channel threshold =
    let height = Array.length y_channel in
    let width = Array.length y_channel.(0) in
    let mask = Array.make_matrix height width false in
    
    (* Calculate background stats *)
    let bg_mean, bg_stddev = estimate_background y_channel in
    let detection_threshold = bg_mean +. threshold *. bg_stddev in
    
    (* Mark star pixels *)
    for y = 0 to height - 1 do
        for x = 0 to width - 1 do
            if y_channel.(y).(x) > detection_threshold then
                mask.(y).(x) <- true
        done
    done;
    
    mask

(* Analyze noise excluding stars *)
let analyze_astronomical_noise rgb_data =
    let ycbcr = Array.map (fun row ->
        Array.map astro_rgb_to_ycbcr row
    ) rgb_data in
    
    (* Extract Y channel *)
    let y_channel = Array.map (fun row ->
        Array.map (fun (y,_,_) -> y) row
    ) ycbcr in
    
    (* Detect stars *)
    let star_mask = detect_stars y_channel 5.0 in
    
    (* Collect background pixels *)
    let bg_y = ref [] in
    let bg_cb = ref [] in
    let bg_cr = ref [] in
    
    Array.iteri (fun y row ->
        Array.iteri (fun x (y_val, cb_val, cr_val) ->
            if not star_mask.(y).(x) then begin
                bg_y := y_val :: !bg_y;
                bg_cb := cb_val :: !bg_cb;
                bg_cr := cr_val :: !bg_cr
            end
        ) row
    ) ycbcr;
    
    (* Calculate statistics *)
    let calc_stats values =
        let n = float_of_int (List.length values) in
        let mean = List.fold_left (+.) 0.0 values /. n in
        let variance = List.fold_left (fun acc v ->
            let diff = v -. mean in
            acc +. (diff *. diff)
        ) 0.0 values /. (n -. 1.0) in
        (mean, sqrt variance)
    in
    
    let y_stats = calc_stats !bg_y in
    let cb_stats = calc_stats !bg_cb in
    let cr_stats = calc_stats !bg_cr in
    
    {
        background_noise = {
            y_stats;
            cb_stats;
            cr_stats
        };
        star_fraction = 
            let total = float_of_int (Array.length rgb_data * Array.length rgb_data.(0)) in
            let stars = Array.fold_left (fun acc row ->
                acc + Array.fold_left (fun a b -> if b then a + 1 else a) 0 row
            ) 0 star_mask in
            float_of_int stars /. total
    }
