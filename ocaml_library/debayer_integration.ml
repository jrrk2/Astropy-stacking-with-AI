(* debayer.ml - Bayer pattern debayering module *)

(* Helper function to get Bayer pattern from FITS header *)
let get_bayer_pattern hdrh =
  match Hashtbl.find_opt hdrh "BAYERPAT=" with
  | Some pat -> 
      let pat_str = match String.split_on_char '\'' pat with
        | _::nxt::_ -> String.trim nxt
        | _ -> String.trim pat
      in
      (match String.uppercase_ascii pat_str with
      | "RGGB" -> Some `RGGB
      | "BGGR" -> Some `BGGR
      | "GRBG" -> Some `GRBG
      | "GBRG" -> Some `GBRG
      | _ -> None)
  | None -> None

(* Helper to describe Bayer pattern for logging *)
let describe_bayer_pattern = function
  | `RGGB -> "RGGB"
  | `BGGR -> "BGGR"
  | `GRBG -> "GRBG"
  | `GBRG -> "GBRG"

(* 2x2 binning respecting RGGB Bayer pattern *)
let bin_2x2_bayer_rggb data width height =
  let bin_height = height / 2 in
  let bin_width = width / 2 in
  let binned = Array.make_matrix bin_height bin_width (0, 0, 0) in
  for y = 0 to bin_height - 1 do
    for x = 0 to bin_width - 1 do
      let r = data.(y*2).(x*2) in         (* R pixel *)
      let g1 = data.(y*2).(x*2+1) in      (* G1 pixel *)
      let g2 = data.(y*2+1).(x*2) in      (* G2 pixel *)
      let b = data.(y*2+1).(x*2+1) in     (* B pixel *)
      let g = (g1 + g2) / 2 in            (* Average G *)
      binned.(y).(x) <- (r, g, b)
    done
  done;
  binned

(* 2x2 binning respecting BGGR Bayer pattern *)
let bin_2x2_bayer_bggr data width height =
  let bin_height = height / 2 in
  let bin_width = width / 2 in
  let binned = Array.make_matrix bin_height bin_width (0, 0, 0) in
  for y = 0 to bin_height - 1 do
    for x = 0 to bin_width - 1 do
      let b = data.(y*2).(x*2) in         (* B pixel *)
      let g1 = data.(y*2).(x*2+1) in      (* G1 pixel *)
      let g2 = data.(y*2+1).(x*2) in      (* G2 pixel *)
      let r = data.(y*2+1).(x*2+1) in     (* R pixel *)
      let g = (g1 + g2) / 2 in            (* Average G *)
      binned.(y).(x) <- (r, g, b)
    done
  done;
  binned

(* 2x2 binning respecting GRBG Bayer pattern *)
let bin_2x2_bayer_grbg data width height =
  let bin_height = height / 2 in
  let bin_width = width / 2 in
  let binned = Array.make_matrix bin_height bin_width (0, 0, 0) in
  for y = 0 to bin_height - 1 do
    for x = 0 to bin_width - 1 do
      let g1 = data.(y*2).(x*2) in        (* G1 pixel *)
      let r = data.(y*2).(x*2+1) in       (* R pixel *)
      let b = data.(y*2+1).(x*2) in       (* B pixel *)
      let g2 = data.(y*2+1).(x*2+1) in    (* G2 pixel *)
      let g = (g1 + g2) / 2 in            (* Average G *)
      binned.(y).(x) <- (r, g, b)
    done
  done;
  binned

(* 2x2 binning respecting GBRG Bayer pattern *)
let bin_2x2_bayer_gbrg data width height =
  let bin_height = height / 2 in
  let bin_width = width / 2 in
  let binned = Array.make_matrix bin_height bin_width (0, 0, 0) in
  for y = 0 to bin_height - 1 do
    for x = 0 to bin_width - 1 do
      let g1 = data.(y*2).(x*2) in        (* G1 pixel *)
      let b = data.(y*2).(x*2+1) in       (* B pixel *)
      let r = data.(y*2+1).(x*2) in       (* R pixel *)
      let g2 = data.(y*2+1).(x*2+1) in    (* G2 pixel *)
      let g = (g1 + g2) / 2 in            (* Average G *)
      binned.(y).(x) <- (r, g, b)
    done
  done;
  binned

(* Apply appropriate binning based on detected Bayer pattern *)
let bin_bayer_pattern data width height = function
  | Some `RGGB -> bin_2x2_bayer_rggb data width height
  | Some `BGGR -> bin_2x2_bayer_bggr data width height
  | Some `GRBG -> bin_2x2_bayer_grbg data width height
  | Some `GBRG -> bin_2x2_bayer_gbrg data width height
  | _ -> failwith "Unknown pattern"

(* Simple binning for monochrome images *)
let bin_2x2_mono data width height =
  let bin_height = height / 2 in
  let bin_width = width / 2 in
  let binned = Array.make_matrix bin_height bin_width (0, 0, 0) in
  for y = 0 to bin_height - 1 do
    for x = 0 to bin_width - 1 do
      let avg = (data.(y*2).(x*2) + data.(y*2).(x*2+1) + 
                 data.(y*2+1).(x*2) + data.(y*2+1).(x*2+1)) / 4 in
      binned.(y).(x) <- (avg, avg, avg)
    done
  done;
  binned
