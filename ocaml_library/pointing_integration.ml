(* pointing_integration.ml - Integration with your PointingModel module *)

open Printf

(* Function to load a model from a JSON file *)
let load_model_from_file filename =
  try
    match Util.load_model_from_file filename with
    | Some model -> model
    | None -> PointingModel.create_empty_model ()
  with e ->
    printf "Warning: Error loading pointing model: %s\n" (Printexc.to_string e);
    PointingModel.create_empty_model ()

(* Function to handle model creation with optional file loading *)
let create ?(model_file=None) () =
  match model_file with
  | Some file -> load_model_from_file file
  | None -> PointingModel.create_empty_model ()

(* Function to apply pointing model correction *)
let correct_position model mount_ra mount_dec ?(focus=0) () =
  PointingModel.correct_position model mount_ra mount_dec focus

(* Calculate pixel offset based on pointing model *)
let predict_offset model ra dec plate_scale =
  let corrected_ra, corrected_dec = correct_position model ra dec () in
  
  (* Calculate difference in arcseconds *)
  let ra_diff = (corrected_ra -. ra) *. 3600.0 *. Float.cos (dec *. Float.pi /. 180.0) in
  let dec_diff = (corrected_dec -. dec) *. 3600.0 in
  
  (* Convert to pixel offsets *)
  let x_offset = ra_diff /. plate_scale in
  let y_offset = dec_diff /. plate_scale in
  
  (x_offset, y_offset)
