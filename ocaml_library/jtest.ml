(* Parse JSON data *)
open Yojson.Basic.Util
open JsonParser

let pth = "/Volumes/X10Pro/stellina-5df04c/2025-02-17_20-42-14_observation_M42/02-images-initial/"
let focus_flt = ref 0.0
let focus_qual = ref 0.0

let parse json_file =
  print_endline json_file;
  let json = Yojson.Basic.from_file json_file in
  let index = json |> member "index" |> to_int in
  let motors = json |> member "motors" in
  let alt = motors |> member "ALT" |> to_float in
  let az = motors |> member "AZ" |> to_float in
  let derot = motors |> member "DER" |> to_float in
  let focus_map = motors |> member "MAP" |> to_int in
  let stacking = json |> member "stackingData" |> member "liveRegistrationResult" in
  let status = stacking |> member "statusMessage" |> to_string in
  let roundness = stacking |> member "roundness" |> safe_float in
  match status with
    | "StackingRoundnessError" -> ()
    | "StackingOk" ->
      let correction_x = stacking |> member "correction" |> member "x" |> safe_float in
      let correction_y = stacking |> member "correction" |> member "y" |> safe_float in
      let correction_rot = stacking |> member "correction" |> member "rot" |> safe_float in
      let coordinates_x = stacking |> member "coordinates" |> member "x" |> safe_float in
      let coordinates_y = stacking |> member "coordinates" |> member "y" |> safe_float in
      let coordinates_rot = stacking |> member "coordinates" |> member "rot" |> safe_float in
      if index = 0 then
	(
	focus_flt := stacking |> member "focus" |> safe_float;
	focus_qual := stacking |> member "focusQuality" |> safe_float;
	);
    | msg -> failwith msg	  
  ()

let _ = Array.iter (fun json_file ->
			let suff = "stacking.json" in
			let slen = String.length suff in
		let len = String.length json_file in
		if len > slen && (String.sub json_file (len-slen) slen) = suff then
parse (pth^json_file);
) (Sys.readdir pth);;

