open Types
open Printf
open Plplot
open Yojson.Basic.Util

(* JSON Parsing module *)

  let safe_float = function
    | `Int i -> float_of_int i
    | `Float f -> f
    | `String s -> float_of_string s
    | _ -> raise (Yojson.Basic.Util.Type_error ("Expected number", `Null))

  let parse_astrometry_json filename =
    try
      let json = Yojson.Basic.from_file filename in
      
      (* Extract the result section *)
      let result = json |> member "result" in
      
      (* Extract astrometry results *)
      let ra = result |> member "ra" |> safe_float in
      let dec = result |> member "de" |> safe_float in
      let rot = result |> member "rot" |> safe_float in
      
      (* Extract timestamp *)
      let timestamp = result |> member "boardTime" |> safe_float in
      
      (* Extract motor positions from acquisition result *)
      let acq_result = result |> member "acqResult" in
      let motors = acq_result |> member "motors" in
      let az = motors |> member "AZ" |> safe_float in
      let alt = motors |> member "ALT" |> safe_float in
      let der = motors |> member "DER" |> safe_float in
      let map = motors |> member "MAP" |> to_int in
      
      (* We'll set solved_ra and solved_dec to 0 for now - will be filled later *)
      Some { ra; dec; solved_ra = 0.0; solved_dec = 0.0; rot; az; alt; der; map; timestamp; src_file = filename }
    with
    | e -> 
        Printf.printf "Error parsing JSON file %s: %s\n" filename (Printexc.to_string e);
        None

  let parse_pointing_json filename =
    try
      let json = Yojson.Basic.from_file filename in
      let src_file = filename in
      
      (* Extract the params section for target coordinates *)
      let params = json |> member "params" in
      let target_ra = params |> member "ra" |> safe_float in
      let target_dec = params |> member "de" |> safe_float in
      
      (* Extract data section for solved coordinates *)
      let data = json |> member "data" in
      
      (* Extract postGuidingAstrometry if available *)
      let astro_data = 
        try
          let post_guiding = data |> member "postGuidingAstrometry" in
          let ra = post_guiding |> member "astrometryRa" |> safe_float in
          let dec = post_guiding |> member "astrometryDe" |> safe_float in
          let rot = post_guiding |> member "rot" |> safe_float in
          let motors = try post_guiding |> member "motor" with _ -> `Null in
          let az = try motors |> member "AZ" |> safe_float with _ -> 0.0 in
          let alt = try motors |> member "ALT" |> safe_float with _ -> 0.0 in
          let der = try motors |> member "DER" |> safe_float with _ -> 0.0 in
          let map = try motors |> member "MAP" |> to_int with _ -> 0 in
          let timestamp = post_guiding |> member "boardTime" |> safe_float in
          Some { ra; dec; solved_ra = target_ra; solved_dec = target_dec; rot; az; alt; der; map; timestamp; src_file }
        with _ -> None
      in
      
      (* If no postGuidingAstrometry, try postPointing *)
      match astro_data with
      | Some data -> Some data
      | None -> 
          try
            let post_pointing = data |> member "postPointing" in
            let astro = post_pointing |> member "astrometry" in
            let ra = astro |> member "astrometryRa" |> safe_float in
            let dec = astro |> member "astrometryDe" |> safe_float in
            let rot = astro |> member "rot" |> safe_float in
            let az = astro |> member "motorAz" |> safe_float in
            let alt = astro |> member "motorAlt" |> safe_float in
            let der = astro |> member "motorDer" |> safe_float in
            let map = astro |> member "motorMap" |> to_int in
            let timestamp = astro |> member "boardTime" |> safe_float in
            Some { ra; dec; solved_ra = target_ra; solved_dec = target_dec; rot; az; alt; der; map; timestamp; src_file }
          with _ -> None
    with
    | e -> 
        Printf.printf "Error parsing pointing JSON file %s: %s\n" filename (Printexc.to_string e);
        None
