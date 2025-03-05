open Types
open Simbad
open Printf

(* Thread-safe reference for results *)
let result_ref = ref None
let result_mutex = Mutex.create()

(* Callback function to process SIMBAD results *)
let simbad_callback result =
  Mutex.lock result_mutex;
  result_ref := Some result;
  Mutex.unlock result_mutex

(* Simple logger function *)
let log_info msg =
  if Types.verbose then
    printf "[INFO] %s\n%!" msg

(* Query SIMBAD for an object *)
let query_simbad target =
  result_ref := None;
  Lwt_main.run (simbad' log_info simbad_callback target);
  
  (* Get the result *)
  Mutex.lock result_mutex;
  let result = !result_ref in
  Mutex.unlock result_mutex;
  
  match result with
  | Some (Found (identifier, ra_deg, dec_deg, mag_v)) ->
      let mag_opt = if Float.is_nan mag_v then None else Some mag_v in
      Some { identifier; ra_deg; dec_deg; mag_v = mag_opt }
  | Some (Error msg) ->
      printf "SIMBAD query error: %s\n" msg;
      None
  | Some (Unmatched _) ->
      printf "Unmatched SIMBAD result format\n";
      None
  | None ->
      printf "No response from SIMBAD\n";
      None

(* Enhanced get_object_coordinates with SIMBAD lookup *)
let get_object_coordinates name =
  match query_simbad name with
  | Some result ->
      printf "Found %s: RA=%.4f°, Dec=%.4f°%s\n" 
        result.identifier 
        result.ra_deg 
        result.dec_deg
        (match result.mag_v with 
         | Some mag -> sprintf ", V=%.1f" mag
         | None -> "");
      Some (result.ra_deg, result.dec_deg)
  | None -> 
      printf "Could not find coordinates for %s\n" name;
      None
