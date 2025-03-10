open Dark_temp_analysis

(* Main entry point *)
let () =
  let flags, files = parse_args () in
  if Array.length files = 0 then
    failwith "No input files specified"
  else if flags.sort_darks then
    sort_dark_frames files flags.base_dir flags.dry_run
  else if flags.show_temp_plot && not (flags.show_dist_plot || flags.show_stats || flags.show_coeff) then
    (* Fast path: only temperature scan needed *)
    let temp_infos = Array.to_list files 
      |> List.filter_map scan_fits_temperature 
      |> Array.of_list in
    if Array.length temp_infos > 0 then
      let stats = Array.map (fun (ti:temp_info) -> 
        {filename = ti.filename; 
         temperature = ti.temp;
         mean_level = 0.0;  (* Not needed for temp plot *)
         std_dev = 0.0;     (* Not needed for temp plot *)
         timestamp = 0.0;
         hdrh = ti.qhdr})  (* Will be filled by get_timestamp *)
        temp_infos in
      plot_temp_vs_time stats
    else
      failwith "No valid temperature data found in files"
  else
    (* Full analysis needed *)
    analyze_dark_frames files flags
