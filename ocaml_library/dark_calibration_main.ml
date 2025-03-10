(* Entry point for command-line usage *)
let main () =
  let dark_dir = ref "" in
  let input_dir = ref "" in  (* New parameter *)
  let output_dir = ref "calibrated" in
  let temp_tolerance = ref 2.0 in
  let force_rebuild = ref false in
  let apply_only = ref false in
  let master_dark_dir = ref "master_darks" in
  let verbose = ref false in
  
  let args = [
    ("-dark", Arg.Set_string dark_dir, "Directory containing dark frames");
    ("-input", Arg.Set_string input_dir, "Input directory containing light frames");  (* New *)
    ("-out", Arg.Set_string output_dir, "Output directory for calibrated images");
    ("-temp-tol", Arg.Set_float temp_tolerance, "Temperature tolerance in °C");
    ("-force", Arg.Set force_rebuild, "Force rebuild of master darks");
    ("-apply", Arg.Set apply_only, "Apply calibration only (don't create masters)");
    ("-master-dir", Arg.Set_string master_dark_dir, "Directory for master dark frames");
    ("-v", Arg.Set verbose, "Verbose output");
  ] in
  
  let usage = "Usage: dark_calibration -dark <dir> -input <dir> [options]" in
  
  Arg.parse args (fun _ -> ()) usage;
  
  if !dark_dir = "" then begin
    printf "Error: Dark frame directory must be specified\n";
    Arg.usage args usage;
    exit 1
  end;
  
  if !input_dir = "" then begin
    printf "Error: Input directory with light frames must be specified\n";
    Arg.usage args usage;
    exit 1
  end;
  
  let options = {
    dark_dir = !dark_dir;
    input_dir = !input_dir;    (* New field *)
    output_dir = !output_dir;
    temp_tolerance = !temp_tolerance;
    force_rebuild = !force_rebuild;
    apply_only = !apply_only;
    master_dark_dir = !master_dark_dir;
    verbose = !verbose;
  } in
  
  ignore (process_with_calibration options)
