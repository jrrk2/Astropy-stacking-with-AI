(* hybrid_stack_cli.ml - Command-line interface for hybrid stacking *)
open Hybrid_stacking

let usage = "Usage: hybrid_stack_cli [options] file1.fits file2.fits ...\n\
            \n\
            Options:\n\
            \  -o <file>         Output file name (default: stacked_hybrid.fits)\n\
            \  -method <method>  Stacking method: average, median, sigmaclip, kappa, weighted\n\
            \  -sigma <value>    Sigma value for sigmaclip method (default: 3.0)\n\
            \  -kappa <value>    Kappa value for kappa method (default: 3.0)\n\
            \  -ref <index>      Index of reference image (default: 0)\n\
            \  -no-plate-solving Disable plate solving alignment\n\
            \  -no-live-stacking Disable live stacking coordinates\n\
            \  -no-hybrid        Disable hybrid mode (use either plate solving or live stacking)\n\
            \  -error-threshold <value> Error threshold for hybrid mode (default: 10.0 pixels)\n\
            \  -v                Enable verbose output\n\
            \  -help             Display this list of options\n\
            \n\
            Examples:\n\
            \  hybrid_stack_cli -o result.fits -method median image1.fits image2.fits image3.fits\n\
            \  hybrid_stack_cli -no-plate-solving -method sigmaclip -sigma 2.5 *.fits\n"

(* Program entry point *)
let _ = main ()
