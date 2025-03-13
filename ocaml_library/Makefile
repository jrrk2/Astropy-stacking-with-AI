LIGHTDIR=/Volumes/X10Pro/stellina-5df04c/2025-02-17_20-42-14_observation_M42
LAT=52.244020
LONG=0.078841
TARGET=M42
WORKERS=8

all: integrate verify stack

integrate: _build/default/stellina_integration.exe
	-@rm -rf lights rgb
	env OCAMLRUNPARAM=b _build/default/stellina_integration.exe --calibrate --darks-dir ../../dark_temps --temp-tolerance 8.0 --lat $(LAT) --lon $(LONG) --target $(TARGET) $(LIGHTDIR)/02-images-initial
	@mkdir rgb
	mv lights/temp_*/cal_*_rgb.fits rgb

verify: _build/default/plate_solve_cli.exe
	-@rm -rf solved_verification
	env OCAMLRUNPARAM=b _build/default/plate_solve_cli.exe -workers $(WORKERS) -i rgb

stack: _build/default/astrometric_stack_cli.exe
	env OCAMLRUNPARAM=b _build/default/astrometric_stack_cli.exe -o stacked.fits solved_verification/cal_*_rgb.fits

test: _build/default/fits_reader_test.exe
	env OCAMLRUNPARAM=b _build/default/fits_reader_test.exe -f solved_verification/cal_20250217_204959_BGGR_rgb.fits

detect: _build/default/star_detection_cli.exe
	-@rm -rf star_detection_results
	_build/default/star_detection_cli.exe -workers $(WORKERS) -visualize -i solved_verification

compare: _build/default/stack_comparison.exe
	_build/default/stack_comparison.exe -ref 0 -dir rgb

shar:
	(cd _build/default && shar `ls -1 *.ml *.mli | fgrep -v .pp.` > ../../archive.shar.txt)

model: _build/default/altaz_radec_model.exe
	env OCAMLRUNPARAM=b _build/default/altaz_radec_model.exe -lat $(LAT) -long $(LONG) -json $(LIGHTDIR) $(LIGHTDIR)/02-images-initial/img-000?r.fits

_build/default/stellina_integration.exe:
	dune build $@

_build/default/plate_solve_cli.exe:
	dune build $@

_build/default/astrometric_stack_cli.exe:
	dune build $@

_build/default/altaz_radec_model.exe:
	dune build $@

_build/default/star_detection_cli.exe:
	dune build $@

_build/default/fits_reader_test.exe:
	dune build $@

_build/default/stack_comparison.exe:
	dune build $@
