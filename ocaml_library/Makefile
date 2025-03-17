LIGHTDIR=/Volumes/X10Pro/stellina-5df04c/2025-02-17_20-42-14_observation_M42
DARKDIR=/Volumes/X10Pro/stellina-5df04c/dark_temps
LAT=52.244020
LONG=0.078841
TARGET=M42
MAXFILES=10

all: integrate verify stack

integrate: _build/default/stellina_integration.exe
	-@rm -rf lights rgb
	env MAXFILES=$(MAXFILES) OCAMLRUNPARAM=b _build/default/stellina_integration.exe --calibrate --darks-dir $(DARKDIR) --lat $(LAT) --lon $(LONG) --target $(TARGET) $(LIGHTDIR)/02-images-initial
	@mkdir rgb
	mv lights/temp_*/cal_*_rgb.fits rgb

verify: _build/default/plate_solve_cli.exe
	-@rm -rf solved_verification
	env OCAMLRUNPARAM=b _build/default/plate_solve_cli.exe -workers 8 -i rgb

stack: _build/default/astrometric_stack_cli.exe
	env OCAMLRUNPARAM=b _build/default/astrometric_stack_cli.exe -o stacked.fits solved_verification/cal_*_rgb.fits

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
