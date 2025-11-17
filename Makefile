# Makefile for astrophotography processing pipeline

# Configuration
DARK_DIR = ../dark_temps
PYTHON = python
PIP = pip
# LIGHTS = /Volumes/X10Pro/stellina-5df04c/2024-11-21_20-02-31_observation_M45/02-images-initial
# TARGET = M45
LIGHTS = /Volumes/X10Pro/stellina-5df04c/2024-01-09_22-12-18_observation_M31/02-images-initial
TARGET = M31

# Required Python packages
REQUIRED_PACKAGES = numpy astropy opencv-python-headless scipy colour-demosaicing reproject tqdm matplotlib psutil astroquery ephem

.PHONY: all clean stack finish check_venv install_deps

all: check_venv stack finish

# Check and setup Python virtual environment
check_venv:
		echo "Virtual environment exists, checking packages..."; \
		$(PIP) install -q $(REQUIRED_PACKAGES); \

# Clean outputs and temporary files
clean:
	rm -rf mmap
	rm -rf __pycache__
	rm -f stacked_*.fits
	rm -f *_solved.fits
	rm -f accum_*.fits
	rm -f template.hdr
	rm -f *.png
	rm -f */images.tbl
	rm -f */proj_images.tbl
	rm -f */diffs.tbl
	rm -f */fits.tbl
	rm -f */corrections.tbl
	rm -f */mosaic.tbl
	rm -f */symlinks.stamp
	rm -f lights/*_mmap-indx.xyls
	rm -f lights/*_mmap.axy
	rm -f lights/*_mmap.corr
	rm -f lights/*_mmap.match
	rm -f lights/*_mmap.rdls
	rm -f lights/*_mmap.solved
	rm -f lights/*_mmap.wcs
	rm -f lights/*_mmap.fits
	rm -f stacked_g-indx.xyls
	rm -f *~

# Initial stacking step
stack: check_venv
	$(PYTHON) main_script.py 'lights/temp_285/light_*.fits' --dark-dir $(DARK_DIR)

# Initial stacking step
quick: check_venv
	$(PYTHON) main_script.py 'lights/temp_290/light_*.fits' --dark-dir $(DARK_DIR)

# Final processing step
finish: check_venv
	$(PYTHON) main_module.py --method median

# Check status
status:
	@echo "Dark directory: $(shell [ -d $(DARK_DIR) ] && echo 'OK' || echo 'Missing')"
	@echo "Input files: $$(ls img-0???r.fits 2>/dev/null | wc -l || echo 0)"
	@echo "Stacked outputs: $$(ls stacked_*.fits 2>/dev/null | wc -l || echo 0)"

pairing:
	python astrometry_script.py --debug --log-file logfile.txt ${LIGHTS} --target $(TARGET)

siril:
	python siril_adapter.py ${LIGHTS} --siril-workspace ./siril_workspace

# Update Python dependencies
update_deps: check_venv
	$(PIP) install --upgrade $(REQUIRED_PACKAGES)
