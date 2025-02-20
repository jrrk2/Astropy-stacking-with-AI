# Makefile for astrophotography processing pipeline

# Configuration
DARK_DIR = /Users/jonathan/stellina-5df04c/expert-mode/dark_temps
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

# Required Python packages
REQUIRED_PACKAGES = numpy astropy opencv-python-headless scipy colour-demosaicing reproject tqdm matplotlib psutil astroquery

.PHONY: all clean stack finish check_venv install_deps

all: check_venv stack finish

# Check and setup Python virtual environment
check_venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV_DIR); \
		$(PIP) install --upgrade pip; \
		$(PIP) install $(REQUIRED_PACKAGES); \
	else \
		echo "Virtual environment exists, checking packages..."; \
		$(PIP) install -q $(REQUIRED_PACKAGES); \
	fi

# Clean outputs and temporary files
clean:
	rm -rf venv
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
	rm -f *_mmap-indx.xyls
	rm -f *_mmap.axy
	rm -f *_mmap.corr
	rm -f *_mmap.match
	rm -f *_mmap.rdls
	rm -f *_mmap.solved
	rm -f *_mmap.wcs
	rm -f stacked_g-indx.xyls
	rm -f *~

# Initial stacking step
stack: check_venv
	$(PYTHON) main_script.py 'img-0???r.fits' --dark-dir $(DARK_DIR)

# Final processing step
finish: check_venv
	$(PYTHON) main_module.py --method median

# Check status
status:
	@echo "Python environment: $(shell [ -d $(VENV_DIR) ] && echo 'OK' || echo 'Missing')"
	@echo "Dark directory: $(shell [ -d $(DARK_DIR) ] && echo 'OK' || echo 'Missing')"
	@echo "Input files: $$(ls img-0???r.fits 2>/dev/null | wc -l || echo 0)"
	@echo "Stacked outputs: $$(ls stacked_*.fits 2>/dev/null | wc -l || echo 0)"

# Update Python dependencies
update_deps: check_venv
	$(PIP) install --upgrade $(REQUIRED_PACKAGES)
