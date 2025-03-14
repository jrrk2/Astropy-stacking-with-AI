# FITS Analysis Tools

This package provides a collection of tools for analyzing and comparing FITS files with a special focus on CD matrix and coordinate transformation parameters (COORDROT, COORDX, COORDY, CORROT, CORX, CORY).

## Features

- Compare FITS files for differences in WCS parameters
- Analyze the relationship between COORDROT and CD matrix rotation angle
- Visualize CD matrix elements and their correlations
- Generate detailed reports on transformation parameters
- Create plots showing relationships between various parameters

## Installation

### Prerequisites

- OCaml (>= 4.08.0)
- Dune build system
- PLplot for visualization
- OPAM packages: cohttp, lwt, yojson, str, ppx_deriving_yojson

### Building from Source

```bash
# Clone the repository
git clone https://github