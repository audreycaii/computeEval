#!/bin/sh
# Build the paper with the bundled Tectonic (XeTeX engine); run from anywhere.
cd "$(dirname "$0")" && ../tools/tectonic -X compile --keep-logs main.tex
