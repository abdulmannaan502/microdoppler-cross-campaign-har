#!/usr/bin/env bash
# Download and verify the Glasgow "Radar signatures of human activities" dataset (CC BY 4.0).
# Record: https://researchdata.gla.ac.uk/848/  DOI: 10.5525/gla.researchdata.848
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/raw/gla848
mkdir -p "$D"
curl -L -C - --retry 5 -o "$D/Dataset_848.7z" https://researchdata.gla.ac.uk/848/20/Dataset_848.7z
curl -L -o "$D/Readme_848.pdf" https://researchdata.gla.ac.uk/848/21/Readme_848.pdf
echo "652f9adc2c9921f9fa08ba2de2d40e3f357d89a91f1b13ac790bc68f4f54ae61  $D/Dataset_848.7z" | sha256sum -c -
PY=${PY:-.venv/Scripts/python}; [[ -x "$PY" ]] || PY=.venv/bin/python
"$PY" -c "import py7zr; py7zr.SevenZipFile('$D/Dataset_848.7z').extractall('$D/extracted')"
echo "extracted to $D/extracted"
