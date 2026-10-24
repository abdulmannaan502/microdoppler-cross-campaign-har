# Reproduction check (Phase 14), 2026-10-24 ~22:15 IST

1. **Clean checkout of the project ZIP** (extracted into an empty directory):
   - `pytest -q tests`: 6 passed
   - `python scripts/analyze.py && python scripts/make_tables.py` regenerated `arxiv/tables/{numbers,protocol,interventions,enrollment}.tex`, **byte-identical** to the submitted versions
2. **LaTeX source ZIP compiled standalone** (pdflatex + bibtex): 5 pages, 0 undefined references, 0 overfull boxes
3. **Fresh-machine run of the full training pipeline**: the remote bundle (code + preprocessed images) ran on a newly created Vast.ai container (different OS, GPU, torch 2.14.1). Unit tests passed there, and all 34 queued conditions completed with rc = 0 (`remote/worker*.out`, `docs/remote_env.txt`)
4. **Cross-platform replication** of source-model scores: within ≈1 point (see `reproducibility_report.md`)
5. **Not re-done here**: raw-data download and re-parsing (2 GB / 16 GB). Verified once, SHA-256 recorded in `scripts/get_data.sh`
6. **Secret scan** of the project ZIP: no private keys, tokens or credentials found (the SSH key used for the GPU host lives only in the user's ~/.ssh and is not part of the project)
