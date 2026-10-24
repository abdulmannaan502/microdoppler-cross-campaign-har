"""Reader and index for the University of Glasgow "Radar signatures of human activities"
dataset (Enlighten record 848, CC BY 4.0, doi:10.5525/gla.researchdata.848).

Each .dat file is ASCII: fc [Hz], Tsweep [ms], NTS (samples per chirp), bandwidth [Hz],
followed by the complex beat-signal samples of consecutive chirps (MATLAB 'a+bi' text).
File names follow KPxxAyyRz[...].dat: K = activity code, xx = subject id (re-used across
campaigns, so identity is (campaign, subject)), yy = activity, z = repetition.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTIVITIES = {1: "walk", 2: "sit", 3: "stand", 4: "pick", 5: "drink", 6: "fall"}
_NAME_RE = re.compile(r"^(\d)P(\d+)A(\d+)R(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Record:
    path: Path
    campaign: str
    subject: int
    activity: int
    repetition: int

    @property
    def subject_key(self) -> str:
        """Globally unique person identifier (subject ids are re-used across campaigns)."""
        return f"{self.campaign}/P{self.subject:02d}"


def parse_name(name: str):
    m = _NAME_RE.match(name)
    if not m:
        return None
    k, subj, act, rep = (int(g) for g in m.groups())
    if k != act:  # leading digit duplicates the activity code; flag inconsistencies
        return None
    return subj, act, rep


def index_dataset(root: str | Path) -> list[Record]:
    """Index all .dat files below `root`; the campaign is the immediate parent folder name."""
    root = Path(root)
    recs = []
    for p in sorted(root.rglob("*.dat")):
        parsed = parse_name(p.name)
        if parsed is None:
            continue
        subj, act, rep = parsed
        recs.append(Record(p, p.parent.name, subj, act, rep))
    return recs


def _to_complex(tok: str) -> complex:
    return complex(tok.replace("i", "j"))


def read_dat(path: str | Path):
    """Return (header dict, complex64 matrix [n_chirps, NTS])."""
    txt = Path(path).read_text(encoding="ascii", errors="ignore").split()
    head = [float(np.real(_to_complex(t))) for t in txt[:4]]
    fc, tsweep_ms, nts, bw = head
    nts = int(round(nts))
    data = np.fromiter((_to_complex(t) for t in txt[4:]), dtype=np.complex64, count=len(txt) - 4)
    n_chirps = data.size // nts
    data = data[: n_chirps * nts].reshape(n_chirps, nts)
    return {"fc": fc, "tsweep": tsweep_ms * 1e-3, "nts": nts, "bw": bw}, data
