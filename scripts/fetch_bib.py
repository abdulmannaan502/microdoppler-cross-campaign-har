"""Verify DOIs against Crossref (or DataCite via doi.org) and fetch BibTeX by content negotiation.
Writes literature/references.bib and literature/doi_verification.csv. Entries that fail are
reported and NOT written."""
import csv, json, re, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = {  # bibkey: DOI
    "shah2022portability": "10.1007/s00521-022-06886-2",
    "yang2023challenge": "10.1109/JBHI.2023.3240895",
    "gla848": "10.5525/gla.researchdata.848",
    "li2020uda": "10.1109/RadarConf2043947.2020.9266433",
    "li2025crossenv": "10.1109/BIBM66473.2025.11356872",
    "liu2024sfda": "10.1109/ICASSP48485.2024.10446797",
    "si2025sfda": "10.1016/j.patcog.2025.111866",
    "zhou2024corruption": "10.1109/TRS.2024.3398127",
    "ayaz2025sp": "10.3390/s25030724",
    "ayaz2026bench": "10.1016/j.pmcj.2026.102198",
    "adl2023whitening": "10.3390/s23177486",
    "shrestha2020bilstm": "10.1109/JSEN.2020.3006386",
    "li2019survey": "10.3390/rs11091068",
    "gurbuz2019spm": "10.1109/MSP.2018.2890128",
    "kim2016dcnn": "10.1109/LGRS.2015.2491329",
    "chen2006microdoppler": "10.1109/TAES.2006.1603402",
    "li2018adabn": "10.1016/j.patcog.2018.03.005",
    "he2016resnet": "10.1109/CVPR.2016.90",
    "sandler2018mobilenetv2": "10.1109/CVPR.2018.00474",
    "rohling1983cfar": "10.1109/TAES.1983.309350",
    "fioranelli2019health": "10.1049/el.2019.2378",
    "seyfioglu2018cae": "10.1109/TAES.2018.2799758",
    "tanaka2025crossenv": "10.48550/arXiv.2512.13018",
    "nguyen2024noise": "10.1371/journal.pone.0308045",
    "shah2019review": "10.1109/MAES.2019.2933971",
    "kim2009svm": "10.1109/TGRS.2009.2012849",
    "ioffe2015bn": "10.48550/arXiv.1502.03167",
    "guo2017calibration": "10.48550/arXiv.1706.04599",
    "wang2021tent": "10.48550/arXiv.2006.10726",
    "ganin2016dann": "10.48550/arXiv.1505.07818",
}


def fetch(url, accept):
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "radarconf-bib/0.1 (mailto:research@example.org)"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8")


if __name__ == "__main__":
    keys = KEYS if len(sys.argv) == 1 else {k: KEYS[k] for k in sys.argv[1:]}
    out_bib, rows = [], []
    for key, doi in keys.items():
        try:
            meta = json.loads(fetch(f"https://doi.org/{doi}", "application/vnd.citationstyles.csl+json"))
            bib = fetch(f"https://doi.org/{doi}", "application/x-bibtex; charset=utf-8").strip()
            bib = re.sub(r"^@(\w+)\{[^,]*,", lambda m: "@" + m.group(1) + "{" + key + ",", bib, count=1)
            out_bib.append(bib)
            title = meta.get("title") if isinstance(meta.get("title"), str) else (meta.get("title") or [""])[0]
            year = (meta.get("issued", {}).get("date-parts") or [[None]])[0][0]
            cont = meta.get("container-title") if isinstance(meta.get("container-title"), str) else ((meta.get("container-title") or [""]) or [""])[0]
            rows.append(dict(key=key, doi=doi, ok=True, year=year, title=title, venue=cont, type=meta.get("type")))
            print("OK ", key, year, title[:80])
        except Exception as e:
            rows.append(dict(key=key, doi=doi, ok=False, year=None, title=repr(e), venue="", type=""))
            print("ERR", key, doi, e)
        time.sleep(0.5)
    (ROOT / "literature" / "references.bib").write_text("\n\n".join(out_bib) + "\n", encoding="utf-8")
    with open(ROOT / "literature" / "doi_verification.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
