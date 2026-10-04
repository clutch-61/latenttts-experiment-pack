#!/usr/bin/env python3
"""Collect Coverage / Majority-Voting from BoN logs (MV does not depend on PRM)."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_log(path: Path) -> dict[str, str | None]:
    text = path.read_text(errors="ignore")
    out: dict[str, str | None] = {"acc": None, "cov": None, "vot": None}
    m = re.search(r"^Accuracy:\s*([\d.]+%)", text, re.M)
    if m:
        out["acc"] = m.group(1)
    m = re.search(r"^Coverage:\s*([\d.]+%)", text, re.M)
    if m:
        out["cov"] = m.group(1)
    m = re.search(r"^Voting Accuracy:\s*([\d.]+%)", text, re.M)
    if m:
        out["vot"] = m.group(1)
    return out


def main() -> None:
    rows: dict[tuple, dict] = {}

    # stability: hard / multiarith / gsm_test seeds
    for p in (ROOT / "logs/bon_stability").glob("*_20260928_101417.log"):
        m = re.match(
            r"(hard|multiarith|gsm_test)_(O2|B0|official)_N(\d+)_s(\d+)_",
            p.name,
        )
        if not m:
            continue
        split, _prm, n, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        d = parse_log(p)
        key = (split, n, seed)
        # keep first with cov/vot
        if key not in rows or (d["cov"] and not rows[key].get("cov")):
            rows[key] = {**d, "log": str(p.relative_to(ROOT))}

    # gsm_test seed42 fill
    for p in (ROOT / "logs/bon_mv").glob("gsm_test_N*_s42_*.log"):
        m = re.match(r"gsm_test_N(\d+)_s42_", p.name)
        if not m:
            continue
        n = int(m.group(1))
        d = parse_log(p)
        key = ("gsm_test", n, 42)
        rows[key] = {**d, "log": str(p.relative_to(ROOT))}

    # official N64 from original matrix (had cov/vot)
    p = ROOT / "logs/bon_matrix/official_N64_20260928_093713.log"
    if p.exists():
        d = parse_log(p)
        if d["vot"]:
            rows.setdefault(("gsm_test", 64, 42), {**d, "log": str(p.relative_to(ROOT))})

    out = ROOT / "results/full/mv_coverage_summary.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["split\tn\tseed\tcov\tvot\tacc_rm_ref\tlog"]
    for key in sorted(rows):
        split, n, seed = key
        r = rows[key]
        lines.append(
            f"{split}\t{n}\t{seed}\t{r.get('cov') or ''}\t{r.get('vot') or ''}\t{r.get('acc') or ''}\t{r.get('log') or ''}"
        )
    out.write_text("\n".join(lines) + "\n")
    print(out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
