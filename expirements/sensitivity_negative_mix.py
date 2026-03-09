"""Sensitivity experiment: vary the negative mix for transformer judge training.

Runs train_transformer_judge.py with each --negatives variant,
parses the final test metrics, and saves a comparison table.

Run:
  python3 expirements/sensitivity_negative_mix.py
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

VARIANTS = [
    "oe",
    "oe+random",
    "oe+random+bigram",
    "oe+random+bigram+transformer",
]

METRIC_RE = re.compile(
    r"Final test metrics \|"
    r"\s*loss=(?P<loss>[\d.]+)"
    r"\s+acc=(?P<accuracy>[\d.]+)"
    r"\s+precision=(?P<precision>[\d.]+)"
    r"\s+recall=(?P<recall>[\d.]+)"
    r"\s+f1=(?P<f1>[\d.]+)"
)

COMP_RE = re.compile(
    r"Dataset composition after balancing:\s*(.+)"
)


def _run_variant(negatives: str, seed: int = 123) -> dict[str, str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "expirements" / "train_transformer_judge.py"),
        "--negatives", negatives,
        "--seed", str(seed),
        "--out", f"models/sensitivity_{negatives.replace('+', '_')}.pt",
    ]
    print(f"\n{'='*60}")
    print(f"Running: --negatives {negatives}")
    print(f"{'='*60}")

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    output = proc.stdout + proc.stderr
    print(output[-600:] if len(output) > 600 else output)

    row: dict[str, str] = {"negatives": negatives}

    m = METRIC_RE.search(output)
    if m:
        row.update(m.groupdict())
    else:
        row.update({"loss": "n/a", "accuracy": "n/a", "precision": "n/a", "recall": "n/a", "f1": "n/a"})

    c = COMP_RE.search(output)
    row["composition"] = c.group(1).strip() if c else "n/a"

    return row


def main() -> None:
    rows: list[dict[str, str]] = []
    for variant in VARIANTS:
        rows.append(_run_variant(variant))

    out_path = (REPO_ROOT / "results" / "sensitivity_negative_mix.csv").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["negatives", "composition", "accuracy", "precision", "recall", "f1", "loss"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'negatives':<35} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}")
    print("-" * 65)
    for r in rows:
        print(f"{r['negatives']:<35} {r.get('accuracy','n/a'):>6} {r.get('precision','n/a'):>6} {r.get('recall','n/a'):>6} {r.get('f1','n/a'):>6}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
