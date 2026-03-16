# Author(s): Amish Kunal
"""Plot sensitivity experiment results: judge F1 across negative mixes.

Reads results/sensitivity_negative_mix.csv (produced by sensitivity_negative_mix.py)
and generates a grouped bar chart + line plot.

Run:
  python3 expirements/plot_sensitivity.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="results/sensitivity_negative_mix.csv")
    ap.add_argument("--out-dir", type=str, default="results")
    args = ap.parse_args()

    in_path = (REPO_ROOT / args.input).resolve()
    if not in_path.exists():
        print(f"Not found: {in_path}")
        print("Run: python3 expirements/sensitivity_negative_mix.py --pretrain-data both")
        sys.exit(1)

    rows = _read_csv(in_path)

    by_pretrain: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        pt = r.get("pretrain_data", "oe")
        by_pretrain.setdefault(pt, []).append(r)

    neg_labels = {
        "oe": "OE only",
        "oe+random": "+random",
        "oe+random+bigram": "+bigram",
        "oe+random+bigram+transformer": "+transformer",
    }
    neg_order = list(neg_labels.keys())

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Line plot: F1 across negative mixes ---
    fig, ax = plt.subplots(figsize=(9, 5))
    pt_colors = {"oe": "#1f77b4", "shakespeare": "#d62728"}
    pt_labels = {"oe": "Base LM: Old English", "shakespeare": "Base LM: Shakespeare"}

    for pt, pt_rows in sorted(by_pretrain.items()):
        neg_to_f1 = {r["negatives"]: float(r["f1"]) for r in pt_rows if r["f1"] != "n/a"}
        xs = list(range(len(neg_order)))
        ys = [neg_to_f1.get(n, 0) for n in neg_order]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=8,
                color=pt_colors.get(pt, "#333"), label=pt_labels.get(pt, pt))
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xticks(range(len(neg_order)))
    ax.set_xticklabels([neg_labels[n] for n in neg_order], fontsize=10)
    ax.set_xlabel("Negative mix (cumulative)", fontsize=11)
    ax.set_ylabel("Test F1", fontsize=11)
    ax.set_title("Judge F1 vs Negative Mix (by Base LM Pretraining)", fontsize=12)
    ax.set_ylim(0.75, 1.02)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    line_path = out_dir / "sensitivity_f1_line.png"
    fig.savefig(line_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {line_path}")

    # --- Grouped bar chart: acc, precision, recall, F1 ---
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for pt, pt_rows in sorted(by_pretrain.items()):
        neg_to_row = {r["negatives"]: r for r in pt_rows}
        fig, ax = plt.subplots(figsize=(10, 5))
        x_pos = list(range(len(neg_order)))
        bar_w = 0.18

        for mi, metric in enumerate(metrics):
            offsets = [x + mi * bar_w for x in x_pos]
            vals = [float(neg_to_row.get(n, {}).get(metric, 0)) for n in neg_order]
            ax.bar(offsets, vals, width=bar_w, label=metric.capitalize(),
                   color=metric_colors[mi], alpha=0.85)

        ax.set_xticks([x + 1.5 * bar_w for x in x_pos])
        ax.set_xticklabels([neg_labels[n] for n in neg_order], fontsize=10)
        ax.set_xlabel("Negative mix (cumulative)", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"Judge Metrics vs Negative Mix (Base LM: {pt_labels.get(pt, pt)})", fontsize=12)
        ax.set_ylim(0.6, 1.05)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        bar_path = out_dir / f"sensitivity_bars_{pt}.png"
        fig.savefig(bar_path, dpi=180)
        plt.close(fig)
        print(f"Saved: {bar_path}")

    # --- Perplexity bar chart (if available) ---
    ppl_path = (REPO_ROOT / "results" / "perplexity_by_source.csv").resolve()
    if ppl_path.exists():
        ppl_rows = _read_csv(ppl_path)
        sources = [r["source"] for r in ppl_rows]
        mean_ppls = [float(r["mean_ppl"]) for r in ppl_rows]
        source_labels = {
            "real_sp": "Shakespeare",
            "real_oe": "Old English",
            "random": "Random",
            "bigram": "Bigram",
            "transformer": "Transformer",
        }
        colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e", "#9467bd"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            range(len(sources)),
            mean_ppls,
            color=[colors[i % len(colors)] for i in range(len(sources))],
            alpha=0.85,
        )
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([source_labels.get(s, s) for s in sources], fontsize=10)
        ax.set_ylabel("Mean Perplexity", fontsize=11)
        ax.set_title("Perplexity by Text Source (Shakespeare-finetuned LM)", fontsize=12)
        ax.set_yscale("log")
        for bar, val in zip(bars, mean_ppls):
            ax.annotate(f"{val:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        ppl_plot = out_dir / "perplexity_by_source.png"
        fig.savefig(ppl_plot, dpi=180)
        plt.close(fig)
        print(f"Saved: {ppl_plot}")
    else:
        print(f"Perplexity CSV not found at {ppl_path}, skipping plot.")


if __name__ == "__main__":
    main()
