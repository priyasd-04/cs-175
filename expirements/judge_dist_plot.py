"""Plot score distributions for both judges across all text categories.

Drop this in your experiments/ folder and run:
  python3 experiments/plot_judge_scores.py

Saves: results/judge_score_distributions.png
"""

from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from classifier.judge import Noise_Shakespeare_Classifier
from classifier.transformer_judge import TransformerJudge
from monkeys.TransformerMonkey import TransformerMonkey, CharTokenizer
from monkeys.bigram import BigramModel
from monkeys.generator import random_monkey
from utils.load_datasets import load_old_english_dataset, load_shakespeare_dataset

# ── config ────────────────────────────────────────────────────────────────────
JUDGE_CKPT   = "models/transformer_judge.pt"
NOISE_FILE   = "noise_dataset/noise.txt"
N_SAMPLES    = 60       # samples per category
SAMPLE_LEN   = 300
OUT_DIR      = REPO_ROOT / "results"
# ──────────────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── load TF-IDF judge ─────────────────────────────────────────────────────────
print("Loading TF-IDF judge...")
tfidf = Noise_Shakespeare_Classifier()
noise_path = (REPO_ROOT / NOISE_FILE).resolve()
tfidf.train(str(noise_path), show_report=False)

# ── load transformer judge ────────────────────────────────────────────────────
print("Loading transformer judge...")
ckpt_path = (REPO_ROOT / JUDGE_CKPT).resolve()
try:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
except TypeError:
    ckpt = torch.load(ckpt_path, map_location=device)

tokenizer: CharTokenizer = ckpt["tokenizer"]
cfg = ckpt["base_model_config"]
base_model = TransformerMonkey(
    tokenizer.vocab_size, tokenizer=tokenizer,
    block_size=int(cfg["block_size"]), n_embd=int(cfg["n_embd"]),
    n_head=int(cfg["n_head"]),        n_layer=int(cfg["n_layer"]),
    dropout=float(cfg["dropout"]),
).to(device)
tr_judge = TransformerJudge(base_model).to(device)
tr_judge.load_state_dict(ckpt["judge_state_dict"])
tr_judge.eval()

def score_transformer(texts):
    scores = []
    block = int(tr_judge.transformer.block_size)
    with torch.no_grad():
        for t in texts:
            ids = tokenizer.encode(t) or [0]
            ids = ids[-block:]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
            scores.append(tr_judge(x, lengths=lengths, pad_id=0).item())
    return scores

# ── build text samples ────────────────────────────────────────────────────────
print("Generating samples...")
sp   = list(load_shakespeare_dataset(0.0, 0.0))
oe   = list(load_old_english_dataset(0.0, 0.0))
bigram = BigramModel(smoothing=0.5).fit([str(x) for x in sp[:2000]])

categories = {
    "Shakespeare\n(real)":  [str(x) for x in sp[:N_SAMPLES]],
    "Old English\n(real)":  [str(x) for x in oe[:N_SAMPLES]],
    "Bigram\n(generated)":  [bigram.generate(SAMPLE_LEN, seed=i) for i in range(N_SAMPLES)],
    "Random\nnoise":         [random_monkey(SAMPLE_LEN) for _ in range(N_SAMPLES)],
}

# ── score everything ──────────────────────────────────────────────────────────
print("Scoring...")
tfidf_scores = {k: tfidf.shakespeare_likeliness(v) for k, v in categories.items()}
tr_scores    = {k: score_transformer(v)             for k, v in categories.items()}

# ── plot ──────────────────────────────────────────────────────────────────────
COLORS = {
    "Shakespeare\n(real)":  "#2ecc71",
    "Old English\n(real)":  "#3498db",
    "Bigram\n(generated)":  "#e67e22",
    "Random\nnoise":         "#e74c3c",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.patch.set_facecolor("#0f0f0f")
for ax in axes:
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

bins = np.linspace(0, 1, 25)

for cat, scores in tfidf_scores.items():
    axes[0].hist(scores, bins=bins, alpha=0.55, color=COLORS[cat],
                 edgecolor="none", label=cat.replace("\n", " "))
axes[0].set_title("TF-IDF Judge", fontsize=13, fontweight="bold", pad=10)
axes[0].set_xlabel("Shakespeare-likeness score")
axes[0].set_ylabel("Count")

for cat, scores in tr_scores.items():
    axes[1].hist(scores, bins=bins, alpha=0.55, color=COLORS[cat],
                 edgecolor="none", label=cat.replace("\n", " "))
axes[1].set_title("Transformer Judge", fontsize=13, fontweight="bold", pad=10)
axes[1].set_xlabel("Shakespeare-likeness score")

# shared legend
patches = [mpatches.Patch(color=c, label=k.replace("\n", " ")) for k, c in COLORS.items()]
fig.legend(handles=patches, loc="lower center", ncol=4, frameon=False,
           labelcolor="white", fontsize=10, bbox_to_anchor=(0.5, -0.04))

fig.suptitle("Judge Score Distributions by Text Source", color="white",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "judge_score_distributions.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved: {out_path}")
plt.show()