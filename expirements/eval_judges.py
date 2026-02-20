"""Compare TF-IDF judge vs transformer judge.

Run:
  python3 expirements/eval_judges.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from classifier.judge import Noise_Shakespeare_Classifier
from classifier.transformer_judge import TransformerJudge
from monkeys.bigram import BigramModel
from monkeys.generator import random_monkey
from monkeys.TransformerMonkey import TransformerMonkey, CharTokenizer
from utils.load_datasets import load_old_english_dataset, load_shakespeare_dataset


def _summarize(scores: list[float]) -> tuple[float, float]:
    if not scores:
        return float("nan"), float("nan")
    return float(mean(scores)), float(max(scores))


def _score_transformer_judge(
    judge: TransformerJudge,
    tokenizer: CharTokenizer,
    texts: list[str],
    *,
    device: str,
) -> list[float]:
    out: list[float] = []
    judge.eval()
    block_size = int(judge.transformer.block_size)
    with torch.no_grad():
        for t in texts:
            ids = tokenizer.encode(t)
            if not ids:
                ids = [0]
            if len(ids) > block_size:
                ids = ids[-block_size:]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
            p = judge(x, lengths=lengths, pad_id=0).item()
            out.append(float(p))
    return out


def _generate_transformer_samples(
    model: TransformerMonkey,
    tokenizer: CharTokenizer,
    *,
    device: str,
    n: int,
    prompt: str,
    max_new_tokens: int,
) -> list[str]:
    model.eval()
    out: list[str] = []
    with torch.no_grad():
        for i in range(n):
            ids = tokenizer.encode(prompt)
            if not ids:
                ids = [0]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            y = model.generate(x, max_new_tokens=max_new_tokens)
            out.append(tokenizer.decode(y[0].tolist()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=60)
    ap.add_argument("--sample-len", type=int, default=300)
    ap.add_argument("--shakespeare-fit-lines", type=int, default=2000)
    ap.add_argument("--noise-file", type=str, default="noise_dataset/noise.txt")
    ap.add_argument("--real-lines", type=int, default=60, help="How many real dataset lines to score")

    ap.add_argument("--judge-ckpt", type=str, default="models/transformer_judge.pt")
    ap.add_argument("--transformer-prompt", type=str, default="To be, or not to be ")
    ap.add_argument("--transformer-new", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # TF-IDF judge
    print("Loading TF-IDF judge...")
    tfidf = Noise_Shakespeare_Classifier()
    noise_path = (REPO_ROOT / args.noise_file).resolve()
    if not noise_path.exists():
        Noise_Shakespeare_Classifier.generate_noise(str(noise_path))
    tfidf.train(str(noise_path), show_report=False)

    # Bigram (trained on Shakespeare)
    print("Training bigram on Shakespeare...")
    sp = load_shakespeare_dataset(0.0, 0.0)
    bigram = BigramModel(smoothing=0.5).fit(list(sp[: args.shakespeare_fit_lines]))

    # Sources
    print("Generating samples...")
    random_texts = [random_monkey(args.sample_len) for _ in range(args.n_samples)]
    bigram_texts = [bigram.generate(args.sample_len, seed=i) for i in range(args.n_samples)]
    real_sp = [str(x) for x in list(sp[: args.real_lines])]
    oe = load_old_english_dataset(0.0, 0.0)
    real_oe = [str(x) for x in list(oe[: args.real_lines])]

    # Transformer judge (optional if checkpoint exists)
    judge_ckpt_path = (REPO_ROOT / args.judge_ckpt).resolve()
    transformer_judge = None
    transformer_tokenizer = None
    transformer_texts: list[str] = []

    if judge_ckpt_path.exists():
        print(f"Loading transformer judge checkpoint: {judge_ckpt_path}")
        # Newer PyTorch defaults to weights_only=True; we store a tokenizer in the checkpoint.
        try:
            ckpt = torch.load(judge_ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(judge_ckpt_path, map_location=device)
        transformer_tokenizer = ckpt["tokenizer"]
        cfg = ckpt["base_model_config"]

        base_model = TransformerMonkey(
            transformer_tokenizer.vocab_size,
            tokenizer=transformer_tokenizer,
            block_size=int(cfg["block_size"]),
            n_embd=int(cfg["n_embd"]),
            n_head=int(cfg["n_head"]),
            n_layer=int(cfg["n_layer"]),
            dropout=float(cfg["dropout"]),
        ).to(device)

        transformer_judge = TransformerJudge(base_model).to(device)
        transformer_judge.load_state_dict(ckpt["judge_state_dict"])
        transformer_judge.eval()

        transformer_texts = _generate_transformer_samples(
            transformer_judge.transformer,
            transformer_tokenizer,
            device=device,
            n=args.n_samples,
            prompt=args.transformer_prompt,
            max_new_tokens=args.transformer_new,
        )
    else:
        print(f"Transformer judge checkpoint not found: {judge_ckpt_path}")
        print("Skipping transformer judge + transformer sample source.")

    # Score with TF-IDF judge
    tfidf_scores = {
        "random": tfidf.shakespeare_likeliness(random_texts),
        "bigram": tfidf.shakespeare_likeliness(bigram_texts),
        "transformer": tfidf.shakespeare_likeliness(transformer_texts) if transformer_texts else [],
        "real_sp": tfidf.shakespeare_likeliness(real_sp),
        "real_oe": tfidf.shakespeare_likeliness(real_oe),
    }

    # Score with transformer judge
    tr_scores = {"random": [], "bigram": [], "transformer": [], "real_sp": [], "real_oe": []}
    if transformer_judge is not None and transformer_tokenizer is not None:
        tr_scores["random"] = _score_transformer_judge(
            transformer_judge, transformer_tokenizer, random_texts, device=device
        )
        tr_scores["bigram"] = _score_transformer_judge(
            transformer_judge, transformer_tokenizer, bigram_texts, device=device
        )
        tr_scores["transformer"] = _score_transformer_judge(
            transformer_judge, transformer_tokenizer, transformer_texts, device=device
        )
        tr_scores["real_sp"] = _score_transformer_judge(
            transformer_judge, transformer_tokenizer, real_sp, device=device
        )
        tr_scores["real_oe"] = _score_transformer_judge(
            transformer_judge, transformer_tokenizer, real_oe, device=device
        )

    # Print table
    print("\n== Judge comparison (mean / max) ==")
    print(f"{'source':<12} | {'tfidf':>18} | {'transformer':>18}")
    print("-" * 56)
    for src in ("real_sp", "real_oe", "random", "bigram", "transformer"):
        a_mean, a_max = _summarize(list(tfidf_scores[src]))
        b_mean, b_max = _summarize(list(tr_scores[src]))
        tfidf_str = f"{a_mean:.3f} / {a_max:.3f}" if tfidf_scores[src] else "n/a"
        tr_str = f"{b_mean:.3f} / {b_max:.3f}" if tr_scores[src] else "n/a"
        print(f"{src:<12} | {tfidf_str:>18} | {tr_str:>18}")


if __name__ == "__main__":
    main()

