# Author(s): Amish Kunal
"""Run 0-shot / few-shot LLM judge baseline on prepared eval examples.

Uses JSONL produced by:
  python3 expirements/build_llm_judge_eval_set.py

Default backend is Ollama CLI.

Run:
  python3 expirements/run_llm_judge_baseline.py --model gemma3:4b
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Row:
    source: str
    label: int
    text: str
    zero_shot_prompt: str
    few_shot_prompt: str


def _read_jsonl(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                Row(
                    source=str(obj["source"]),
                    label=int(obj["label"]),
                    text=str(obj["text"]),
                    zero_shot_prompt=str(obj["zero_shot_prompt"]),
                    few_shot_prompt=str(obj["few_shot_prompt"]),
                )
            )
    return rows


def _predict_ollama(*, model: str, prompt: str, timeout_s: int) -> str:
    """Return raw model output string from Ollama."""
    proc = subprocess.run(
        ["ollama", "run", model, prompt],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(f"ollama failed (code={proc.returncode}): {stderr[:300]}")
    return proc.stdout.strip()


def _parse_label(output: str) -> tuple[int, str]:
    """Parse LLM output into binary class.

    Expected token:
      SHAKESPEARE / NOT_SHAKESPEARE
    """
    text = output.strip().upper()
    compact = " ".join(text.split())

    if "NOT_SHAKESPEARE" in compact or "NOT SHAKESPEARE" in compact:
        return 0, "parsed_not_shakespeare"
    if "SHAKESPEARE" in compact:
        return 1, "parsed_shakespeare"

    if compact.startswith("YES"):
        return 1, "fallback_yes"
    if compact.startswith("NO"):
        return 0, "fallback_no"

    return 0, "fallback_default_negative"


def _compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec) / max(1e-9, prec + rec)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _run_mode(
    rows: list[Row],
    *,
    mode: str,
    model: str,
    timeout_s: int,
    sleep_ms: int,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, dict[str, float]]]:
    assert mode in ("zero_shot", "few_shot")
    y_true: list[int] = []
    y_pred: list[int] = []
    by_source: dict[str, list[tuple[int, int]]] = defaultdict(list)
    out_rows: list[dict[str, object]] = []

    for i, r in enumerate(rows, start=1):
        prompt = r.zero_shot_prompt if mode == "zero_shot" else r.few_shot_prompt
        raw = _predict_ollama(model=model, prompt=prompt, timeout_s=timeout_s)
        pred, parse_note = _parse_label(raw)

        y_true.append(r.label)
        y_pred.append(pred)
        by_source[r.source].append((r.label, pred))

        out_rows.append(
            {
                "mode": mode,
                "index": i,
                "source": r.source,
                "label": r.label,
                "pred": pred,
                "parse_note": parse_note,
                "raw_output": raw,
            }
        )

        if i % 20 == 0:
            print(f"[{mode}] processed {i}/{len(rows)} examples")
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    overall = _compute_metrics(y_true, y_pred)
    per_source: dict[str, dict[str, float]] = {}
    for src, pairs in sorted(by_source.items()):
        t = [a for a, _ in pairs]
        p = [b for _, b in pairs]
        per_source[src] = _compute_metrics(t, p)

    return out_rows, overall, per_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="results/llm_judge_eval_examples.jsonl")
    ap.add_argument("--model", type=str, default="gemma3:4b")
    ap.add_argument("--timeout-s", type=int, default=120)
    ap.add_argument("--sleep-ms", type=int, default=0, help="Optional delay between calls.")
    ap.add_argument(
        "--modes",
        type=str,
        default="zero_shot,few_shot",
        help="Comma-separated: zero_shot,few_shot",
    )
    ap.add_argument("--max-examples", type=int, default=None, help="Optional cap for quick testing.")
    ap.add_argument("--out-prefix", type=str, default="results/llm_judge_baseline")
    args = ap.parse_args()

    in_path = (REPO_ROOT / args.input).resolve()
    if not in_path.exists():
        raise FileNotFoundError(
            f"Input JSONL not found: {in_path}\n"
            "Run: python3 expirements/build_llm_judge_eval_set.py"
        )

    rows = _read_jsonl(in_path)
    if args.max_examples is not None:
        rows = rows[: int(args.max_examples)]
    if not rows:
        raise RuntimeError("No input examples found.")

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    for m in modes:
        if m not in ("zero_shot", "few_shot"):
            raise ValueError(f"Unsupported mode: {m}")

    print(f"Input examples: {len(rows)}")
    print(f"Model: {args.model}")
    print(f"Modes: {modes}")

    out_prefix = (REPO_ROOT / args.out_prefix).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_pred_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for mode in modes:
        print(f"\nRunning mode: {mode}")
        pred_rows, overall, per_source = _run_mode(
            rows,
            mode=mode,
            model=args.model,
            timeout_s=int(args.timeout_s),
            sleep_ms=int(args.sleep_ms),
        )
        all_pred_rows.extend(pred_rows)

        summary_rows.append({"mode": mode, "source": "overall", **overall})
        for src, metrics in per_source.items():
            summary_rows.append({"mode": mode, "source": src, **metrics})

        print(
            f"{mode} overall | "
            f"acc={overall['accuracy']:.3f} "
            f"prec={overall['precision']:.3f} "
            f"rec={overall['recall']:.3f} "
            f"f1={overall['f1']:.3f}"
        )

    pred_csv = Path(str(out_prefix) + "_predictions.csv")
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["mode", "index", "source", "label", "pred", "parse_note", "raw_output"],
        )
        w.writeheader()
        for r in all_pred_rows:
            w.writerow(r)

    summary_csv = Path(str(out_prefix) + "_summary.csv")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "source",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "tp",
                "tn",
                "fp",
                "fn",
            ],
        )
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"\nSaved predictions: {pred_csv}")
    print(f"Saved summary:     {summary_csv}")


if __name__ == "__main__":
    main()

