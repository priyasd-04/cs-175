# Author(s): Amish Kunal
"""Evolve a transformer monkey's lm_head using a fixed transformer judge."""
import copy
import argparse
from pathlib import Path
import sys
import torch
import random

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monkeys.TransformerMonkey import TransformerMonkey
from classifier.transformer_judge import TransformerJudge

def _lm_head_params(model):
    if not hasattr(model, "lm_head"):
        raise AttributeError("Model has no lm_head. Expected a TransformerMonkey-like model.")
    return list(model.lm_head.parameters())


def mutate_model(model, mutation_rate=0.3, mutation_strength=0.05):
    """Copy a model and mutate ONLY the lm_head weights (not the full network)."""
    child = copy.deepcopy(model)
    with torch.no_grad():
        for param in _lm_head_params(child):
            if random.random() < mutation_rate:
                # Add gaussian noise scaled by mutation_strength
                noise = torch.randn_like(param) * mutation_strength
                param.add_(noise)
    return child


def crossover_lm_head(parent_a, parent_b, *, p_swap=0.5):
    """Create a child by mixing lm_head weights from two parents."""
    child = copy.deepcopy(parent_a)
    with torch.no_grad():
        for pa, pb, pc in zip(_lm_head_params(parent_a), _lm_head_params(parent_b), _lm_head_params(child)):
            # element-wise mask: choose from parent_b with probability p_swap
            mask = torch.rand_like(pc, dtype=torch.float32) < float(p_swap)
            pc.copy_(torch.where(mask, pb, pa))
    return child

def generate_sample(model, tokenizer, device, prompt=None, max_new_tokens=200):
    """Generate a text sample from a monkey model."""
    model.eval()
    if prompt is None:
        # start from a random seed token
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        start = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = model.generate(start, max_new_tokens=max_new_tokens)
    
    return tokenizer.decode(generated[0].tolist())

def score_monkey(model, judge, tokenizer, device, n_samples=3, *, prompt=None, max_new_tokens=200):
    """Score monkey by averaging judge scores across multiple samples."""
    scores = []
    for _ in range(n_samples):
        text = generate_sample(model, tokenizer, device, prompt=prompt, max_new_tokens=max_new_tokens)
        ids = tokenizer.encode(text)
        if not ids:
            ids = [0]
        block = int(judge.transformer.block_size)
        ids = ids[-block:]
        encoded = torch.tensor([ids], dtype=torch.long, device=device)
        lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
        with torch.no_grad():
            score = judge(encoded, lengths=lengths, pad_id=0).item()
        scores.append(score)
    return sum(scores) / len(scores)

def evolve(
    base_model,        # pretrained TransformerMonkey to seed the population
    judge,             # TransformerJudge
    tokenizer,
    device,
    population_size=20,
    max_generations=50,
    top_k=5,           # how many survive each generation
    score_samples=2,   # how many generated samples to average per monkey
    prompt=None,       # optional prompt for generation
    max_new_tokens=200,
    mutation_rate=0.3, # probability that any lm_head parameter tensor gets mutated
    mutation_strength=0.05, # how large the weight perturbations are
    crossover_rate=0.7, # probability we crossover (else clone+mutate)
    crossover_swap_p=0.5, # element-wise probability to take weights from parent_b
):
    # Seed population from base model with initial lm_head mutations.
    population = [mutate_model(base_model, mutation_rate, mutation_strength) for _ in range(population_size)]

    best_monkey = None
    best_score_ever = -1

    for generation in range(max_generations):
        # score every monkey
        scored = [
            (score_monkey(m, judge, tokenizer, device, n_samples=score_samples, prompt=prompt, max_new_tokens=max_new_tokens), m)
            for m in population
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_monkey_this_gen = scored[0]
        
        # track overall best
        if best_score > best_score_ever:
            best_score_ever = best_score
            best_monkey = copy.deepcopy(best_monkey_this_gen)

        if generation % 5 == 0:
            sample = generate_sample(best_monkey_this_gen, tokenizer, device, prompt=prompt, max_new_tokens=max_new_tokens)
            print(f"\nGen {generation:4d} | Best Score: {best_score:.4f}")
            print(f"Sample: {sample[:200]}")

        # keep top k survivors
        survivors = [m for score, m in scored[:top_k]]

        # weighted reproduction so better monkeys have more children
        top_scores = [score for score, _ in scored[:top_k]]
        total = sum(top_scores)
        weights = [s / total for s in top_scores]

        new_population = []
        # take best monkey directly to next gen unchanged
        new_population.append(copy.deepcopy(survivors[0]))

        while len(new_population) < population_size:
            if random.random() < crossover_rate and len(survivors) >= 2:
                parent_a, parent_b = random.choices(survivors, weights=weights, k=2)
                child = crossover_lm_head(parent_a, parent_b, p_swap=crossover_swap_p)
            else:
                parent = random.choices(survivors, weights=weights, k=1)[0]
                child = copy.deepcopy(parent)

            child = mutate_model(child, mutation_rate, mutation_strength)
            new_population.append(child)

        population = new_population

    return best_monkey, best_score_ever


def _load_judge_ckpt(path: str | Path, device: str):
    """Load the checkpoint created by expirements/train_transformer_judge.py."""
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    tokenizer = ckpt["tokenizer"]
    cfg = ckpt["base_model_config"]

    base = TransformerMonkey(
        tokenizer.vocab_size,
        tokenizer=tokenizer,
        block_size=int(cfg["block_size"]),
        n_embd=int(cfg["n_embd"]),
        n_head=int(cfg["n_head"]),
        n_layer=int(cfg["n_layer"]),
        dropout=float(cfg.get("dropout", 0.2)),
    ).to(device)

    judge = TransformerJudge(base).to(device)
    judge.load_state_dict(ckpt["judge_state_dict"])
    judge.eval()
    return judge, tokenizer


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evolve only lm_head using a fixed transformer judge")
    ap.add_argument("--judge-ckpt", type=str, default="models/transformer_judge.pt")
    ap.add_argument("--save-lm-head", type=str, default="models/best_evolved_lm_head.pt")
    ap.add_argument("--population", type=int, default=20)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--samples-per-score", type=int, default=2)
    ap.add_argument("--mutation-rate", type=float, default=0.3)
    ap.add_argument("--mutation-strength", type=float, default=0.05)
    ap.add_argument("--crossover-rate", type=float, default=0.7)
    ap.add_argument("--crossover-swap-p", type=float, default=0.5)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    judge, tokenizer = _load_judge_ckpt(args.judge_ckpt, device=device)
    base_model = copy.deepcopy(judge.transformer)

    best_monkey, best_score = evolve(
        base_model=base_model,
        judge=judge,
        tokenizer=tokenizer,
        device=device,
        population_size=args.population,
        max_generations=args.generations,
        top_k=args.top_k,
        score_samples=args.samples_per_score,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        crossover_rate=args.crossover_rate,
        crossover_swap_p=args.crossover_swap_p,
    )

    print(f"\nBest score achieved: {best_score:.4f}")
    print("Final sample:")
    print(generate_sample(best_monkey, tokenizer, device, prompt=args.prompt, max_new_tokens=300))

    if args.save_lm_head:
        save_path = (REPO_ROOT / args.save_lm_head).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"lm_head": best_monkey.lm_head.state_dict(), "best_score": float(best_score)}, save_path)
        print(f"Saved evolved lm_head: {save_path}")