# evolution.py
import copy
import torch
import random
from monkeys.TransformerMonkey import TransformerMonkey, train_monkey
from classifier.transformer_judge import TransformerJudge
from utils.load_datasets import load_old_english_dataset

def mutate_model(model, mutation_rate=0.01, mutation_strength=0.05):
    """Copy a model and change its weights slightly."""
    child = copy.deepcopy(model)
    with torch.no_grad():
        for param in child.parameters():
            if random.random() < mutation_rate:
                # Add gaussian noise scaled by mutation_strength
                noise = torch.randn_like(param) * mutation_strength
                param.add_(noise)
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

def score_monkey(model, judge, tokenizer, device, n_samples=3):
    """
    Score monkey by averaging judge scores across multiple samples.
    """
    scores = []
    for _ in range(n_samples):
        text = generate_sample(model, tokenizer, device)
        encoded = torch.tensor(
            [tokenizer.encode(text)], dtype=torch.long, device=device
        )
        encoded = encoded[:, -judge.transformer.block_size:]  # clamp to block size
        with torch.no_grad():
            score = judge(encoded).item()
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
    mutation_rate=0.3, # probability that any given parameter tensor gets mutated
    mutation_strength=0.05, # how large the weight perturbations are
):
    # seed population from base model with initial mutations
    population = [mutate_model(base_model, mutation_rate, mutation_strength) 
                  for _ in range(population_size)]

    best_monkey = None
    best_score_ever = -1

    for generation in range(max_generations):
        # score every monkey
        scored = [
            (score_monkey(m, judge, tokenizer, device), m) 
            for m in population
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_monkey_this_gen = scored[0]
        
        # track overall best
        if best_score > best_score_ever:
            best_score_ever = best_score
            best_monkey = copy.deepcopy(best_monkey_this_gen)

        if generation % 5 == 0:
            sample = generate_sample(best_monkey_this_gen, tokenizer, device)
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
            parent = random.choices(survivors, weights=weights, k=1)[0]
            child = mutate_model(parent, mutation_rate, mutation_strength)
            new_population.append(child)

        population = new_population

    return best_monkey, best_score_ever


if __name__ == '__main__':
    from utils.load_datasets import load_old_english_dataset
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Training base monkey...")
    train_ds = load_old_english_dataset(0, 0, seed=13625442)
    full_corpus = '\t'.join(train_ds)
    base_model, tokenizer, device = train_monkey(full_corpus, epochs=1000, batch_size=32, block_size=128)

    print("Loading judge...")
    judge_base = torch.load("transformer_judge.pt")
    judge = TransformerJudge(judge_base)
    judge = judge.to(device)
    
    print("Evolving...")
    best_monkey, best_score = evolve(
        base_model=base_model,
        judge=judge,
        tokenizer=tokenizer,
        device=device,
        population_size=20,
        max_generations=50,
        top_k=5,
    )

    torch.save(best_monkey, "best_monkey.pt")
    print(f"\nBest score achieved: {best_score:.4f}")
    print("Final sample:")
    print(generate_sample(best_monkey, tokenizer, device, max_new_tokens=500))