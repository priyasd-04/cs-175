# Authors: Davin Makris
# Early experiment showing the evolution of outputs corresponding with increase in judge score, using old tfidf judge.

import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import src.classifier.judge
from src.monkeys.evolution import *

target = "So is it in the music of men's lives. And here have I the daintiness of ear"

sj = src.classifier.judge.Noise_Shakespeare_Classifier()
sj.train("noise_dataset/noise.txt")

def evolve2(target, population_size=200, max_generations=1000, mutation_rate=0.02):
    length = len(target)
    population = [random_monkey(length) for _ in range(population_size)]
    
    best_scores = []
    best_individuals = []

    for generation in range(max_generations):
        scored = [(fitness(individual, target), individual) for individual in population]
        scored.sort(reverse=True)
        
        best_score, best_individual = scored[0]

        best_scores.append(best_score)
        best_individuals.append(best_individual)
        
        if generation % 50 == 0:
            print(f"Gen {generation:4d} | Fitness: {best_score:.3f} | {best_individual}")

        if best_score >= 0.99:
            break
        
        surviving_monkeys = [individual for score, individual in scored[:population_size // 5]]
        
        new_population = []
        while len(new_population) < population_size:
            parent = random.choice(surviving_monkeys)
            child = mutate(parent, mutation_rate)
            new_population.append(child)
            
        population = new_population
        
    return best_individuals, best_scores
        

best_individuals, best_scores = evolve2(target, max_generations=5000)

probas = sj.shakespeare_likeliness(best_individuals)

results_dir = Path(REPO_ROOT) / "results"
results_dir.mkdir(exist_ok=True)

csv_path = results_dir / "evolution_proba_across_generations.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["generation", "fitness", "judge_prob"])
    for i, (score, prob) in enumerate(zip(best_scores, probas)):
        writer.writerow([i, f"{score:.6f}", f"{prob:.6f}"])
print(f"Saved CSV: {csv_path}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(best_scores, label="Fitness (char match)", linewidth=1.5)
ax.plot(probas, label="Judge Prob (Shakespeare)", linewidth=1.5)
ax.set_xlabel("Generation")
ax.set_ylabel("Score")
ax.set_title("Evolution Progress: Fitness vs Judge Score")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

plot_path = results_dir / "evolution_proba_across_generations.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Saved plot: {plot_path}")