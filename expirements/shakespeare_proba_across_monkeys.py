# Authors: Davin Makris
# Early experiment showing the evolution of outputs corresponding with increase in judge score, using old tfidf judge.

import classifier.judge
from monkeys.evolution import *
import matplotlib.pyplot as plt

target = "So is it in the music of men's lives. And here have I the daintiness of ear"

sj = classifier.judge.Noise_Shakespeare_Classifier()
sj.train("noise_dataset/noise.txt")

def evolve2(target, population_size=200, max_generations=1000, mutation_rate=0.02): #borrowing this, but taking all best monkeys
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

def plot_evolution_progress(scores, probabilities):
    plt.plot(scores, label='Fitness')
    plt.plot(probabilities, label='Judge Prob')

    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


plot_evolution_progress(best_scores, probas)