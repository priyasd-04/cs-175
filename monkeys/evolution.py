import random
from monkeys.generator import random_monkey

# calculate fitness score of a string compared to target
def fitness(string, target):
    score = 0
    for i in range(len(target)):
        if string[i] == target[i]:
            score += 1
    return score / len(target)

def mutate(string, mutation_rate=0.02):
    alphabet = "abcdefghijklmnopqrstuvwxyz .,;:!?"
    new_text = list(string)
    for i in range(len(new_text)):
        if random.random() < mutation_rate:
            new_text[i] = random.choice(alphabet)
    return "".join(new_text)

def evolve(target, population_size=200, max_generations=1000, mutation_rate=0.02):
    length = len(target)
    population = [random_monkey(length) for _ in range(population_size)]
    
    best_scores = []
    
    for generation in range(max_generations):
        scored = [(fitness(individual, target), individual) for individual in population]
        scored.sort(reverse=True)
        
        best_score, best_individual = scored[0]
        best_scores.append(best_score)
        
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
        
    return best_individual, best_scores
        