import random
from monkeys.generator import random_monkey
from classifier.judge import Noise_Shakespeare_Classifier
import torch
from classifier.transformer_judge import TransformerJudge

print("Initializing judge...")
# judge = Noise_Shakespeare_Classifier()
judge = TransformerJudge(base_model=torch.load("transformer_judge.pt"))
judge.generate_noise("noise_dataset/noise.txt")
judge.train("noise_dataset/noise.txt")

# calculate fitness score of a string compared to target
def fitness(string, tokenizer, device):
    encoded_text = torch.tensor([tokenizer.encode(string)], dtype=torch.long, device=device)
    encoded_text = encoded_text[:, -judge.transformer.block_size:] # clamp to block size
    with torch.no_grad():
        score = judge(encoded_text).item()
    return score
    # return judge.shakespeare_likeliness(string)[0]

def mutate(string, mutation_rate=0.02):
    alphabet = "abcdefghijklmnopqrstuvwxyz .,;:!?"
    new_text = list(string)
    for i in range(len(new_text)):
        if random.random() < mutation_rate:
            new_text[i] = random.choice(alphabet)
    return "".join(new_text)

def evolve(population_size=200, max_generations=300, mutation_rate=0.02):
    # length = len(target)
    # population = [random_monkey(length) for _ in range(population_size)]
    
    best_scores = []
    top_k = 10
    
    for generation in range(max_generations):
        scored = [(fitness(ind), ind) for ind in population]
        scored.sort(reverse=True)
        
        best_score, best_individual = scored[0]
        best_scores.append(best_score)
        
        if generation % 10 == 0:
            print(f"Gen {generation:4d} | Fitness: {best_score:.3f} | {best_individual}")

        # if best_score >= 0.99:
            # break
        
        surviving_monkeys = [individual for score, individual in scored[:top_k]]
        
        # weighted selection by fitness score so better monkeys reproduce more
        scores_only = [score for score, _ in scored[:top_k]]
        total = sum(scores_only)
        weights = [s / total for s in scores_only]
        
        new_population = []
        while len(new_population) < population_size:
            parent = random.choice(surviving_monkeys, weights=weights, k=1)[0]
            child = mutate(parent, mutation_rate)
            new_population.append(child)
            
        population = new_population
        
    return best_individual, best_scores
        