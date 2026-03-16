# Author(s): Priya Deshmukh
# Run the evolution experiment to evolve a line of text towards Shakespeare-likeliness as judged by a transformer-based judge. 
import random
import torch

from src.classifier.judge import Noise_Shakespeare_Classifier
from src.monkeys.evolution import evolve, mutate
from src.monkeys.transformer_monkey import train_monkey

from src.utils.load_datasets import load_old_english_dataset

print("Initializing Judge...")
judge = Noise_Shakespeare_Classifier()
judge.generate_noise("noise_dataset/noise.txt")
judge.train("noise_dataset/noise.txt")


def fitness(text):
    return judge.shakespeare_likeliness(text)[0]

print("Loading dataset...")
train_ds, val_ds, test_ds = load_old_english_dataset(0.01, 0.01, seed=13625442)

EOS_TOKEN = '\t'
full_corpus = EOS_TOKEN.join(train_ds)

print("Training Transformer...")
model, tokenizer, device = train_monkey(
    full_corpus,
    epochs=1000,
    batch_size=32,
    block_size=128
)

model.eval()

def transformer_generate(prompt="The", max_tokens=120):
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_tokens)
    return tokenizer.decode(generated[0].tolist())

print("Generating initial population from transformer...")
population_size = 100
population = [transformer_generate("The") for _ in range(population_size)]


def evolve_with_judge(population, generations=100, mutation_rate=0.01):

    for gen in range(generations):

        scored = [(fitness(ind), ind) for ind in population]
        scored.sort(reverse=True)

        best_score, best_text = scored[0]

        if gen % 10 == 0:
            print(f"\nGen {gen} | Score: {best_score:.4f}")
            print(best_text)

        survivors = [ind for score, ind in scored[:len(population)//5]]

        new_population = []
        while len(new_population) < len(population):
            parent = random.choice(survivors)
            child = mutate(parent, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_text


print("\nStarting Evolution...")
best = evolve_with_judge(population)

print("\nFinal Output:")
print(best)
