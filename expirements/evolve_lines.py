# Author(s): Priya Deshmukh
# Run the evolution experiment to evolve a line of text towards a target.
# Used as initial project baseline on random monkeys to verify evolution logic and push to target
from monkeys.evolution import evolve

target = "to be or not to be, that is the question"

best_individual, best_scores = evolve(target, max_generations=1500)
print("\nFinal Output:")
print(best_individual)

