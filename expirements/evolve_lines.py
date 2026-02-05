from monkeys.evolution import evolve

target = "to be or not to be, that is the question"

best_individual, best_scores = evolve(target, max_generations=1500)
print("\nFinal Output:")
print(best_individual)

