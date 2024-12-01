import random

# Define the fitness function
def fitness_function(x):
    return x ** 2

# Generate initial population
def generate_population(size, lower_bound, upper_bound):
    return [random.uniform(lower_bound, upper_bound) for _ in range(size)]

# Selection - select individuals based on fitness
def selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]
    selected = random.choices(population, weights=probabilities, k=len(population))
    return selected

# Crossover - create new offspring by combining parents
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2
    else:
        return parent1, parent2

# Mutation - introduce random variations
def mutate(individual, mutation_rate, lower_bound, upper_bound):
    if random.random() < mutation_rate:
        individual += random.uniform(-1, 1)
        individual = max(lower_bound, min(upper_bound, individual))  # Keep within bounds
    return individual

# Genetic Algorithm
def genetic_algorithm(population_size, lower_bound, upper_bound, generations, mutation_rate, crossover_rate):
    population = generate_population(population_size, lower_bound, upper_bound)

    for generation in range(generations):
        # Evaluate fitness
        fitness_values = [fitness_function(ind) for ind in population]

        # Selection
        selected_population = selection(population, fitness_values)

        # Crossover
        next_generation = []
        for i in range(0, len(selected_population), 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1 if i + 1 < len(selected_population) else 0]
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            next_generation.extend([child1, child2])

        # Mutation
        population = [mutate(ind, mutation_rate, lower_bound, upper_bound) for ind in next_generation]

        # Log best fitness of the generation
        best_fitness = max(fitness_values)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.4f}")

    # Return the best fitness value from the final generation
    return max(fitness_function(ind) for ind in population)

# Parameters
population_size = 10
lower_bound = -10
upper_bound = 10
generations = 9
mutation_rate = 0.1
crossover_rate = 0.8

# Run Genetic Algorithm
best_fitness = genetic_algorithm(population_size, lower_bound, upper_bound, generations, mutation_rate, crossover_rate)
print(f"Best fitness found: {best_fitness:.4f}")
