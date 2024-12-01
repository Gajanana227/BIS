import random
import operator
import math

# Constants for the genetic algorithm
POPULATION_SIZE = 100
GENERATIONS = 5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
MAX_TREE_DEPTH = 5
FUNCTIONS = ['+', '*', '/']
TERMINALS = ['x', '1', '2', '3']

# Class to represent an individual in the population
class Individual:
    def __init__(self, expression):
        self.expression = expression
        self.fitness = float('inf')

    # Function to evaluate the fitness of an individual
    def evaluate_fitness(self, x_value):
        try:
            expr = self.expression.replace('x', str(x_value))
            # Using eval to evaluate the expression
            self.fitness = eval(expr)
        except Exception as e:
            self.fitness = float('inf')

# Function to generate a random individual
def generate_random_individual():
    expression = generate_random_expression(MAX_TREE_DEPTH)
    return Individual(expression)

# Function to generate a random expression (tree-like structure)
def generate_random_expression(depth):
    if depth == 0 or random.random() < 0.3:
        # Return a terminal (e.g., x or constants)
        return random.choice(TERMINALS)
    else:
        # Return a function with two subexpressions
        function = random.choice(FUNCTIONS)
        left = generate_random_expression(depth - 1)
        right = generate_random_expression(depth - 1)
        return f"({left} {function} {right})"

# Function to perform crossover between two individuals
def crossover(parent1, parent2):
    # For simplicity, we just swap subexpressions between two individuals
    expr1, expr2 = parent1.expression, parent2.expression
    split1 = random.choice(expr1.split())
    split2 = random.choice(expr2.split())
    offspring_expr = expr1.replace(split1, split2, 1)
    return Individual(offspring_expr)

# Function to mutate an individual
def mutate(individual):
    if random.random() < MUTATION_RATE:
        # Replace a random part of the expression with a new one
        mutated_expr = individual.expression
        split_expr = mutated_expr.split()
        mutated_expr = mutated_expr.replace(random.choice(split_expr), generate_random_expression(MAX_TREE_DEPTH), 1)
        individual.expression = mutated_expr

# Function to select the best individual
def select_best_individual(population, x_value):
    best_individual = min(population, key=lambda ind: ind.fitness)
    best_individual.evaluate_fitness(x_value)
    return best_individual

# Main function to run the GEP algorithm
def run_gep_algorithm():
    population = [generate_random_individual() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        # Evaluate fitness for each individual
        for individual in population:
            individual.evaluate_fitness(3)  # Example with x=3

        # Select the best individual
        best_individual = select_best_individual(population, 3)

        # Print the fitness of the best individual in each generation
        print(f"Generation {generation + 1}: Best fitness = {best_individual.fitness}")

        # Create a new population using crossover and mutation
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            if random.random() < CROSSOVER_RATE:
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                offspring = crossover(parent1, parent2)
                new_population.append(offspring)
            else:
                individual = random.choice(population)
                mutate(individual)
                new_population.append(individual)

        population = new_population

# Run the algorithm
if __name__ == "__main__":
    run_gep_algorithm()
