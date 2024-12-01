import numpy as np

# Objective function: Example energy consumption function
def objective_function(x):
    # Simulate a function to optimize (e.g., energy consumption)
    return np.sum(x**2)  # Minimize the energy consumption

# Levy flight step
def levy_flight(Lambda):
    # Generate step sizes for Levy flights
    u = np.random.normal(0, 1, 1)
    v = np.random.normal(0, 1, 1)
    step = u / abs(v)**(1 / Lambda)
    return step

# Cuckoo Search Algorithm
def cuckoo_search(n, max_iter, dim, bounds, pa=0.25, alpha=0.01, Lambda=1.5):
    # Initialize n nests with random solutions
    nests = np.random.uniform(bounds[0], bounds[1], (n, dim))
    fitness = np.array([objective_function(nest) for nest in nests])
    best_nest = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    for iteration in range(max_iter):
        # Generate new solutions using Levy flights
        new_nests = np.copy(nests)
        for i in range(n):
            step_size = alpha * levy_flight(Lambda)
            new_nests[i] += step_size * (nests[i] - best_nest)
            new_nests[i] = np.clip(new_nests[i], bounds[0], bounds[1])

        # Evaluate new solutions
        new_fitness = np.array([objective_function(nest) for nest in new_nests])
        
        # Replace nests based on fitness
        for i in range(n):
            if new_fitness[i] < fitness[i]:
                nests[i] = new_nests[i]
                fitness[i] = new_fitness[i]
        
        # Abandon some nests (with probability pa)
        for i in range(n):
            if np.random.rand() < pa:
                nests[i] = np.random.uniform(bounds[0], bounds[1], dim)
                fitness[i] = objective_function(nests[i])
        
        # Update best solution
        best_nest = nests[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        
        print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {best_fitness:.4f}")
    
    return best_nest, best_fitness

# Parameters
n = 20               # Number of nests
max_iter = 10       # Maximum iterations
dim = 5              # Number of dimensions (e.g., parameters in wireless network)
bounds = [-10, 10]   # Search space bounds
pa = 0.25            # Probability of abandoning nests
alpha = 0.01         # Step size scaling factor

# Run the Cuckoo Search Algorithm
best_solution, best_fitness = cuckoo_search(n, max_iter, dim, bounds, pa, alpha)

print(f"Best Solution: {best_solution}")
print(f"Best Fitness (Optimized Energy): {best_fitness:.4f}")
