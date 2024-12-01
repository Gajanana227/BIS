import numpy as np

# Define the objective function
def objective_function(x):
    return x**2

# Particle Swarm Optimization implementation
def particle_swarm_optimization(obj_func, num_particles=30, num_iterations=5, bounds=(-10, 10), w=0.5, c1=1.5, c2=1.5):
    # Initialize particle positions and velocities
    positions = np.random.uniform(bounds[0], bounds[1], num_particles)
    velocities = np.random.uniform(-1, 1, num_particles)
    
    # Initialize personal best positions and global best
    personal_best_positions = positions.copy()
    personal_best_scores = obj_func(personal_best_positions)
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = obj_func(global_best_position)
    
    # PSO iterations
    for iteration in range(num_iterations):
        for i in range(num_particles):
            # Update velocity
            r1, r2 = np.random.rand(), np.random.rand()  # Random factors
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_positions[i] - positions[i])
                + c2 * r2 * (global_best_position - positions[i])
            )
            
            # Update position
            positions[i] += velocities[i]
            
            # Enforce bounds
            positions[i] = np.clip(positions[i], bounds[0], bounds[1])
            
            # Update personal best
            score = obj_func(positions[i])
            if score < personal_best_scores[i]:
                personal_best_positions[i] = positions[i]
                personal_best_scores[i] = score
        
        # Update global best
        best_particle_index = np.argmin(personal_best_scores)
        if personal_best_scores[best_particle_index] < global_best_score:
            global_best_position = personal_best_positions[best_particle_index]
            global_best_score = personal_best_scores[best_particle_index]
        
        # Print iteration details (optional)
        print(f"Iteration {iteration+1}/{num_iterations}, Global Best Score: {global_best_score}")
    
    return global_best_position, global_best_score

# Run PSO to minimize f(x) = x^2
best_position, best_score = particle_swarm_optimization(objective_function)

# Output the results
print("\nOptimization Results:")
print(f"Best Position: {best_position}")
print(f"Best Score: {best_score}")
