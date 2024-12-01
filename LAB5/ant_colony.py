import numpy as np

# Parameters
NUM_CITIES = 10  # Number of cities
NUM_ANTS = 20    # Number of ants
ITERATIONS = 10 # Number of iterations
ALPHA = 1.0      # Pheromone importance
BETA = 2.0       # Heuristic importance
EVAPORATION_RATE = 0.5
Q = 100          # Pheromone deposit factor

# Distance matrix
distance_matrix = np.random.randint(1, 100, size=(NUM_CITIES, NUM_CITIES))
np.fill_diagonal(distance_matrix, 0)

# Initialize pheromone levels
pheromones = np.ones((NUM_CITIES, NUM_CITIES))

def calculate_route_length(route):
    length = 0
    for i in range(len(route) - 1):
        length += distance_matrix[route[i], route[i + 1]]
    length += distance_matrix[route[-1], route[0]]  # Return to the start city
    return length

def construct_route(start_city):
    route = [start_city]
    for _ in range(NUM_CITIES - 1):
        current_city = route[-1]
        probabilities = []
        for next_city in range(NUM_CITIES):
            if next_city not in route:
                prob = (pheromones[current_city, next_city] ** ALPHA) * \
                       ((1 / distance_matrix[current_city, next_city]) ** BETA)
                probabilities.append(prob)
            else:
                probabilities.append(0)
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        next_city = np.random.choice(range(NUM_CITIES), p=probabilities)
        route.append(next_city)
    return route

def update_pheromones(pheromones, all_routes, all_lengths):
    pheromones *= (1 - EVAPORATION_RATE)  # Evaporation
    for route, length in zip(all_routes, all_lengths):
        pheromone_deposit = Q / length
        for i in range(len(route) - 1):
            pheromones[route[i], route[i + 1]] += pheromone_deposit
            pheromones[route[i + 1], route[i]] += pheromone_deposit
        # Closing the route (return to start city)
        pheromones[route[-1], route[0]] += pheromone_deposit
        pheromones[route[0], route[-1]] += pheromone_deposit

def aco():
    best_route = None
    best_length = float('inf')

    for _ in range(ITERATIONS):
        all_routes = []
        all_lengths = []

        for _ in range(NUM_ANTS):
            start_city = np.random.randint(0, NUM_CITIES)
            route = construct_route(start_city)
            route_length = calculate_route_length(route)

            all_routes.append(route)
            all_lengths.append(route_length)

            if route_length < best_length:
                best_length = route_length
                best_route = route

        update_pheromones(pheromones, all_routes, all_lengths)

    return best_route, best_length

# Run the ACO algorithm
best_route, best_length = aco()
print("Best Route:", best_route)
print("Best Length:", best_length)
