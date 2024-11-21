import numpy as np
import math

# Step 1: Define the objective function to optimize
# Example: A simple sphere function (minimize the sum of squares)
def objective_function(x):
    return np.sum(x ** 2)

# Step 2: Initialize parameters
n_nests = 20  # number of nests
n_iterations = 100  # number of iterations
pa = 0.25  # probability of discovery
dim = 5  # dimensionality of the problem (number of variables)
lower_bound = -5  # lower bound of the search space
upper_bound = 5  # upper bound of the search space

# Step 3: Initialize population of nests with random positions
def initialize_population(n_nests, dim, lower_bound, upper_bound):
    nests = np.random.uniform(lower_bound, upper_bound, (n_nests, dim))
    return nests

# Step 4: Evaluate fitness
def evaluate_fitness(nests, objective_function):
    fitness = np.apply_along_axis(objective_function, 1, nests)
    return fitness

# Step 5: Lévy flight step (for generating new solutions)
def levy_flight(Lambda, dim):
    # Generate Lévy flights using the formula
    beta = 1.5
    sigma = (math.gamma(1 + beta) * np.sin(math.pi * beta / 2) / 
             (math.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2))) ** (1 / beta)
    s = np.random.normal(0, sigma, dim)  # Standard normal
    t = np.random.normal(0, 1, dim)  # Standard normal
    flight = s / np.power(np.abs(t), 1 / beta)  # Lévy flight
    return flight

# Step 6: Abandon worst nests and replace with new random positions
def abandon_worst_nests(nests, fitness, pa, lower_bound, upper_bound):
    # Sort nests by their fitness values (ascending order, best first)
    sorted_indices = np.argsort(fitness)
    best_nests = nests[sorted_indices[:int(pa * len(fitness))]]  # Keep the best nests
    new_nests = np.random.uniform(lower_bound, upper_bound, (len(fitness) - len(best_nests), nests.shape[1]))
    return np.vstack([best_nests, new_nests])

# Step 7: Cuckoo Search main loop
def cuckoo_search(n_nests, n_iterations, lower_bound, upper_bound, objective_function, pa=0.25, lambda_=1.5):
    # Step 3: Initialize population
    nests = initialize_population(n_nests, dim, lower_bound, upper_bound)
    fitness = evaluate_fitness(nests, objective_function)
    
    # Initialize best solution
    best_solution = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    # Main loop
    for iteration in range(n_iterations):
        # Step 5: Generate new solutions by Lévy flights
        for i in range(n_nests):
            new_nest = nests[i] + levy_flight(lambda_, dim)
            # Bound check
            new_nest = np.clip(new_nest, lower_bound, upper_bound)
            # Evaluate the new solution
            new_fitness = objective_function(new_nest)
            # Replace nest if new solution is better
            if new_fitness < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness
        
        # Step 6: Abandon worst nests
        nests = abandon_worst_nests(nests, fitness, pa, lower_bound, upper_bound)
        fitness = evaluate_fitness(nests, objective_function)
        
        # Update the best solution
        current_best_solution = nests[np.argmin(fitness)]
        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best_solution = current_best_solution
            best_fitness = current_best_fitness
        
        # Print progress
        print(f"Iteration {iteration + 1}: Best fitness = {best_fitness}")
    
    return best_solution, best_fitness

# Run the Cuckoo Search algorithm
best_solution, best_fitness = cuckoo_search(n_nests, n_iterations, lower_bound, upper_bound, objective_function)
print("Best solution found:", best_solution)
print("Best fitness:", best_fitness)
