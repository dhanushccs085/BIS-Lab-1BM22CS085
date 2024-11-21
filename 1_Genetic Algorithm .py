import numpy as np
import random

def fitness_function(x):
    """The mathematical function to optimize."""
    return x * np.sin(x) + x * np.cos(2 * x)

def initialize_population(pop_size, lower_bound, upper_bound):
    """Create an initial population within the specified bounds."""
    return np.random.uniform(lower_bound, upper_bound, pop_size)

def evaluate_fitness(population):
    """Evaluate the fitness of each individual."""
    return np.array([fitness_function(ind) for ind in population])

def select_parents(population, fitness):
    """Select individuals for reproduction using roulette-wheel selection."""
    probabilities = fitness / fitness.sum()
    return population[np.random.choice(len(population), size=2, p=probabilities)]

def crossover(parent1, parent2, crossover_rate):
    """Perform crossover between two parents."""
    if random.random() < crossover_rate:
        point = random.randint(0, len(parent1))
        return (parent1[:point] + parent2[point:], parent2[:point] + parent1[point:])
    return parent1, parent2

def mutate(individual, mutation_rate, lower_bound, upper_bound):
    """Apply mutation to an individual."""
    if random.random() < mutation_rate:
        individual += np.random.uniform(-1, 1)
        individual = np.clip(individual, lower_bound, upper_bound)
    return individual

# Parameters
population_size = 20
num_generations = 100
mutation_rate = 0.1
crossover_rate = 0.7
lower_bound, upper_bound = -10, 10

# Initialize population
population = initialize_population(population_size, lower_bound, upper_bound)

# Evolution process
for generation in range(num_generations):
    # Evaluate fitness
    fitness = evaluate_fitness(population)

    # Track the best solution
    best_individual = population[np.argmax(fitness)]
    best_fitness = np.max(fitness)

    # Print progress
    print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.4f}")

    # Create new population
    new_population = []
    for _ in range(population_size // 2):
        # Select parents
        parent1, parent2 = select_parents(population, fitness)

        # Crossover
        offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)

        # Mutation
        offspring1 = mutate(offspring1, mutation_rate, lower_bound, upper_bound)
        offspring2 = mutate(offspring2, mutation_rate, lower_bound, upper_bound)

        # Add to new population
        new_population.extend([offspring1, offspring2])

    population = np.array(new_population)

# Final best solution
fitness = evaluate_fitness(population)
best_individual = population[np.argmax(fitness)]
best_fitness = np.max(fitness)
print(f"\nBest Solution: x = {best_individual:.4f}, Fitness = {best_fitness:.4f}")
