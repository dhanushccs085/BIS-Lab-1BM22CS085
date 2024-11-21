import numpy as np

# Define parameters
population_size = 10
chromosome_length = 5
generations = 100
crossover_rate = 0.8
mutation_rate = 0.1

# Objective function to maximize
def objective_function(chromosome):
    return np.sum(chromosome ** 2)

# Initialize population
def initialize_population(size, length):
    return np.random.uniform(-10, 10, (size, length))

# Evaluate fitness of population
def evaluate_fitness(population):
    return np.array([objective_function(individual) for individual in population])

# Select parents using roulette-wheel selection
def select_parents(population, fitness):
    # Normalize fitness to ensure non-negative values
    fitness = fitness - fitness.min() if fitness.min() < 0 else fitness

    # Compute probabilities
    if fitness.sum() == 0:
        probabilities = np.ones(len(fitness)) / len(fitness)  # Equal probabilities
    else:
        probabilities = fitness / fitness.sum()

    # Select parents
    indices = np.random.choice(len(population), size=2, p=probabilities)
    return population[indices[0]], population[indices[1]]

# Perform crossover between two parents
def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))  # Choose crossover point
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    else:
        return parent1, parent2

# Perform mutation on an individual
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.normal(0, 1)  # Add Gaussian noise
    return individual

# Genetic algorithm
def genetic_algorithm():
    # Initialize population
    population = initialize_population(population_size, chromosome_length)

    for generation in range(generations):
        # Evaluate fitness
        fitness = evaluate_fitness(population)

        # Print best fitness of the generation
        print(f"Generation {generation + 1}: Best Fitness = {fitness.max()}")

        # Create a new population
        new_population = []

        for _ in range(population_size // 2):
            # Select parents
            parent1, parent2 = select_parents(population, fitness)

            # Crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate)

            # Mutation
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            # Add children to the new population
            new_population.extend([child1, child2])

        # Replace the old population with the new one
        population = np.array(new_population)

    # Evaluate final population
    final_fitness = evaluate_fitness(population)
    best_individual = population[np.argmax(final_fitness)]

    print(f"\nBest Individual: {best_individual}")
    print(f"Best Fitness: {final_fitness.max()}")

# Run the genetic algorithm
genetic_algorithm()


# Final best solution
fitness = evaluate_fitness(population)
best_individual = population[np.argmax(fitness)]
best_fitness = np.max(fitness)
print(f"\nBest Solution: x = {best_individual:.4f}, Fitness = {best_fitness:.4f}")
