import numpy as np

# 1. Define the Problem: The Rastrigin function
def rastrigin_function(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# 2. Initialize Parameters
n_particles = 30      # Number of particles
n_dimensions = 5      # Number of dimensions (can be changed)
max_iter = 100        # Maximum number of iterations

w = 0.5               # Inertia weight
c1 = 1.5              # Cognitive coefficient
c2 = 1.5              # Social coefficient

# 3. Initialize Particles: Random positions and velocities
positions = np.random.uniform(-5.12, 5.12, (n_particles, n_dimensions))
velocities = np.random.uniform(-1, 1, (n_particles, n_dimensions))

# Best positions found by each particle and the global best position
personal_best_positions = np.copy(positions)
personal_best_scores = np.array([rastrigin_function(p) for p in positions])
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = np.min(personal_best_scores)

# 4. Evaluate Fitness
def evaluate_fitness(positions):
    scores = np.array([rastrigin_function(p) for p in positions])
    return scores

# 5. Update Velocities and Positions
for iteration in range(max_iter):
    for i in range(n_particles):
        # Generate random numbers for cognitive and social components
        r1 = np.random.rand(n_dimensions)
        r2 = np.random.rand(n_dimensions)
        
        # Velocity update equation
        cognitive_velocity = c1 * r1 * (personal_best_positions[i] - positions[i])
        social_velocity = c2 * r2 * (global_best_position - positions[i])
        velocities[i] = w * velocities[i] + cognitive_velocity + social_velocity
        
        # Position update equation
        positions[i] += velocities[i]
        
        # Check for boundaries (constrain within a range)
        positions[i] = np.clip(positions[i], -5.12, 5.12)
    
    # Evaluate the fitness of all particles
    scores = evaluate_fitness(positions)
    
    # Update personal best positions and scores
    for i in range(n_particles):
        if scores[i] < personal_best_scores[i]:
            personal_best_scores[i] = scores[i]
            personal_best_positions[i] = positions[i]
    
    # Update global best position and score
    min_score_idx = np.argmin(personal_best_scores)
    if personal_best_scores[min_score_idx] < global_best_score:
        global_best_score = personal_best_scores[min_score_idx]
        global_best_position = personal_best_positions[min_score_idx]
    
    # Output the current best score every few iterations
    if (iteration + 1) % 10 == 0:
        print(f"Iteration {iteration + 1}/{max_iter}, Global Best Score: {global_best_score}")

# 6. Output the Best Solution
print("\nOptimization Finished!")
print(f"Global Best Position: {global_best_position}")
print(f"Global Best Score: {global_best_score}")
