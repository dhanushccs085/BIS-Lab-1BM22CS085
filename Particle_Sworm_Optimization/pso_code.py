import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

# 1. Load Dataset
data = load_iris()
X = data.data
y = data.target

# 2. Define the fitness function for PSO
def svm_fitness(params):
    C, gamma = params
    model = SVC(C=C, gamma=gamma)
    accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    return -accuracy  # Negative for minimization in PSO

# 3. Particle Swarm Optimization (PSO) from scratch
class Particle:
    def __init__(self, lb, ub):
        self.position = np.random.uniform(lb, ub)
        self.velocity = np.random.uniform(-1, 1, len(lb))
        self.best_position = self.position.copy()
        self.best_value = float('inf')

    def update_velocity(self, global_best_position, w, c1, c2):
        inertia = w * self.velocity
        cognitive = c1 * np.random.random() * (self.best_position - self.position)
        social = c2 * np.random.random() * (global_best_position - self.position)
        self.velocity = inertia + cognitive + social

    def update_position(self, lb, ub):
        self.position += self.velocity
        self.position = np.clip(self.position, lb, ub)

# 4. PSO main optimization loop
def pso_optimization(n_particles, n_iterations, lb, ub, w, c1, c2):
    particles = [Particle(lb, ub) for _ in range(n_particles)]
    global_best_position = None
    global_best_value = float('inf')

    for iteration in range(n_iterations):
        for particle in particles:
            fitness = svm_fitness(particle.position)

            if fitness < particle.best_value:
                particle.best_value = fitness
                particle.best_position = particle.position.copy()

            if fitness < global_best_value:
                global_best_value = fitness
                global_best_position = particle.position.copy()

        for particle in particles:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position(lb, ub)

        print(f"Iteration {iteration + 1}, Best Fitness: {-global_best_value}")

    return global_best_position, -global_best_value

# 5. Define bounds and PSO parameters
lb = [0.1, 0.001]  # Lower bounds for C and gamma
ub = [100, 1]      # Upper bounds for C and gamma
w = 0.5            # Inertia weight
c1 = 1.5           # Cognitive parameter
c2 = 1.5           # Social parameter

# Run PSO to find the best hyperparameters for SVM
best_params, best_accuracy = pso_optimization(n_particles=10, n_iterations=20, lb=lb, ub=ub, w=w, c1=c1, c2=c2)

# Output the results
print(f"Best C: {best_params[0]}")
print(f"Best gamma: {best_params[1]}")
print(f"Best cross-validation accuracy: {best_accuracy}")
