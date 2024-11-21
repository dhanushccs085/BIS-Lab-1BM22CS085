import random
import numpy as np
import matplotlib.pyplot as plt

class AntColonyOptimization:
    def __init__(self, cities, n_ants, n_iterations, alpha, beta, rho, q0):
        self.cities = cities
        self.n_cities = len(cities)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha   # pheromone importance
        self.beta = beta     # heuristic information importance
        self.rho = rho       # pheromone evaporation rate
        self.q0 = q0         # probability for exploration vs exploitation
        self.pheromone = np.ones((self.n_cities, self.n_cities))  # pheromone initialization
        self.distances = self.compute_distances()

    def compute_distances(self):
        distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                distance = np.linalg.norm(self.cities[i] - self.cities[j])
                distances[i][j] = distances[j][i] = distance
        return distances

    def select_next_city(self, current_city, visited):
        probabilities = np.zeros(self.n_cities)
        tau = self.pheromone[current_city]
        eta = 1.0 / (self.distances[current_city] + 1e-10)
        for i in range(self.n_cities):
            if i not in visited:
                probabilities[i] = (tau[i] ** self.alpha) * (eta[i] ** self.beta)

        probabilities_sum = probabilities.sum()
        probabilities /= probabilities_sum  # normalize probabilities

        # Choose next city based on the exploration-exploitation balance
        if random.random() < self.q0:
            # Exploitation: choose the best next city (highest probability)
            next_city = np.argmax(probabilities)
        else:
            # Exploration: probabilistic choice based on pheromone and heuristic info
            next_city = np.random.choice(range(self.n_cities), p=probabilities)

        return next_city

    def construct_solution(self):
        # Start with a random city and construct a tour
        visited = [random.randint(0, self.n_cities - 1)]
        tour = visited[:]
        while len(visited) < self.n_cities:
            current_city = visited[-1]
            next_city = self.select_next_city(current_city, visited)
            visited.append(next_city)
            tour.append(next_city)
        # Return to the starting city to complete the cycle
        tour.append(tour[0])
        return tour

    def update_pheromones(self, ants_solutions, ants_lengths):
        # Evaporate pheromones
        self.pheromone *= (1 - self.rho)

        # Deposit new pheromones based on the ants' solutions
        for i in range(self.n_ants):
            solution = ants_solutions[i]
            length = ants_lengths[i]
            for j in range(self.n_cities):
                from_city = solution[j]
                to_city = solution[j + 1]
                self.pheromone[from_city][to_city] += 1.0 / length
                self.pheromone[to_city][from_city] += 1.0 / length  # pheromone is symmetric

    def optimize(self):
        best_solution = None
        best_length = float('inf')
        all_lengths = []

        for iteration in range(self.n_iterations):
            ants_solutions = []
            ants_lengths = []

            # Step 3: Construct solutions
            for _ in range(self.n_ants):
                solution = self.construct_solution()
                length = self.calculate_total_length(solution)
                ants_solutions.append(solution)
                ants_lengths.append(length)

                # Update the best solution found so far
                if length < best_length:
                    best_solution = solution
                    best_length = length

            # Step 4: Update pheromones
            self.update_pheromones(ants_solutions, ants_lengths)

            all_lengths.append(best_length)
            print(f"Iteration {iteration + 1}, Best Length: {best_length}")

        return best_solution, best_length, all_lengths

    def calculate_total_length(self, solution):
        total_length = 0
        for i in range(self.n_cities):
            from_city = solution[i]
            to_city = solution[i + 1]
            total_length += self.distances[from_city][to_city]
        return total_length

    def plot_solution(self, solution):
        # Visualize the solution
        x = [self.cities[city][0] for city in solution]
        y = [self.cities[city][1] for city in solution]
        plt.plot(x, y, marker='o')
        plt.plot([x[0], x[-1]], [y[0], y[-1]], marker='o', linestyle="--", color='r')  # return to start
        plt.title(f"Best Tour Length: {self.calculate_total_length(solution):.2f}")
        plt.show()


# Define the cities (as an example, you can change the coordinates)
cities = np.array([
    [0, 0],
    [1, 3],
    [3, 1],
    [5, 3],
    [6, 6],
    [8, 3],
    [9, 0],
    [7, -2]
])

# Parameters for the ACO algorithm
n_ants = 10
n_iterations = 100
alpha = 1.0  # pheromone influence
beta = 2.0   # distance heuristic influence
rho = 0.5    # pheromone evaporation rate
q0 = 0.9     # exploration vs exploitation

# Initialize and run the ACO algorithm
aco = AntColonyOptimization(cities, n_ants, n_iterations, alpha, beta, rho, q0)
best_solution, best_length, all_lengths = aco.optimize()

# Output the best solution
print(f"Best Solution: {best_solution}")
print(f"Best Length: {best_length}")

# Plot the best solution
aco.plot_solution(best_solution)
