
import numpy as np
import random

distance = np.array([
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 3],
    [9, 6, 0, 8, 5],
    [10, 4, 8, 0, 6],
    [7, 3, 5, 6, 0]
], dtype=float)

n_cities = len(distance)
n_ants = 10
n_iterations = 100
alpha = 1.0
beta = 2.0
rho = 0.5
Q = 100.0

pheromone = np.ones((n_cities, n_cities))
heuristic = np.zeros_like(distance)
heuristic[distance > 0] = 1.0 / distance[distance > 0]


def construct_tour():
    start = random.randrange(n_cities)
    tour = [start]
    unvisited = set(range(n_cities))
    unvisited.remove(start)

    while unvisited:
        current = tour[-1]
        candidates = list(unvisited)

        weights = np.array([
            pheromone[current, city] ** alpha *
            heuristic[current, city] ** beta
            for city in candidates
        ])

        probabilities = weights / weights.sum()
        next_city = random.choices(
            candidates, weights=probabilities, k=1
        )[0]

        tour.append(next_city)
        unvisited.remove(next_city)

    return tour


def tour_length(tour):
    return sum(
        distance[tour[i], tour[(i + 1) % n_cities]]
        for i in range(n_cities)
    )


best_tour = None
best_length = float("inf")

for iteration in range(n_iterations):
    tours = []
    lengths = []

    for ant in range(n_ants):
        tour = construct_tour()
        length = tour_length(tour)

        tours.append(tour)
        lengths.append(length)

        if length < best_length:
            best_length = length
            best_tour = tour.copy()

    pheromone *= (1 - rho)

    for tour, length in zip(tours, lengths):
        deposit = Q / length

        for i in range(n_cities):
            a = tour[i]
            b = tour[(i + 1) % n_cities]
            pheromone[a, b] += deposit
            pheromone[b, a] += deposit

    if (iteration + 1) % 10 == 0:
        print(f"Iteration {iteration + 1}: Best distance = {best_length:.2f}")

print("\nBest tour found:")
print(" -> ".join(map(str, best_tour + [best_tour[0]])))
print("Shortest distance:", best_length)
