import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#define problem and fitness function
def fitness(chromosome):
    clashes = 0
    for i in range(len(chromosome)):
        for j in range(i + 1, len(chromosome)):
            if chromosome[i] == chromosome[j] or abs(chromosome[i] - chromosome[j]) == abs(i - j):
                clashes += 1
    return clashes


def generate_population(size, n):
    population = set()
    while len(population) < size:
        chromosome = tuple(random.sample(range(n), n))
        population.add(chromosome)
    return list(population)

def crossover(parent1, parent2):
    n = len(parent1)
    crossover_point = random.randint(1, n - 2)
    diagonal_indices = [i for i, (q1, q2) in enumerate(zip(parent1, parent2)) if q1 == q2]
    valid_crossover_points = [p for p in range(1, n - 1) if p not in diagonal_indices]
    
    if valid_crossover_points:
        crossover_point = random.choice(valid_crossover_points)

    child = list(parent1[:crossover_point])
    child.extend(q for q in parent2 if q not in parent1[:crossover_point])

    return tuple(child)


def mutate(chromosome, mutation_rate):
    chromosome = list(chromosome)
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            swap_index = random.randint(0, len(chromosome) - 1)
            chromosome[i], chromosome[swap_index] = chromosome[swap_index], chromosome[i]
    return tuple(chromosome)


def evolutionary_algorithm(population_size, n, generations, mutation_rate):
    population = generate_population(population_size, n)

    for generation in range(generations):
        population = sorted(population, key=lambda x: fitness(x))
        if fitness(population[0]) == 0:
            return population[0]

        new_population = population[:2]
        for _ in range(population_size - 2):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    return sorted(population, key=lambda x: fitness(x))[0]

#check if the solution is correct or not
def is_solution_valid(solution):
    n = len(solution)
    for i in range(n):
        for j in range(i + 1, n):
            # Check if queens are in the same row or on the same diagonal
            if solution[i] == solution[j] or abs(solution[i] - solution[j]) == abs(i - j):
                return False
    return True

#visualize
def plot_solution(solution):
    n = len(solution)
    board = np.zeros((n, n))

    fig, ax = plt.subplots()
    cmap = plt.cm.binary
    ax.imshow(board, cmap=cmap, origin='lower', extent=(-0.5, n - 0.5, -0.5, n - 0.5))

    # Draw the chessboard
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='white', edgecolor='none'))
            else:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='gray', edgecolor='none'))

    # Draw the queens as red circles
    for i in range(n):
        ax.add_patch(plt.Circle((solution[i], i), 0.3, color='red', zorder=2))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # ax.grid(True, which='both', color='black', linewidth=1)
    plt.show()


n_values = [8, 16]
population_size = 100
generations = 1000
mutation_rate = 0.1

for n in n_values:
    solution = evolutionary_algorithm(population_size, n, generations, mutation_rate)
    print(f"Solution for n={n}: {solution}")
    print(f"Is the solution valid? {is_solution_valid(solution)}")
    plot_solution(solution)
