import numpy as np
import random

TEST_SIZE = 0.2
POPULATION_SIZE = 100
MAX_GEN = 500
ELITISM = 0.1
TOURNAMENT_SIZE = 10
MUTATE_RATE_CHILD = 0.4
MUTATE_RATE_ELITISM = 0.3
CROSSOVER_RATE = 0.5
HIDDEN_LAYERS = [16, 64, 1]
global TRAIN_SIZE
TRAIN_SIZE = 0


def get_strings():
    global TRAIN_SIZE
    inputs = []
    labels = []

    with open("nn0.txt") as file:
        for line in file:
            if line.strip() != "":
                string, label = line.strip().split()
                inputs.append(np.array([list(letter) for letter in string], dtype=int).flatten())
                if label == '1':
                    labels.append(1)
                else:
                    labels.append(-1)

    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    shuffled_inputs = np.take(inputs, indices, axis=0)
    shuffled_labels = np.take(labels, indices, axis=0)
    TRAIN_SIZE = int(0.8 * len(inputs))  # 80% for training, 20% for testing

    # Split the shuffled data into training and testing sets
    input_train, input_test = shuffled_inputs[:TRAIN_SIZE], shuffled_inputs[TRAIN_SIZE:]
    labels_train, labels_test = shuffled_labels[:TRAIN_SIZE], shuffled_labels[TRAIN_SIZE:]

    return input_train, labels_train, input_test, labels_test


def init_population():
    population = []
    for i in range(POPULATION_SIZE):
        child = []
        for j in range(len(HIDDEN_LAYERS) - 1):
            b = np.ones(HIDDEN_LAYERS[j + 1])
            w = np.random.uniform(-1, 1, size=(HIDDEN_LAYERS[j], HIDDEN_LAYERS[j + 1]))
            b_and_w = (b, w)
            child.append(b_and_w)

        population.append(child)

    return population


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_fitnesses(population, X, Y):
    fitness_scores = []
    for child in population:
        hits = 0
        for i, x in enumerate(X):
            prev = x
            for b_and_w in child:
                b = b_and_w[0]
                w = b_and_w[1]
                z = np.dot(prev, w) + b
                a = sigmoid(z)
                prev = a
            prev = prev[0]
            if prev >= 0.5:
                prev = 1
            else:
                prev = -1
            if prev * Y[i] > 0:
                hits += 1
        fitness_scores.append(hits / TRAIN_SIZE)

    return fitness_scores


def get_parents(sorted_population, sorted_fitness):
    parents = random.choices(sorted_population, sorted_fitness, k=2)
    return parents[0], parents[1]
    # tournament = random.sample(population_with_fitnesses, TOURNAMENT_SIZE)
    # sorted_tournament = sorted(tournament, key=lambda x: x[1], reverse=True)[::-1]
    # parent1 = sorted_tournament[0][0]
    # parent2 = sorted_tournament[1][0]
    # return parent1, parent2


# def unflatten(child1_genes):
#     shape1 = (16, 64)
#     shape2 = (64, 32)
#     shape3 = (32, 1)
#
#     split1 = shape1[0] * shape1[1]
#     split2 = split1 + shape2[0] * shape2[1]
#
#     genes1 = child1_genes[:split1].reshape(shape1)
#     genes2 = child1_genes[split1:split2].reshape(shape2)
#     genes3 = child1_genes[split2:].reshape(shape3)
#
#     return [genes1, genes2, genes3]


def crossover(parent1, parent2):
    if random.uniform(0, 1) > CROSSOVER_RATE:
        return parent1.tolist(), parent2.tolist()

    child1 = []
    child2 = []

    size = len(parent1)

    for i in range(size):
        b1 = parent1[i][0]
        b2 = parent2[i][0]
        w1 = parent1[i][1]
        w2 = parent2[i][1]

        split_w = random.randint(0, w1.shape[0])

        # child 1
        new_b1 = b1
        new_w1 = np.concatenate((w1[split_w:], w2[:split_w]), axis=None).reshape(w1.shape[0], w1.shape[1])
        b_and_w1 = (new_b1, new_w1)
        child1.append(b_and_w1)

        # child 2
        new_b2 = b2
        new_w2 = np.concatenate((w1[:split_w], w2[split_w:]), axis=None).reshape(w1.shape[0], w1.shape[1])
        b_and_w2 = (new_b2, new_w2)
        child2.append(b_and_w2)

    return child1, child2



    #
    #
    # genes1 = np.concatenate([a.flatten() for a in parent1])
    # genes2 = np.concatenate([a.flatten() for a in parent2])
    # split = random.randint(0, len(genes1) - 1)
    # child1_genes = np.asarray(genes1[:split].tolist() + genes2[split:].tolist())
    # child2_genes = np.asarray(genes2[:split].tolist() + genes1[split:].tolist())
    # child1 = unflatten(child1_genes)
    # child2 = unflatten(child2_genes)
    #
    # return child1, child2


def mutate(child, mutate_rate):
    new_child = []
    if random.uniform(0, 1) < mutate_rate:
        for b_and_w in child:
            b = b_and_w[0]
            w = b_and_w[1]
            rows = w.shape[0]
            cols = w.shape[1]
            for row in range(rows):
                for col in range(cols):
                    r = random.uniform(0, 1)
                    if r < 0.25:
                        w[row][col] = 0
                    elif r < 0.5:
                        w[row][col] *= (w[row][col] * (-1))  # opposite
                    elif r < 0.75:
                        w[row][col] += 0.1  # increase a little
                    else:
                        w[row][col] -= 0.1  # decrease a little
            r = random.uniform(-1 + b[0], 1 + b[0])
            b = np.array([r] * b.shape[0])
            new_child.append((b, w))
    else:
        return child
    return new_child


def genetic_algorithm():
    input_train, labels_train, input_test, labels_test = get_strings()
    population = init_population()

    for gen in range(MAX_GEN):
        fitness_scores = calculate_fitnesses(population, input_train, labels_train)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = np.take(population, sorted_indices, axis=0)
        sorted_fitness = np.take(fitness_scores, sorted_indices, axis=0)

        if sorted_fitness[0] == 1:
            print("you win!!")
            break

        tmp_population = sorted_population[:round(ELITISM * POPULATION_SIZE)].tolist().copy()
        best_child = tmp_population[0].copy()
        new_population = [best_child.copy()]
        for child in tmp_population:
            new_population.append(mutate(child.copy(), MUTATE_RATE_ELITISM))


        print(max(sorted_fitness))
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = get_parents(sorted_population.copy(), sorted_fitness.copy())
            child1, child2 = crossover(parent1.copy(), parent2.copy())
            new_population.append(mutate(child1.copy(), MUTATE_RATE_CHILD))
            new_population.append(mutate(child2.copy(), MUTATE_RATE_CHILD))
        population = new_population.copy()


if __name__ == "__main__":
    genetic_algorithm()
