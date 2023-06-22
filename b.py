import numpy as np
import random

TEST_SIZE = 0.2
POPULATION_SIZE = 200
MAX_GEN = 500
ELITISM = 0.2
TOURNAMENT_SIZE = 10
MUTATE_RATE = 0.3
CROSSOVER_RATE = 0.4

global TRAIN_SIZE
TRAIN_SIZE = 0

def get_strings():
    inputs = []
    labels = []
    global TRAIN_SIZE

    with open("nn0.txt") as file:
        for line in file:
            if line.strip() != "":
                string, label = line.strip().split()
                input_with_bias = np.concatenate(([1], np.array([list(letter) for letter in string], dtype=int).flatten()))
                inputs.append(input_with_bias)
                labels.append(int(label))

    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    shuffled_inputs = np.take(inputs, indices, axis=0)
    shuffled_labels = np.take(labels, indices, axis=0)

    TRAIN_SIZE = int(0.8 * len(inputs))  # 80% for training, 20% for testing

    # Split the shuffled data into training and testing sets
    input_train, input_test = shuffled_inputs[:TRAIN_SIZE], shuffled_inputs[TRAIN_SIZE:]
    labels_train, labels_test = shuffled_labels[:TRAIN_SIZE], shuffled_labels[TRAIN_SIZE:]

    # Update labels to 1 and -1
    labels_train = np.array(labels_train) * 2 - 1
    labels_test = np.array(labels_test) * 2 - 1

    return input_train, labels_train, input_test, labels_test


def init_population():
    population = []
    for i in range(POPULATION_SIZE):
        population.append(np.random.uniform(-1, 1, size=(17, 1)))
    return population


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


def calculate_fitnesses(population, X, Y):
    fitness_scores = []
    for w in population:
        hits = 0
        for i, x in enumerate(X):
            pred = np.dot(x, w)[0]
            y = Y[i]
            if pred * y > 0:
                hits += 1

        fitness_scores.append(hits / TRAIN_SIZE)

    return fitness_scores



def get_parents(population_with_fitnesses):
    tournament = random.sample(population_with_fitnesses, TOURNAMENT_SIZE)
    sorted_tournament = sorted(tournament, key=lambda x: x[1], reverse=True)[::-1]
    parent1 = sorted_tournament[0][0]
    parent2 = sorted_tournament[1][0]

    return parent1, parent2




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
    if random.uniform(0, 1) < CROSSOVER_RATE:
        genes1 = parent1
        genes2 = parent2
        split = random.randint(0, len(genes1) - 1)
        child1_genes = np.concatenate((genes1[:split], genes2[split:]))
        child2_genes = np.concatenate((genes2[:split], genes1[split:]))
        child1 = child1_genes.reshape((17, 1))
        child2 = child2_genes.reshape((17, 1))

        return child1, child2
    return parent1, parent2



def mutate(new_population):
    for index, child in enumerate(new_population):
        if random.uniform(0, 1) < MUTATE_RATE:
            i = random.randint(0, child.size - 1)
            child[i] = random.uniform(-1, 1)
            new_population[index] = child.copy()
    return new_population


def genetic_algorithm():
    input_train, labels_train, input_test, labels_test = get_strings()
    population = init_population()

    for gen in range(1):
        fitness_scores = calculate_fitnesses(population, input_train, labels_train)

        sorted_indices = np.argsort(fitness_scores)[::-1]
        tmp = np.take(population, sorted_indices, axis=0)
        sorted_population = [np.array(l) for l in tmp]
        sorted_fitness = np.take(fitness_scores, sorted_indices, axis=0)

        best = sorted_fitness[0].copy()
        if best == 1:
            print("you win!!")
            with open("wnet0.npy", 'wb') as f:
                np.save(f, sorted_population[0])
            break

        new_population = sorted_population[:round(ELITISM * POPULATION_SIZE)].copy()
        print("gen: " + str(gen) + " , hit rate: " + str(best) + " , avg: " + str(np.average(np.array(fitness_scores)))
              + " , min: " + str(min(fitness_scores)))

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = get_parents(list(zip(sorted_population, sorted_fitness))[:30])
            child1, child2 = crossover(parent1.copy(), parent2.copy())
            new_population.append(child1.copy())
            new_population.append(child2.copy())

        new_population = mutate(new_population.copy())
        population = new_population.copy()




if __name__ == "__main__":
    genetic_algorithm()