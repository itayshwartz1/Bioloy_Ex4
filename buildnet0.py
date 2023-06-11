import numpy as np
import random



TEST_SIZE = 0.2
POPULATION_SIZE = 100
MAX_GEN = 200
ELITISM = 0.2
TOURNAMENT_SIZE = 10
MUTATE_RATE = 0.5
def get_strings():
    inputs = []
    labels = []

    with open("nn0.txt") as file:
        for line in file:
            if line.strip() != "":
                string, label = line.strip().split()
                inputs.append(np.array([list(letter) for letter in string], dtype=int).reshape(1, -1))
                labels.append(int(label))

    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    shuffled_inputs = np.take(inputs, indices, axis=0)
    shuffled_labels = np.take(labels, indices, axis=0)

    split_point = int(0.8 * len(inputs))  # 80% for training, 20% for testing

    # Split the shuffled data into training and testing sets
    input_train, input_test = shuffled_inputs[:split_point], shuffled_inputs[split_point:]
    labels_train, labels_test = shuffled_labels[:split_point], shuffled_labels[split_point:]

    return input_train, labels_train, input_test, labels_test


def init_population():
    population = []
    for i in range(POPULATION_SIZE):

        population.append([np.random.uniform(-1, 1, size=(16, 64)),
                           np.random.uniform(-1, 1, size=(64, 32)),
                           np.random.uniform(-1, 1, size=(32, 1))])
    return population


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_fitnesses(population, input, labels):
    fitness_scores = []
    labels = np.squeeze(labels)
    for W in population:
        tmp = input
        for layer in W:
            z = np.dot(tmp, layer)
            a = sigmoid(z)
            tmp = a

        tmp = np.squeeze(tmp)
        fitness_scores.append(np.dot(tmp - labels, tmp - labels))

    return fitness_scores


def get_parents(population_with_fitnesses):
    tournament = random.sample(population_with_fitnesses, TOURNAMENT_SIZE)
    parent1 = min(tournament, key=lambda x: x[1])
    tournament.remove(parent1)
    parent2 = min(tournament, key=lambda x: x[1])

    return parent1[0], parent2[0]


def unflatten(child1_genes):
    shape1 = (16, 64)
    shape2 = (64, 32)
    shape3 = (32, 1)

    split1 = shape1[0] * shape1[1]
    split2 = split1 + shape2[0] * shape2[1]

    genes1 = child1_genes[:split1].reshape(shape1)
    genes2 = child1_genes[split1:split2].reshape(shape2)
    genes3 = child1_genes[split2:].reshape(shape3)

    return [genes1, genes2, genes3]

def crossover(parent1, parent2):
    genes1 = np.concatenate([a.flatten() for a in parent1])
    genes2 = np.concatenate([a.flatten() for a in parent2])
    split = random.randint(0, len(genes1) - 1)
    child1_genes = np.asarray(genes1[:split].tolist() + genes2[split:].tolist())
    child2_genes = np.asarray(genes2[:split].tolist() + genes1[split:].tolist())
    child1 = unflatten(child1_genes)
    child2 = unflatten(child2_genes)

    return child1, child2


def mutate(new_population):
    for index, child in enumerate(new_population):
        if random.uniform(0, 1) < MUTATE_RATE:
            for _ in range(30):
                m = random.randint(0, len(child) - 1)
                i = random.randint(0, len(child[m]) - 1)
                j = random.randint(0, len(child[m][0]) - 1)
                child[m][i][j] = random.uniform(-1, 1)
            new_population[index] = child

    return new_population


def genetic_algorithm():
    input_train, labels_train, input_test, labels_test = get_strings()
    population = init_population()

    for gen in range(MAX_GEN):
        fitness_scores = calculate_fitnesses(population, input_train, labels_train)
        sorted_indices = np.argsort(fitness_scores)
        sorted_population = np.take(population, sorted_indices, axis=0)
        sorted_fitness = np.take(fitness_scores, sorted_indices, axis=0)

        if sorted_fitness[0] == 0:
            print("you win!!")
            break

        new_population = sorted_population[:round(ELITISM * POPULATION_SIZE)].tolist()
        print(min(sorted_fitness))
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = get_parents(list(zip(sorted_population, sorted_fitness)))
            child1, child2 = crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)

        new_population = mutate(new_population)
        population = new_population



if __name__ == "__main__":
    genetic_algorithm()
