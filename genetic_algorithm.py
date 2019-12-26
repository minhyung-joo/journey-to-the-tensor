import numpy as np
import math

POPULATION_SIZE = 100
CHROMOSOME_LENGTH = 10
GENES = [chr(x) for x in range(ord('a'), ord('z') + 1)]
TARGET = "balenciaga"
TARGET_CHROMOSOME = [c for c in TARGET]

def generate_genome():
    return [np.random.choice(GENES) for _ in range(CHROMOSOME_LENGTH)]

def mate(chromosome1, chromosome2):
    new_chromosome = []
    for i in range(len(chromosome1)):
        randval = np.random.random()
        if randval < 0.45:
            new_chromosome.append(chromosome1[i])
        elif randval < 0.9:
            new_chromosome.append(chromosome2[i])
        else:
            new_chromosome.append(np.random.choice(GENES))
    
    return new_chromosome

def fitness(chromosome):
    score = 0
    for i in range(len(TARGET)):
        if chromosome[i] != TARGET[i]:
            score = score + 1
    
    return score

generation = 1
population = [generate_genome() for _ in range(POPULATION_SIZE)]
while True:
    new_population = []
    population = sorted(population, key=fitness)
    if population[0] == TARGET_CHROMOSOME:
        break

    new_population.extend(population[:10])
    for _ in range(90):
        father = population[math.floor(np.random.random() * 50)]
        mother = population[math.floor(np.random.random() * 50)]
        new_population.append(mate(father, mother))
    
    population = new_population
    print("Generation: {}\tString: {}\tFitness: {}".format(generation, population[0], fitness(population[0])))
    generation = generation + 1

print("Generation: {}\tString: {}\tFitness: {}".format(generation, population[0], fitness(population[0])))