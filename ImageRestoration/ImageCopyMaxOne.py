import random
import numpy as np
import cv2

__author__ = 'Alexandre Catalano'

prob_crossover = 0.6
prob_mutation = 0.05
population_size = 100
size_gene = 8
iteration_limit = 100000
coef_amp = (30.0 / 255.0)
coef_freq = (0.01 / 255.0)
maximum_pixels_error = 255 * 512 * 512
nb_parameter = 3

max_one = 3 * 8


def total_fitness(pop):
    total = 0
    for i in pop:
        for j in i:
            total += j
    return total


def random_individual(n):
    genes = []
    for i in range(0, n, 1):
        genes.append(str(random.randrange(0, 2)))
    return genes


def initial_population(n):
    population = []
    for i in range(0, n, 1):
        population.append(random_individual(max_one))
    return population


def gene_to_noise_params(individual):
    first_gene = individual[:8]
    second_gene = individual[8:16]
    third_gene = individual[16:]
    first_gene = int(''.join(first_gene), 2)
    second_gene = int(''.join(second_gene), 2)
    third_gene = int(''.join(third_gene), 2)
    noise_amp = first_gene * coef_amp
    noise_freq_row = second_gene * coef_freq
    noise_freq_col = third_gene * coef_freq

    return noise_amp, noise_freq_row, noise_freq_col


def make_noise(params):
    NoiseAmp, NoiseFreqRow, NoiseFreqCol = params
    h, w = imgOriginal.shape
    y = np.arange(h)
    x = np.arange(w)
    col, row = np.meshgrid(x, y, sparse=True)
    noise = NoiseAmp * np.sin(2*np.pi * NoiseFreqRow * row + 2*np.pi * NoiseFreqCol * col)
    return noise


def corrupt_image(noise_params):
    noise = make_noise(noise_params)
    signal_noisy = imgOriginal + noise
    return noise, signal_noisy

dm = 255*512*512

def lena_fitness(gene):
    noise_params = gene_to_noise_params(gene)
    noise, lena_noisy = corrupt_image(noise_params)
    noisy_diff = imgCorrupted - lena_noisy
    noisy_diff = np.sum(np.abs(noisy_diff)) #/ lena_N  # normalized
    # gene_fitness = lena_N - noisy_diff  # negative / minimize
    gene_fitness = 1 - (noisy_diff / dm)
    return gene_fitness


def fitness(individual):

    return lena_fitness(individual)


def check_stop(fits_pop, turn):
    for i in fits_pop:
        if i[0] == max_one:
            print 'victory in %d iterations the best is %s' % (turn, i)
            return True
    return False


def select(fits_pop):
    fitness_population = 0
    parent_pairs = []
    for i in fits_pop:
        fitness_population += i[0]
    for individual in range(0, population_size, 1):
        random_numbers = random.randrange(0, population_size)
        random_numbers_bis = random_numbers
        while random_numbers_bis == random_numbers:
            random_numbers_bis = random.randrange(0, population_size)
        if fits_pop[random_numbers][0] > fits_pop[random_numbers_bis][0]:
            parent_pairs.append(fits_pop[random_numbers][1])
        else:
            parent_pairs.append(fits_pop[random_numbers_bis][1])
    return parent_pairs


def crossover(father, mother):
    size = len(father)
    cross_point0 = random.randrange(0, size - 1)
    cross_point1 = random.randrange(cross_point0, size)
    kid0 = father[:]
    kid1 = mother[:]
    kid0[cross_point0: cross_point1] = mother[cross_point0: cross_point1]
    kid1[cross_point0: cross_point1] = father[cross_point0: cross_point1]
    return [kid0, kid1]


def mutation(child):
    gene = random.randrange(0, len(child))
    if child[gene] == '0':
        child[gene] = '1'
    else:
        child[gene] = '0'
    return child


def breed_population(fits_pop):
    # parents = selection(fits_pop)
    parents = select(fits_pop)
    next_population = []
    father = 0
    while father < population_size:
        mother = father + 1
        cross = random.uniform(0, 1) < prob_crossover
        if cross is True:
            children = crossover(parents[father], parents[mother])
        else:
            children = [parents[father], parents[mother]]
        for child in children:
            mutate = random.uniform(0, 1) < prob_mutation
            if mutate is True:
                child = mutation(child)
            next_population.append(child)
        father += 2
    return next_population


def print_pop(population, turn):
    population_fitness_value = 0.0
    best_fitness_value = 0.0
    best_gene = []
    for k in population:
        fitness_value = k[0]
        if fitness_value > best_fitness_value:
            best_fitness_value = fitness_value
            best_gene = gene_to_noise_params(k[1])
        population_fitness_value += fitness_value
    print "iteration : ", turn
    print "population fitness = %f, best_fitness_value %f" % (population_fitness_value, best_fitness_value)
    print best_fitness_value, best_gene


def run():
    population = initial_population(population_size)
    iteration = 0
    for i in range(0, iteration_limit, 1):
        fits_pop = []
        for individual in population:
            fits_pop.append([fitness(individual), individual])
        if check_stop(fits_pop, i):
            break
        population = breed_population(fits_pop)
        print_pop(fits_pop, i)


imgCorrupted = cv2.imread('lena_noisy.png', cv2.IMREAD_GRAYSCALE).astype(np.dtype)
imgOriginal = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE).astype(np.dtype)
lena_N = np.prod(imgOriginal.shape[:2])
noise_target = imgCorrupted - imgOriginal
height, width = imgOriginal.shape
totalPixel = height * width
run()
