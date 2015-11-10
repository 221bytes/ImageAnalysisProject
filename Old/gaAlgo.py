import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'necocityhunters'

prob_crossover = 0.60
prob_mutation = 0.05
population_size = 2
size_gene = 8
iteration_limit = 100
coefAmp = (30.0 / 255.0)
coefFreq = (0.01 / 255.0)
maximum_pixels_error = 255 * 512 * 512
nb_parameter = 3

best_fit_list = []


def random_individual():
    i = 0
    individual = []
    while i < nb_parameter:
        j = 0
        while j < size_gene:
            individual.append(random.randrange(0, 2))
            j += 1
        i += 1
    # individual = ["{0:08b}".format(random.randint(0, 255)),
    #               "{0:08b}".format(random.randint(0, 255)),
    #               "{0:08b}".format(random.randint(0, 255)), ]
    return [0, individual]


def initial_population(population_size):
    population = []
    for i in range(0, population_size):
        population.append(random_individual())
    return population


def fitness(individual):
    my_img = imgOriginal.copy()
    wrong_pixel = 0
    first_gene = ''.join(str(x) for x in individual[0:size_gene * 1])
    second_gene = ''.join(str(x) for x in individual[size_gene * 1:size_gene * 2])
    third_gene = ''.join(str(x) for x in individual[size_gene * 2:size_gene * 3])
    first_gene = int(first_gene, 2)
    noise_amp = first_gene * coefAmp
    second_gene = int(second_gene, 2)
    noise_freq_row = second_gene * coefFreq
    third_gene = int(third_gene, 2)
    noise_freq_col = third_gene * coefFreq
    # print "noiseAmp : %f noiseFreqRow : %f  noiseFreqCol : %f" % (noise_amp, noise_freq_row, noise_freq_col)

    for row in range(0, height, 1):
        for col in range(0, width, 1):
            n0 = (2.0 * np.pi * noise_freq_row * float(row))
            n1 = (2.0 * np.pi * noise_freq_col * float(col))

            n2 = n0 + n1

            sinus = np.sin(np.radians(n2))

            pxValue = noise_amp * sinus

            my_img[row, col] += pxValue
            px = int(my_img[row, col]) - int(imgCorrupted[row, col])
            if px < 0:
                px *= -1
            wrong_pixel += px

    gene_fitness = 1 - float(wrong_pixel) / float(maximum_pixels_error)
    # print "Fitness = %f" % gene_fitness
    return gene_fitness


def check_stop(fits_pop, turn):
    for i in fits_pop:
        if i[0] == 1:
            print 'victory in %d iterations the best is %s' % (turn, i)
            return True
    return False


def selection_tournament(fits_pop):
    parent_pairs = []
    for individual in range(0, population_size, 1):
        random_numbers = random.randrange(0, population_size)
        random_numbers_bis = random.randrange(0, population_size)
        if fits_pop[random_numbers][0] > fits_pop[random_numbers_bis][0]:
            parent_pairs.append([fits_pop[random_numbers][0], fits_pop[random_numbers][1]])
        else:
            parent_pairs.append([fits_pop[random_numbers_bis][0], fits_pop[random_numbers_bis][1]])
    return parent_pairs


def crossover(father, mother):
    cross_point = random.randrange(0, len(father))
    genes = 0
    kid0 = father[:]
    kid1 = mother[:]
    while genes < len(father[1]):
        if genes >= cross_point:
            kid0[1][genes] = mother[1][genes]
            kid1[1][genes] = father[1][genes]
        genes += 1
    return [kid0, kid1]


def mutation(child):
    gene = random.randrange(0, len(child))
    if child[gene] == 0:
        child[gene] = 1
    else:
        child[gene] = 0
    return child


def breed_population(fits_pop):
    parents = selection_tournament(fits_pop)
    max = 0
    for i in fits_pop:
        max += i[0]
    max = 0
    for i in parents:
        max += i[0]

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
        fits_pop_2 = []
    # for individual in next_population:
    #     fits_pop_2.append([fitness(individual), individual])
    return next_population


def print_pop(population, iteration):
    best_fitness_value = 0
    population_fitness_value = 0
    for individual in population:
        gene_fitness = fitness(individual)
        if gene_fitness > best_fitness_value:
            best_fitness_value = gene_fitness
        population_fitness_value += gene_fitness
    best_fit_list.append(best_fitness_value)
    print "population fitness = %f, best_fitness_value %f" % (population_fitness_value, best_fitness_value)
    print 'iteration = ' + str(iteration)


def run():
    population = initial_population(population_size)
    for i in range(0, iteration_limit):
        fits_pop = []
        for individual in population:
            fits_pop.append([fitness(individual[1]), individual[1]])
        if check_stop(fits_pop, i):
            break
        population = breed_population(fits_pop)
        # print_pop(population, i)


imgCorrupted = cv2.imread('lena_noisy.png', cv2.IMREAD_GRAYSCALE)
imgOriginal = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
height, width = imgOriginal.shape
totalPixel = height * width
run()

x_axis = [best_fit_list]

# for x in range(0, len(l), 1):
plt.plot(x_axis, label='iteration')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.show()
