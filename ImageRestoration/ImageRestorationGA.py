import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'necocityhunters'

prob_crossover = 0.6
prob_mutation = 0.05
population_size = 10
size_gene = 8
iteration_limit = 100
coef_amp = (30.0 / 255.0)
coef_freq = (0.01 / 255.0)
maximum_pixels_error = 255 * 512 * 512
nb_parameter = 3


def random_individual():
    i = 0
    individual = []
    while i < nb_parameter:
        j = 0
        l = []
        while j < size_gene:
            l.append(str(random.randrange(0, 2)))
            j += 1
        individual.append(l)
        i += 1
    return [0, individual]


def initial_population():
    population = []
    for i in range(0, population_size):
        population.append(random_individual())
    return population


def check_stop(fits_pop, turn):
    for i in fits_pop:
        if i[0] == 1:
            print 'victory in %d iterations the best is %s' % (turn, i)
            return True
    return False


def fitness(individual):
    my_img = imgOriginal.copy()
    error_rate = 0
    first_gene = individual[1][0]
    second_gene = individual[1][1]
    third_gene = individual[1][2]
    first_gene = int(''.join(first_gene), 2)
    noise_amp = first_gene * coef_amp
    second_gene = int(''.join(second_gene), 2)
    noise_freq_row = second_gene * coef_freq
    third_gene = int(''.join(third_gene), 2)
    noise_freq_col = third_gene * coef_freq
    # print "noiseAmp : %f noiseFreqRow : %f  noiseFreqCol : %f" % (noise_amp, noise_freq_row, noise_freq_col)
    # print "first_gene: %d second_gene : %d  third_gene: %d" % (first_gene, second_gene, third_gene)

    for row in range(0, height, 1):
        for col in range(0, width, 1):
            n0 = (2.0 * np.pi * noise_freq_row * float(row + 1))
            n1 = (2.0 * np.pi * noise_freq_col * float(col + 1))

            n2 = n0 + n1

            sinus = np.sin(np.radians(n2))

            px_value = noise_amp * sinus

            my_img[row, col] += px_value
            px = int(my_img[row, col]) - int(imgCorrupted[row, col])
            if px < 0:
                px *= -1
            error_rate += px

    gene_fitness = 1 - float(error_rate) / float(maximum_pixels_error)
    individual[0] = gene_fitness
    return individual


def rank_selection(fits_pop):
    pop_tmp = fits_pop[:]
    ranked_pop = []
    parents = []
    while len(pop_tmp) > 0:
        tmp = [1]
        for i in pop_tmp:
            if i[0] <= tmp[0]:
                tmp = i
        pop_tmp.remove(tmp)
        ranked_pop.append(tmp)
    i = 0
    total = 0
    while i < len(fits_pop):
        total += i + 1
        i += 1
    for individual in range(0, len(fits_pop)):
        random_number = random.randrange(0, total)
        tmp = 0
        i = 0
        while i < len(ranked_pop):
            tmp += i + 1
            if tmp >= random_number:
                parents.append(ranked_pop[i])
                break
            i += 1
    return parents


def selection_roulette(fits_pop):
    parents = []
    pop_tmp = fits_pop[:]
    fitness_population = 0
    for i in fits_pop:
        fitness_population += i[0]
    ranked_pop = []
    while len(pop_tmp) > 0:
        tmp = [0]
        for indi in pop_tmp:
            if indi[0] >= tmp[0]:
                tmp = indi
        pop_tmp.remove(tmp)
        ranked_pop.append(tmp)
    for individual in range(0, population_size, 1):
        random_numbers = random.uniform(0, fitness_population)
        tmp = 0
        for i in fits_pop:
            tmp += i[0]
            if tmp >= random_numbers:
                parents.append(i)
                break
    return parents


def selection_tournament(fits_pop):
    parents = []
    for individual in range(0, population_size, 1):
        random_number = random.randrange(0, population_size)
        random_number_bis = random_number
        while random_number_bis == random_number:
            random_number_bis = random.randrange(0, population_size)
        if fits_pop[random_number][0] > fits_pop[random_number_bis][0]:
            parents.append(fits_pop[random_number][:])
        else:
            parents.append(fits_pop[random_number_bis][:])
    return parents


def crossover(father, mother):
    print 'crossover'
    cross_point = random.randrange(0, len(father[1][0]) * 3)
    kid0 = father[:]
    kid1 = mother[:]
    x = 0
    y = 0
    while x < 3:
        while y < len(father[1][0]):
            tmp = x * len(father[1][0]) + y
            if tmp >= cross_point:
                kid0[1][x][y] = mother[1][x][y]
                kid1[1][x][y] = father[1][x][y]
            y += 1
        x += 1
        y = 0
    return [kid0, kid1]


def mutation(child):
    print 'mutation'
    gene = random.randrange(0, len(child[1][1]) * 3)
    x = gene / 8
    y = gene % 8
    if child[1][x][y] == '0':
        child[1][x][y] = '1'
    else:
        child[1][x][y] = '0'
    return child


def breed_population(fits_pop):
    parents = selection_tournament(fits_pop)
    # parents = rank_selection(fits_pop)
    # parents = selection_roulette(fits_pop)
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
    return next_population[:]


def print_pop(population, iteration):
    average_fitness = 0
    best_fit = 0
    the_best = []
    for i in population:
        fit = i[0]
        if fit > best_fit:
            best_fit = fit
            the_best = i
        average_fitness += fit

    print '///////////////////////////////////////////////////////////////////'
    print 'Iteration = %d average = %f fitness_general = %f and best = %f' % (
        iteration, average_fitness / len(population), average_fitness, best_fit)
    print 'the_best = ' + str(the_best)
    print '*****************************************************************'
    # for i in population:
    #     print i


def run():
    population = initial_population()
    print_pop(population, 0)
    for iteration in range(0, iteration_limit):
        fits_pop = []
        for individual in population:
            # print '---------------------------------------------------------'
            fits_pop.append(fitness(individual))
            # print "Fitness = %f iteration = %d" % (individual[0], iteration)

        if check_stop(fits_pop, iteration):
            break
        population = breed_population(fits_pop)
        print_pop(population, iteration)


imgCorrupted = cv2.imread('lena_noisy.png', cv2.IMREAD_GRAYSCALE)
imgOriginal = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
height, width = imgOriginal.shape
totalPixel = height * width
run()

#
# # x_axis = [best_fit_list]
#
# # for x in range(0, len(l), 1):
# plt.plot(x_axis, label='iteration')
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#            ncol=2, mode="expand", borderaxespad=0.)
#
# plt.show()
