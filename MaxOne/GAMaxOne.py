import random
import matplotlib.pyplot as plt
import time
import numpy as np

__author__ = 'Alexandre Catalano'

prob_crossover = 0.60
prob_mutation = 0.05
iteration_limit = 10000000
population_size = 1000
max_one = 1920
l = [10000000]

x_axis = []

for test in l:
    iteration_limit = test
    # population_size = test
    # prob_crossover = test / 100
    # prob_mutation = test / 100
    best_fit_list = []


    def total_fitness(pop):
        total = 0
        for i in pop:
            for j in i:
                total += j
        return total


    def random_individual(n):
        genes = []
        for i in range(0, n, 1):
            genes.append(random.randrange(0, 2))
        return genes


    def initial_population(n):
        population = []
        for i in range(0, n, 1):
            population.append(random_individual(max_one))
        return population


    def fitness(c):
        parent_fitness = 0
        for gene in c:
            parent_fitness += gene
        return parent_fitness


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
            random_numbers_bis = random.randrange(0, population_size)
            if fits_pop[random_numbers][0] > fits_pop[random_numbers_bis][0]:
                parent_pairs.append(fits_pop[random_numbers][1])
            else:
                parent_pairs.append(fits_pop[random_numbers_bis][1])
        return parent_pairs


    def crossover(father, mother):
        size = len(father)
        cross_point0 = random.randrange(0, size)
        cross_point1 = random.randrange(cross_point0, size)
        kid0 = father[:]
        kid1 = mother[:]
        # for i in range(cross_point0, size, 1):
        #     kid0[i] = mother[i]
        #     kid1[i] = father[i]
        kid0[cross_point0: cross_point1] = mother[cross_point0: cross_point1]
        kid1[cross_point0: cross_point1] = father[cross_point0: cross_point1]
        return [kid0, kid1]


    def mutation(child):
        gene = random.randrange(0, len(child))
        if child[gene] == 0:
            child[gene] = 1
        else:
            child[gene] = 0
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
        population_fitness_value = 0
        best_fitness_value = 0
        for k in population:
            fitness_value = 0
            for j in k:
                fitness_value += j
            if fitness_value > best_fitness_value:
                best_fitness_value = fitness_value
            population_fitness_value += fitness_value
        best_fit_list.append(best_fitness_value)
        print "population fitness = %d, best_fitness_value %d" % (population_fitness_value, best_fitness_value)
        print 'iteration = ' + str(turn)


    def run():
        population = initial_population(population_size)
        # print_pop(population, 0)
        arr = np.empty(0)
        for i in range(0, iteration_limit, 1):
            start = time.clock()
            fits_pop = []
            for individual in population:
                fits_pop.append([fitness(individual), individual])
            if check_stop(fits_pop, i):
                break
            population = breed_population(fits_pop)
            end = time.clock()
            arr = np.append(arr, (end-start))
            # print_pop(population, i)
        print np.sum(arr) / len (arr)

    def star():
        arr = np.empty(0)
        for i in range(0, 100, 1):
            start = time.clock()
            run()
            end = time.clock()
            algo_time = end - start
            arr = np.append(arr, algo_time)
            print algo_time
        print np.sum(arr) / len(arr)

    star()
#     x_axis.append(best_fit_list)
#
# for x in range(0, len(l), 1):
#     plt.plot(x_axis[x], label=str(l[x]))
#     plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#                ncol=2, mode="expand", borderaxespad=0.)
#
# plt.show()
