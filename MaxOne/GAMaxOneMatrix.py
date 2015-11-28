import random
import numpy as np
import time

__author__ = 'Alexandre Catalano'


class GAMaxOneMatrix(object):
    def __init__(self):
        self.prob_crossover = 0.6
        self.prob_mutation = 0.05
        self.iteration_limit = 100
        self.population_size = 10
        self.max_one = 32

    def initial_population(self):
        return np.random.randint(2, size=(self.population_size, self.max_one))

    def print_pop(self, population):
        print population

    def fitness(self, population):
        return np.sum(population, axis=1)

    def check_stop(self, fits_pop, turn):
        for i in range(len(fits_pop[0])):
            fit = fits_pop[0][i]
            if fit == self.max_one:
                # print 'victory in %d iterations' % turn
                # print fits_pop[1][i]
                return True
        return False

    def selection(self, fits_pop):
        parent_pairs = []
        for individual in range(0, self.population_size, 1):
            random_numbers = random.randrange(0, self.population_size)
            random_numbers_bis = random.randrange(0, self.population_size)
            if fits_pop[0][random_numbers] > fits_pop[0][random_numbers_bis]:
                parent_pairs.append(fits_pop[1][random_numbers])
            else:
                parent_pairs.append(fits_pop[1][random_numbers_bis])
        return parent_pairs

    def crossover(self, father, mother):
        cross_point0 = random.randrange(0, self.max_one)
        cross_point1 = random.randrange(cross_point0, self.max_one)
        kid0 = father.copy()
        kid1 = mother.copy()
        kid0[cross_point0: cross_point1] = mother[cross_point0: cross_point1]
        kid1[cross_point0: cross_point1] = father[cross_point0: cross_point1]
        ret = np.vstack((kid0, kid1))
        return ret

    def mutation(self, child):
        gene = random.randrange(0, len(child))
        if child[gene] == 0:
            child[gene] = 1
        else:
            child[gene] = 0
        return child

    def breed_population(self, fits_pop):
        parents = self.selection(fits_pop)
        next_population = np.empty((self.population_size, self.max_one), dtype=int)
        father = 0
        while father < self.population_size:
            if father + 1 == self.population_size:
                next_population[father] = parents[father]
                break
            mother = father + 1
            cross = random.uniform(0, 1) < self.prob_crossover
            if cross is True or cross is np.True_:
                children = self.crossover(parents[father], parents[mother])
            else:
                children = np.vstack((parents[father], parents[mother]))
            for child in children:
                mutate = random.uniform(0, 1) < self.prob_mutation
                if mutate is True or mutate is np.True_:
                    self.mutation(child)
                next_population[father:mother + 1] = children
            father += 2
        return next_population

    def run(self):
        population = self.initial_population()
        arr = np.empty(0)
        iteration = 0
        for iteration in range(self.iteration_limit):
            start = time.clock()
            fits_pop = [self.fitness(population), population]
            if self.check_stop(fits_pop, iteration):
                break
            population = self.breed_population(fits_pop)
            end = time.clock()
            arr = np.append(arr, (end - start))
        # print np.sum(arr) / len(arr)
        return iteration

def start():
    ga = GAMaxOneMatrix()
    arr = np.empty(0)
    for i in range(0, 100, 1):
        start = time.clock()
        ga.run()
        end = time.clock()
        algo_time = end - start
        arr = np.append(arr, algo_time)
        # print algo_time
    print np.sum(arr) / len(arr)

start()


    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # x = np.random.standard_normal(100)
    # y = np.random.standard_normal(100)
    # z = np.random.standard_normal(100)
    # c = np.random.standard_normal(100)
    #
    # ax.scatter(x, y, z, c=c, cmap=plt.hot())
    # plt.show()
