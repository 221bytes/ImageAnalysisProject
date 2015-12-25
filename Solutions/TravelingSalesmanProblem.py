import random
import numpy as np
import time
import csv

__author__ = 'Alexandre Catalano'


class TSPMatrix(object):
    def __init__(self):
        self.prob_crossover = 0.6
        self.prob_mutation = 0.2
        self.iteration_limit = 500
        self.population_size = 100
        self.max_one = 8
        self.distances = np.empty(0)
        self.nb_cities = 0
        self.gene_size = self.nb_cities - 1

    def initial_population(self):
        arr = np.empty((self.population_size, self.nb_cities), dtype=int)
        for i in range(self.population_size):
            path = np.arange(self.nb_cities, dtype=int)
            np.random.shuffle(path)
            arr[i] = path
        return arr

    def print_pop(self, population):
        print population

    def initial_distance(self):
        with open('tsp.cities.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            l = []
            for row in reader:
                l.append(row)
            l = l[2:]
            self.nb_cities = len(l)
            self.gene_size = self.nb_cities - 1
            i = 0
            j = 0
            arr = np.empty((len(l), len(l)), dtype=float)
            while i < len(l):
                while j < len(l):
                    x = l[i]
                    y = l[j]
                    arr[i, j] = np.hypot(float(x[0]) - float(y[0]), float(x[1]) - float(y[1]))
                    j += 1
                j = 0
                i += 1
        self.distances = arr

    def fitness(self, population):
        arr = np.zeros((self.population_size, self.nb_cities - 1), dtype=float)
        for x in range(self.population_size):
            for y in range(self.nb_cities - 1):
                arr[x, y] = self.distances[population[x, y], population[x, y + 1]]
        ret = np.sum(arr, axis=1)
        # index_min = np.argmin(ret)
        # print ret[index_min]
        # print population[index_min]
        return ret

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
            if fits_pop[0][random_numbers] < fits_pop[0][random_numbers_bis]:
                parent_pairs.append(fits_pop[1][random_numbers])
            else:
                parent_pairs.append(fits_pop[1][random_numbers_bis])
        return parent_pairs

    def swap(self, child, x, y):
        tmp = child[x]
        child[x] = child[y]
        child[y] = tmp
        return child

    def crossover(self, father, mother):
        cross_point0 = random.randrange(0, self.nb_cities + 1)
        cross_point1 = random.randrange(cross_point0, self.nb_cities + 1)
        kid0 = father.copy()
        kid1 = mother.copy()
        gene_mother = mother[cross_point0: cross_point1]
        gene_father = father[cross_point0: cross_point1]
        x = 0
        for i in range(cross_point0, cross_point1):
            try:
                itemindex = np.where(father == gene_mother[x])
                kid0 = self.swap(kid0, i, itemindex[0][0])
                itemindex = np.where(mother == gene_father[x])
                kid1 = self.swap(kid1, i, itemindex[0][0])
            except Exception as e:
                print e
                exit(-1)
            x += 1
        # kid0[cross_point0: cross_point1] = gene_mother
        # kid1[cross_point0: cross_point1] = gene_father
        ret = np.vstack((kid0, kid1))
        return ret

    def mutation(self, child):
        gene_0 = random.randrange(0, len(child))
        gene_1 = random.randrange(0, len(child))
        tmp = child[gene_0]
        child[gene_0] = child[gene_1]
        child[gene_1] = tmp
        return child

    def breed_population(self, fits_pop):
        parents = self.selection(fits_pop)
        next_population = np.zeros((self.population_size, self.nb_cities), dtype=int)
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
        self.initial_distance()
        population = self.initial_population()
        print self.fitness(population)
        arr = np.empty(0)
        iteration = 0
        for iteration in range(self.iteration_limit):
            start = time.clock()
            fits_pop = [self.fitness(population), population]
            # if self.check_stop(fits_pop, iteration):
            #     break
            population = self.breed_population(fits_pop)
            end = time.clock()
            arr = np.append(arr, (end - start))
            # print np.sum(arr) / len(arr)
            # print '///////////////////////////////////////// iteration' + str(iteration)
        return iteration


ga = TSPMatrix()
ga.run()
