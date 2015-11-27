import random
import numpy as np
import time

__author__ = 'Alexandre Catalano'

prob_crossover = 0.60
prob_mutation = 0.05
iteration_limit = 10000000
population_size = 10
max_one = 8


def initial_population():
    return np.random.randint(2, size=(population_size, max_one))


def print_pop(population):
    print population


def fitness(population):
    return np.sum(population, axis=1)


def check_stop(fits_pop, turn):
    f, p = fits_pop
    bf_idx = np.argmax(f)  # best_fit index
    fit = f[bf_idx]
    if fit == max_one:
        print 'victory in %d iterations' % turn
        print fit, p[bf_idx]
        return True
    return False


def selection(fits_pop):
    N = population_size
    parent_pairs = [None] * N  # allocate at once (append is sloooooooooow!)
    f, p = fits_pop  # alias
    for i in range(population_size):
        i1, i2 = random.randrange(N), random.randrange(N)
        f1, f2 = f[i1], f[i2]
        strong = p[i1]
        if f1 < f2: strong = p[i2]  # select the stronger (tournament)
        parent_pairs[i] = strong
    return parent_pairs


def crossover(father, mother):
    i1, i2 = random.randrange(1, max_one - 1), random.randrange(1, max_one - 1),
    if i1 > i2:
        i1, i2 = i2, i1  # swap
    kid0 = father.copy()
    kid1 = mother.copy()
    kid0[i1: i2] = mother[i1: i2]
    kid1[i1: i2] = father[i1: i2]
    ret = np.vstack((kid0, kid1))
    return ret


def mutation(child):
    gene = random.randrange(len(child))
    child[gene] = 1 - child[gene]
    return child


def breed_population(fits_pop):
    parents = selection(fits_pop)
    next_population = np.empty((population_size, max_one),
                               dtype=int)  # XXX toDo: use 2 fixed arrays and swtich between them # (current_population, tmp_population) # avoid creating a new one in every iteration!!!
    father = 0
    while father < population_size:
        mother = father + 1
        cross = random.uniform(0, 1) < prob_crossover
        if cross is True:
            children = crossover(parents[father], parents[mother])
        else:
            children = np.vstack((parents[father], parents[mother]))
        for child in children:
            mutate = random.uniform(0, 1) < prob_mutation
            if mutate is True:
                mutation(child)
        next_population[father:mother + 1] = children
        father += 2
    return next_population


def run():
    population = initial_population()
    arr = np.empty(iteration_limit)
    for i in range(iteration_limit):
        start = time.clock()
        fits_pop = [fitness(population), population]
        if check_stop(fits_pop, i):
            break
        population = breed_population(fits_pop)
        end = time.clock()
        arr[i] = (end - start)
    print np.sum(arr) / len(arr)


def star():
    N = 100
    #	N = 1
    arr = np.empty(N)
    for i in range(N):
        start = time.clock()
        run()
        end = time.clock()
        algo_time = end - start
        arr[i] = algo_time
        print algo_time
    print np.sum(arr) / len(arr)


star()
