import random
import matplotlib.pyplot as plt

__author__ = 'Alexandre Catalano'

prob_crossover = 0.60
prob_mutation = 0.05
iteration_limit = 1000
population_size = 1000
max_one = 196
l = [10, 50, 100, 1000]
xaxis = []

for test in l:
    iteration_limit = test
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


    def selection(fits_pop):
        fitness_population = 0
        parent_pairs = []
        for i in fits_pop:
            fitness_population += i[0]
        for individual in range(0, population_size, 1):
            nb = random.randrange(0, fitness_population)
            tmp = 0
            for i in fits_pop:
                tmp += i[0]
                if tmp >= nb:
                    parent_pairs.append(i[1])
                    break
        return parent_pairs


    def crossover(father, mother):
        size = len(father)
        cross_point = random.randrange(0, size)
        kid0 = father[:]
        kid1 = mother[:]
        for i in range(cross_point, size, 1):
            kid0[i] = mother[i]
            kid1[i] = father[i]
        return [kid0, kid1]


    def mutation(child):
        gene = random.randrange(0, len(child))
        if child[gene] == 0:
            child[gene] = 1
        else:
            child[gene] = 0
        return child


    def breed_population(fits_pop):
        parents = selection(fits_pop)
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
        print_pop(population, 0)
        for i in range(0, iteration_limit, 1):
            fits_pop = []
            for individual in population:
                fits_pop.append([fitness(individual), individual])
            if check_stop(fits_pop, i):
                break
            population = breed_population(fits_pop)
            print_pop(population, i)
    run()
    xaxis.append(best_fit_list)

for x in range(0, len(l), 1):
    plt.plot(xaxis[x], label=str(l[x]))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.show()
