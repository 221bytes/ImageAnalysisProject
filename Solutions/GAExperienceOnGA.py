# import matplotlib.cm as cmx
# import matplotlib.colors as colors
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import GAMaxOneMatrix as MaxOneSolution
import TravelingSalesmanProblem as TSPSoltuion



def initial_population():
    return np.random.randint(2, size=(population_size, nb_parameter * gene_size))


def parameter_from_population(individual):
    first_gene = individual[:8]
    second_gene = individual[8:16]
    third_gene = individual[16:]
    first_gene = np.packbits(first_gene, axis=-1)
    second_gene = np.packbits(second_gene, axis=-1)
    third_gene = np.packbits(third_gene, axis=-1)
    param_mutation = first_gene * coef_mut
    param_cross = second_gene * coef_cross
    param_population_size = int(third_gene * coef_population_size)
    return param_cross[0], param_mutation[0], param_population_size


def fitness(population):
    fitness_population = np.empty(population_size)
    for individual in range(population_size):
        n = 10
        param_cross, param_mutation, param_population_size = parameter_from_population(population[individual])
        arr = np.empty(n)
        ga.prob_crossover = param_cross
        ga.prob_mutation = param_mutation
        ga.population_size = param_population_size
        for i in range(n):
            iteration = ga.run()
            arr[i] = iteration
        fitness_population[individual] = ga.iteration_limit - np.sum(arr) / len(arr)
    index = np.argmax(fitness_population)
    best.append([parameter_from_population(population[index]), fitness_population[index]])
    print parameter_from_population(population[index]), fitness_population[index]
    return fitness_population


def run():
    population = initial_population()
    for i in range(0, iteration_limit, 1):
        print 'iteration %d' % i
        fits_pop = [fitness(population), population]
        tool.population_size = population_size
        population = tool.breed_population(fits_pop)
    return


nb_parameter = 3
gene_size = 8
ga = TSPSoltuion.TSPMatrix()
tool = MaxOneSolution.GAMaxOneMatrix()
tool.max_one = nb_parameter * gene_size
population_size = 100
iteration_limit = 100
coef_mut = (1 / 255.0)
coef_cross = (1 / 255.0)
coef_population_size = (ga.population_size / 255.0)
best = []

run()

#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
cross_value = np.empty(len(best))
mutation_value = np.empty(len(best))
population_size_value = np.empty(len(best))
fitness_value = np.empty(len(best))

for i in range(1, len(best)):
    individual = best[i]
    cross_value[i] = individual[0][0]
    mutation_value[i] = individual[0][1]
    population_size_value[i] = individual[0][2]
    fitness_value[i] = individual[1]

print fitness_value

# x = np.random.standard_normal(100)
# y = np.random.standard_normal(100)
# z = np.random.standard_normal(100)
# c = np.random.standard_normal(100)

# ax.scatter(cross_value, mutation_value, fitness_value, c=population_size, cmap=plt.hot())
# plt.show()


#
# def scatter3d(x, y, z, cs, colorsMap='jet'):
#     cm = plt.get_cmap(colorsMap)
#     cNorm = colors.Normalize(vmin=min(cs), vmax=max(cs))
#     scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
#     scalarMap.set_array(cs)
#     fig.colorbar(scalarMap, shrink=.5, pad=.2, aspect=10, label='Test')
#     plt.show()
#
#
# scatter3d(cross_value, mutation_value, fitness_value, population_size_value)
