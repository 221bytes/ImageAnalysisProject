import GAMaxOneMatrix as MaxOneSolution
import numpy as np

population_size = 10
nb_parameter = 3
gene_size = 8
iteration_limit = 100
coef_mut = (1 / 255.0)
coef_cross = (1 / 255.0)
coef_population_size = (population_size / 255.0)


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
    return param_cross, param_mutation, param_population_size


def fitness(population):
    fitness_population = np.empty(population_size)
    for individual in range(population_size):
        n = 10
        param_cross, param_mutation, param_population_size = parameter_from_population(population[individual])
        arr = np.empty(n)
        ga.prob_crossover = param_cross[0]
        ga.prob_mutation = param_mutation[0]
        ga.population_size = param_population_size
        for i in range(n):
            iteration = ga.run()
            arr[i] = iteration
        fitness_population[individual] = iteration_limit - np.sum(arr) / len(arr)
    index = np.argmax(fitness_population)
    print parameter_from_population(population[index]), fitness_population[index]
    return fitness_population


def run():
    population = initial_population()
    for i in range(0, iteration_limit, 1):
        fits_pop = [fitness(population), population]
        population = tool.breed_population(fits_pop)
    return


ga = MaxOneSolution.GAMaxOneMatrix()
tool = MaxOneSolution.GAMaxOneMatrix()
tool.max_one = nb_parameter * gene_size
run()
