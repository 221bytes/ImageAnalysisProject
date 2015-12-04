import GAMaxOneMatrix as MaxOneSolution
import numpy as np
import cv2

prob_crossover = 0.6
prob_mutation = 0.05
population_size = 100
gene_size = 8
nb_parameter = 3
iteration_limit = 100
coef_amp = (30.0 / 255.0)
coef_freq = (0.01 / 255.0)
maximum_pixels_error = 255 * 512 * 512


def initial_population():
    return np.random.randint(2, size=(population_size, nb_parameter * gene_size))


DTYPE = np.float


def imread(fn, dtype=DTYPE):  # img as array
    return cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(dtype)


def make_noise(params):
    NoiseAmp, NoiseFreqRow, NoiseFreqCol = params
    h, w = global_lena.shape
    y = np.arange(h)
    x = np.arange(w)
    col, row = np.meshgrid(x, y, sparse=True)
    noise = NoiseAmp * np.sin(2 * np.pi * NoiseFreqRow * row + 2 * np.pi * NoiseFreqCol * col)
    return noise


def corrupt_image(im, noise_params):
    # read image
    signal = im if type(im) != type("") else imread(im)
    # noise
    noise = make_noise(noise_params)
    # corrupt it
    signal_noisy = signal + noise
    # voila!
    return signal, noise, signal_noisy


def gene_to_noise_params(individual, display=False):
    first_gene = individual[:gene_size]
    second_gene = individual[gene_size:gene_size * 2]
    third_gene = individual[gene_size * 2:]
    if display:
        print first_gene, second_gene, third_gene
    first_gene = np.packbits(first_gene, axis=-1)
    second_gene = np.packbits(second_gene, axis=-1)
    third_gene = np.packbits(third_gene, axis=-1)
    noise_amp = first_gene * coef_amp
    noise_freq_row = second_gene * coef_freq
    noise_freq_col = third_gene * coef_freq
    return noise_amp[0], noise_freq_row[0], noise_freq_col[0]


def lena_fitness(population):
    fitness_population = np.empty(population_size)
    for individual in range(population_size):
        noise_params = gene_to_noise_params(population[individual])
        lena, noise, lena_noisy = corrupt_image(global_lena, noise_params)
        noisy_diff = global_lena_noisy - lena_noisy
        noisy_diff = np.sum(np.abs(noisy_diff)) / global_lena_N  # normalized
        fitness = - noisy_diff  # negative / minimize
        fitness_population[individual] = fitness
    return fitness_population


def run():
    population = initial_population()
    for i in range(0, iteration_limit, 1):
        print 'iteration %d' % i
        fits_pop = [lena_fitness(population), population]
        index = np.argmax(fits_pop[0])
        print gene_to_noise_params(fits_pop[1][index], True), fits_pop[0][index]
        ga.population_size = population_size
        population = ga.breed_population(fits_pop)
    return


global_lena = imread("../Pictures/lena.png")
global_lena_noisy = imread("../Pictures/lena_noisy.png")
global_lena_N = np.prod(global_lena.shape[:2])

ga = MaxOneSolution.GAMaxOneMatrix()
ga.max_one = nb_parameter * gene_size
np.set_printoptions(precision=15)
run()
