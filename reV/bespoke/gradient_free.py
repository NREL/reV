# -*- coding: utf-8 -*-
"""
a simple genetic algorithm
"""
import numpy as np
import time
from math import log
import logging

logger = logging.getLogger(__name__)


class GeneticAlgorithm():
    """a simple genetic algorithm used to select bespoke turbine locations
    """

    def __init__(self, bits, bounds, variable_type, objective_function,
                 max_generation=100, population_size=0, crossover_rate=0.1,
                 mutation_rate=0.01, tol=1E-6, convergence_iters=5,
                 max_time=3600):
        """
        Parameters
        ----------
        bits : array of ints
            The number of bits assigned to each of the design variables.
            The number of discretizations for each design variables will be
            2^n where n is the number of bits assigned to that variable.
        bounds : array of tuples
            The bounds for each design variable. This parameter looks like:
            np.array([(lower, upper), (lower, upper)...])
        variable_type : array of strings ('int' or 'float')
            The type of each design variable (int or float).
        objective_function : function handle for the objective that is to be
            minimized. Should take a single variable as an input which is a
            list/array of the design variables.
        max_generation : int, optional
            The maximum number of generations that will be run in the genetic
            algorithm.
        population_size : int, optional
            The population size in the genetic algorithm.
        crossover_rate : float, optional
            The probability of crossover for a single bit during the crossover
            phase of the genetic algorithm.
        mutation_rate : float, optional
            The probability of a single bit mutating during the mutation phase
            of the genetic algorithm.
        tol : float, optional
            The absolute tolerance to determine convergence.
        convergence_iters : int, optional
            The number of generations to determine convergence.
        max_time : float
            The maximum time (in seconds) to run the genetic algorithm.
        """

        logger.debug('Initializing GeneticAlgorithm...')
        logger.debug('Minimum convergence iterations: {}'
                     .format(convergence_iters))
        logger.debug('Max iterations (generations): {}'.format(max_generation))
        logger.debug('Population size: {}'.format(population_size))
        logger.debug('Crossover rate: {}'.format(crossover_rate))
        logger.debug('Mutation rate: {}'.format(mutation_rate))
        logger.debug('Convergence tolerance: {}'.format(tol))
        logger.debug('Maximum runtime (in seconds): {}'.format(max_time))

        # inputs
        self.bits = bits
        self.bounds = bounds
        self.variable_type = variable_type
        self.objective_function = objective_function
        self.max_generation = max_generation
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tol = tol
        self.convergence_iters = convergence_iters
        self.max_time = max_time

        # internal variables, you could output some of this info if you wanted
        self.design_variables = np.array([])  # the desgin variables as they
        # are passed into self.objective function
        self.nbits = 0  # the total number of bits in each chromosome
        self.nvars = 0  # the total number of design variables
        self.parent_population = np.array([])  # 2D array containing all of the
        # parent individuals
        self.offspring_population = np.array([])  # 2D array containing all of
        # the offspring individuals
        self.parent_fitness = np.array([])  # array containing all of the
        # parent fitnesses
        self.offspring_fitness = np.array([])  # array containing all of the
        # offspring fitnesses
        self.discretized_variables = {}  # a dict of arrays containing all of
        # the discretized design variable

        # outputs
        self.solution_history = np.array([])
        self.optimized_function_value = 0.0
        self.optimized_design_variables = np.array([])

        self.initialize_design_variables()
        self.initialize_bits()
        if self.population_size % 2 == 1:
            self.population_size += 1
        self.initialize_population()
        self.initialize_fitness()

        if self.population_size > 5:
            n = 5
        else:
            n = self.population_size
        logger.debug('The first few parent individuals are: {}'
                     .format(self.parent_population[0:n]))
        logger.debug('The first few parent fitness values are: {}'
                     .format(self.parent_fitness[0:n]))

    def initialize_design_variables(self):
        """initialize the design variables from the randomly initialized
        population
        """
        # determine the number of design variables and initialize
        self.nvars = len(self.variable_type)
        self.design_variables = np.zeros(self.nvars)
        float_ind = 0
        for i in range(self.nvars):
            if self.variable_type[i] == "float":
                ndiscretizations = 2**self.bits[i]
                self.discretized_variables["float_var%s" % float_ind] = \
                    np.linspace(self.bounds[i][0], self.bounds[i][1],
                                ndiscretizations)
                float_ind += 1

    def initialize_bits(self):
        """determine the total number of bits"""
        # determine the total number of bits
        for i in range(self.nvars):
            if self.variable_type[i] == "int":
                int_range = self.bounds[i][1] - self.bounds[i][0]
                int_bits = int(np.ceil(log(int_range, 2)))
                self.bits[i] = int_bits
            self.nbits += self.bits[i]

    def initialize_population(self):
        """randomly initialize the parent and offspring populations"""
        all_bits_on = np.ones((1, self.nbits))
        random_bits_on = np.random.randint(
            0, high=2, size=(self.population_size - 1, self.nbits)
        )
        self.parent_population = np.r_[all_bits_on, random_bits_on]
        self.offspring_population = np.zeros_like(self.parent_population)

    def initialize_fitness(self):
        """initialize the fitness of member of the parent population"""
        # initialize the fitness arrays
        self.parent_fitness = np.zeros(self.population_size)
        self.offspring_fitness = np.zeros(self.population_size)

        # initialize fitness of the parent population
        for i in range(self.population_size):
            self.chromosome_2_variables(self.parent_population[i])
            self.parent_fitness[i] = \
                self.objective_function(self.design_variables)

    def chromosome_2_variables(self, chromosome):
        """convert the binary chromosomes to design variable values"""

        first_bit = 0
        float_ind = 0

        for i in range(self.nvars):
            binary_value = 0
            for j in range(self.bits[i]):
                binary_value += chromosome[first_bit + j] * 2**j
            first_bit += self.bits[i]

            if self.variable_type[i] == "float":
                self.design_variables[i] = \
                    self.discretized_variables["float_var%s"
                                               % float_ind][binary_value]
                float_ind += 1

            elif self.variable_type[i] == "int":
                self.design_variables[i] = self.bounds[i][0] + binary_value

    def crossover(self):
        """perform crossover between individual parents"""
        self.offspring_population[:, :] = self.parent_population[:, :]

        # mate conscutive pairs of parents (0, 1), (2, 3), ...
        # The population is shuffled so this does not need to be randomized
        for i in range(int(self.population_size / 2)):
            # trade bits in the offspring
            crossover_arr = np.random.rand(self.nbits)
            for j in range(self.nbits):
                if crossover_arr[j] < self.crossover_rate:
                    self.offspring_population[2 * i][j], \
                        self.offspring_population[2 * i + 1][j] = \
                        self.offspring_population[2 * i + 1][j], \
                        self.offspring_population[2 * i][j]

    def mutate(self):
        """randomly mutate bits of each chromosome"""
        for i in range(int(self.population_size)):
            # mutate bits in the offspring
            mutate_arr = np.random.rand(self.nbits)
            for j in range(self.nbits):
                if mutate_arr[j] < self.mutation_rate:
                    self.offspring_population[i][j] = \
                        (self.offspring_population[i][j] + 1) % 2

    def optimize_ga(self):
        """run the genetic algorithm"""

        converged = False
        ngens = 1
        generation = 1
        difference = self.tol * 10000.0
        self.solution_history = np.zeros(self.max_generation + 1)
        self.solution_history[0] = np.min(self.parent_fitness)

        run_time = 0.0
        start_time = time.time()
        while converged is False and ngens < self.max_generation and \
                run_time < self.max_time:
            self.crossover()
            self.mutate()
            # determine fitness of offspring
            for i in range(self.population_size):
                self.chromosome_2_variables(self.offspring_population[i])
                self.offspring_fitness[i] = \
                    self.objective_function(self.design_variables)

            # rank the total population from best to worst
            total_fitness = np.append(self.parent_fitness,
                                      self.offspring_fitness)
            ranked_fitness = \
                np.argsort(total_fitness)[0:int(self.population_size)]

            total_population = \
                np.vstack([self.parent_population, self.offspring_population])
            self.parent_population[:, :] = total_population[ranked_fitness, :]
            self.parent_fitness[:] = total_fitness[ranked_fitness]

            # store solution history and wrap up generation
            self.solution_history[generation] = np.min(self.parent_fitness)

            if generation > self.convergence_iters:
                difference = \
                    self.solution_history[generation - self.convergence_iters]\
                    - self.solution_history[generation]
            else:
                difference = 1000
            if abs(difference) <= self.tol:
                converged = True

            # shuffle up the order of the population
            shuffle_order = np.arange(1, self.population_size)
            np.random.shuffle(shuffle_order)
            shuffle_order = np.append([0], shuffle_order)
            self.parent_population = self.parent_population[shuffle_order]
            self.parent_fitness = self.parent_fitness[shuffle_order]

            generation += 1
            ngens += 1

            run_time = time.time() - start_time

        # Assign final outputs
        self.solution_history = self.solution_history[0:ngens]
        self.optimized_function_value = np.min(self.parent_fitness)
        self.chromosome_2_variables(
            self.parent_population[np.argmin(self.parent_fitness)])
        self.optimized_design_variables = self.design_variables

        logger.debug('The GA ran for this many generations: {}'
                     .format(ngens))
        logger.debug('The GA ran for this many seconds: {:.3f}'
                     .format(run_time))
        logger.debug('The optimized function value was: {:.3e}'
                     .format(self.optimized_function_value))
        logger.debug('The optimal design variables were: {}'
                     .format(self.optimized_design_variables))
