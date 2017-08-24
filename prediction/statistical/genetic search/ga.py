import numpy as np


class GA:
    __n_epoch = None
    __population_size = None
    __estimator = None
    __n_gens = None

    def __init__(self, n_epoch, population_size, n_gens, estimator):
        self.__n_epoch = n_epoch
        self.__population_size = population_size
        self.__estimator = estimator
        self.__n_gens = n_gens

    __current_population = None

    def optimize(self):
        self.__population_creating()

    def __population_creating(self):
        self.__current_population = np.random.choice([0, 1], size=(self.__population_size, self.__n_gens), p=[0.5, 0.5])
        self.__search()

    def __search(self):
        for i in range(self.__n_epoch):
            self.__make_new_generation()

    def __make_new_generation(self):
        for i in range(self.__population_size):
            self.__crossingover()
            self.__mutation()

    def __crossingover(self):
        None

    def __mutation(self):
        None

g = GA(2, 5, 5, 5)
g.optimize()
