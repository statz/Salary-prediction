import numpy as np


class GA:
    __n_epoch = None
    __population_size = None
    __estimator = None
    __n_gens = None
    __p_mutation = None

    __train_x = None
    __train_y = None
    __test_x = None
    __test_y = None

    __current_population = None

    def __init__(self, n_epoch, population_size, n_gens, p_mutation, estimator):
        self.__n_epoch = n_epoch
        self.__population_size = population_size
        self.__estimator = estimator
        self.__n_gens = n_gens
        self.p_mutation = p_mutation

    def optimize(self, train_x, train_y, test_x, test_y):
        self.__train_x = train_x
        self.__train_y = train_y
        self.__test_x = test_x
        self.__test_y = test_y

        self.__population_creating()
        self.__search()

    def __population_creating(self):
        self.__current_population = np.random.choice([0, 1],
                                                     size=(self.__population_size, self.__n_gens), p=[0.5, 0.5])

    def __search(self):
        for i in range(self.__n_epoch):
            self.__make_new_generation()

    def __make_new_generation(self):
        fitness_current = []
        for i in range(self.__population_size):
            fitness_current.append(self.__evaluate(self.__current_population[i, :]))
        new_population = []
        for i in range(self.__population_size):
            a, b = self.__roulette_wheel_selection(fitness_current)
            new = (self.__double_point_crossingover(self.__current_population[a, :], self.__current_population[b, :]))
            new_population.append(self.__mutation(new))

    def __roulette_wheel_selection(self, fitness_current):
        cum = np.cumsum(fitness_current)/np.sum(fitness_current)
        a = np.random.ranf(1)
        i = 0
        while a < cum[i]: i += 1
        j = 0
        while i == j:
            b = np.random.ranf(1)
            while b < cum[j]: j += 1
        return [i-1, j-1]

    def __double_point_crossingover(self, first, second):
        a, b = np.sort(np.random.random_integers(0, self.__n_gens, 2))
        first[a:b] = second[a:b]
        return first

    def __mutation(self, genome):
        mutation = np.random.choice([1, 0], size=self.__n_gens, p=[self.__p_mutation, 1-self.__p_mutation])
        return np.mod(genome + mutation)

    def __evaluate(self, genome):
        self.__estimator.fit(self.__train_x[:, np.nonzero(genome)], self.__train_y)
        return self.__estimator.score(self.__test_x[:, np.nonzero(genome)], self.__test_y)

g = GA(2, 5, 5, 5)
g.optimize()
