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
    __fitness_current = None

    def __init__(self, n_epoch, population_size, n_gens, p_mutation, estimator):
        self.__n_epoch = n_epoch
        self.__population_size = population_size
        self.__estimator = estimator
        self.__n_gens = n_gens
        self.__p_mutation = p_mutation

    def optimize(self, train_x, train_y, test_x, test_y):
        self.__train_x = train_x
        self.__train_y = train_y
        self.__test_x = test_x
        self.__test_y = test_y
        self.__population_creating()
        self.__search()

    def __population_creating(self):
        self.__fitness_current = np.empty(self.__population_size)
        self.__current_population = np.random.choice([0, 1],
                                                     size=(self.__population_size, self.__n_gens), p=[0.7, 0.3])
        for i in range(self.__population_size):
            print(i)
            self.__fitness_current[i] = (self.__evaluate(self.__current_population[i, :]))

        print(self.__fitness_current)

    def __search(self):
        print("Search began")
        for i in range(self.__n_epoch):
            print("########### %d iteration"%i)
            new_gen = np.array(self.__make_new_generation())
            fitness_new = []
            for j in range(self.__population_size):
                fitness_new.append(self.__evaluate(new_gen[j, :]))
            joint_population = np.concatenate((self.__current_population, new_gen))
            fitness_new = np.concatenate((self.__fitness_current, fitness_new), axis = 0)
            next_ind = np.argpartition(fitness_new, self.__population_size)
            self.__current_population = joint_population[next_ind, :]
            self.__fitness_current = fitness_new[next_ind]
            print("best %f"%np.min(self.__fitness_current))

    def __make_new_generation(self):
        new_population = []
        for i in range(self.__population_size):
            a, b = self.__roulette_wheel_selection()
            new = (self.__double_point_crossingover(self.__current_population[a, :], self.__current_population[b, :]))
            new_population.append(self.__mutation(new))
        return new_population

    def __roulette_wheel_selection(self):
        cum = np.cumsum(self.__fitness_current**-1)/np.sum(self.__fitness_current**-1)
        a = np.random.ranf(1)
        i = 0
        while i < len(cum) and a < cum[i]:
            i += 1
        j = 0
        while i == j:
            j = 0
            b = np.random.ranf(1)
            while j < len(cum) and b < cum[j] :
                j += 1
        return [i-1, j-1]

    def __double_point_crossingover(self, first, second):
        a, b = np.sort(np.random.random_integers(0, self.__n_gens, 2))
        first[a:b] = second[a:b]
        return first

    def __mutation(self, genome):
        mutation = np.random.choice([1, 0], size=self.__n_gens, p=[self.__p_mutation, 1-self.__p_mutation])
        return np.mod(genome + mutation, 2)

    def __evaluate(self, genome):
        ind = np.array(np.where(genome))
        ind = ind.reshape(ind.shape[1])
        self.__estimator.fit(self.__train_x[:, ind], self.__train_y)
        return np.sum(np.abs(self.__estimator.predict(self.__test_x[:, ind])
                             - self.__test_y[:])/self.__test_y[:])/len(self.__test_y)

