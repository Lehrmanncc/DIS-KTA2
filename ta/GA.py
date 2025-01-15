from deap import base, creator, tools
import numpy as np
import copy
import random
import array
import warnings
from ioh import problem
from ta.base_ta import TA


class GA(TA):
    def __init__(self, pop_num, eta_c, eta_m):
        super().__init__()
        self.population_num = pop_num
        self.eta_c = eta_c
        self.eta_m = eta_m

    def evaluate(self, ind):
        fit_value = self.instance(ind)
        return (fit_value,)

    def evolve(self, seed, instance):
        self.instance = instance

        if isinstance(self.instance, problem.BBOB):
            bm_dim = self.instance.meta_data.n_variables
            bm_low = self.instance.bounds.lb[0]
            bm_up = self.instance.bounds.ub[0]
        else:
            bm_dim = self.instance.num_variables
            bm_low = self.instance.min_bounds[0]
            bm_up = self.instance.max_bounds[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            creator.create("GA_FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("GA_Individual", array.array, typecode='d',
                           fitness=creator.GA_FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, bm_low, bm_up)
        toolbox.register("individual", tools.initRepeat, creator.GA_Individual,
                         toolbox.attr_float, bm_dim)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=bm_low, up=bm_up, eta=self.eta_c)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=bm_low, up=bm_up, eta=self.eta_m,
                         indpb=1.0 / bm_dim)
        toolbox.register("select", tools.selTournament, tournsize=2)

        self.seed = seed
        random.seed(self.seed)
        # pbar = tqdm(desc="pop={}, eta_c={:.4f}, eta_m={:.4f}: {}-th evolve"
        #             .format(self.population_num, self.eta_c, self.eta_m, seed), total=self.max_fes)

        pop = toolbox.population(self.population_num)
        hof_pop = tools.HallOfFame(1)
        hof_arch = tools.HallOfFame(1)

        fitnesses = list(map(self.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        arch = copy.deepcopy(pop)
        fes = self.population_num

        hof_pop.update(pop)
        while fes < self.max_fes:
            if (self.max_fes - fes) < len(pop):
                offspring = toolbox.select(pop, self.max_fes - fes)
            else:
                offspring = toolbox.select(pop, len(pop))
            # 可以应用遗传算子而不影响原始种群
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values

            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values

            # 没有交叉或变异的个体适应度值不变，其余个体的适应度值为空。
            # 查找为空适应度值的个体，并进行再次评估适应度
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for i, ind in enumerate(invalid_ind):
                fitness = self.evaluate(ind)
                ind.fitness.values = fitness

                fes += 1
                arch.append(ind)
                hof_arch.update(arch)
                arch_fitness = np.array([ind.fitness.values for ind in arch])

                self.stop_strategy(fes, arch_fitness)
                if self.break_flag:
                    fes_norm = (fes - self.fes_threshold) / (self.max_fes - self.fes_threshold)
                    return hof_arch[0].fitness.values[0], fes_norm

            combined_pop = pop + offspring
            pop = tools.selBest(combined_pop, len(pop))
            hof_pop.update(pop)

        fes_norm = (fes - self.fes_threshold) / (self.max_fes - self.fes_threshold)

        return hof_pop[0].fitness.values[0], fes_norm


if __name__ == '__main__':
    from ioh import get_problem, ProblemClass

    fun_instances = [get_problem(1, i, 10, ProblemClass.BBOB) for i in range(2)]
    instance = fun_instances[0]

    ta = GA(100, 20, 20)
    res = ta.evolve(1, instance)
    print(res)
    print("instance optimum value:{}".format(instance.optimum.y))

# if __name__ == "__main__":
#     from optproblems.cec2005 import F1, F2, F3, F4
#     import matplotlib.pyplot as plt
#
#     train_instances_feature = [{'dim': 10, 'low': -100, 'up': 100},
#                                {'dim': 10, 'low': -100, 'up': 100},
#                                {'dim': 10, 'low': -100, 'up': 100},
#                                {'dim': 10, 'low': -100, 'up': 100}]
#     instance1, instance2, instance3, instance4 = F1(10), F2(10), F3(10), F4(10)
#     instances = [instance1, instance2, instance3, instance4]
#
#     # print("the 1-th run:")
#     # print("Configuration 1: pop_num:105, F:0.1, CR:0.31")
#     # print("The result of running the 2 instance: ")
#     for instance in instances:
#         print("The result of running the {} instance: ".format(str(instance)))
#         for pop_num in [30]:
#             print("pop_num:{}".format(pop_num))
#             ta1 = GA(instance, train_instances_feature[1], pop_num, 0.4, 0.1)
#             print(ta1.evolve(0))
