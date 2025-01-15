import numpy as np
from history.individual import Individual
import warnings
from copy import deepcopy
import random


class ConfigHistory:
    def __init__(self, init_configs, init_fitness_pad, init_fes_pad, multi_instance=True):

        self.configs = deepcopy(init_configs)
        self.fitness = deepcopy(init_fitness_pad)
        self.fes = deepcopy(init_fes_pad)

        if multi_instance:
            self.min_max = self.fitness_min_max()
            self.fitness_norm = self.normalize(init_fitness_pad)
            self.quality_score = np.nanmean(self.fitness_norm, axis=1).reshape((-1, 1))
        else:
            self.quality_score = self.fitness
        self.change_key = False
        self.fes_score = np.nanmean(init_fes_pad, axis=1).reshape((-1, 1))
        self.configs_obj = np.hstack((self.quality_score, self.fes_score))
        self.new_configs_obj = None

    def add(self, configs, fitness_pad, fes_pad, multi_instance=True):
        self.configs.extend(configs)
        self.fitness = np.vstack((self.fitness, fitness_pad))
        self.fes = np.vstack((self.fes, fes_pad))

        if multi_instance:
            new_min_max = self.fitness_min_max()
            if np.allclose(self.min_max, new_min_max, rtol=0, atol=0, equal_nan=True):
                self.change_key = False
                new_fitness_norm = self.normalize(fitness_pad)
                self.fitness_norm = np.vstack((self.fitness_norm, new_fitness_norm))
            else:
                self.change_key = True
                self.min_max = new_min_max
                self.fitness_norm = self.normalize(self.fitness)
            self.quality_score = np.nanmean(self.fitness_norm, axis=1).reshape((-1, 1))
        else:
            self.quality_score = self.fitness

        self.fes_score = np.nanmean(self.fes, axis=1).reshape((-1, 1))
        self.configs_obj = np.hstack((self.quality_score, self.fes_score))
        self.new_configs_obj = self.configs_obj[-len(configs):, :]

    def normalize(self, fitness_array):
        return (fitness_array - self.min_max[0, :]) / (self.min_max[1, :] - self.min_max[0, :])

    def fitness_min_max(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.array([np.nanmin(self.fitness, axis=0), np.nanmax(self.fitness, axis=0)])

    def update_obj(self, arch):
        arch_array = np.array([ind.x for ind in arch])
        configs_array = np.array([config.get_array() for config in self.configs])
        arch_index = np.where((arch_array[:, None, :] == configs_array).all(axis=-1))[1]

        arch_fitness = self.fitness[arch_index, :]
        arch_fes = self.fes[arch_index, :]
        arch_obj = self.configs_obj[arch_index, :]
        new_arch = [Individual(config, obj, fitness, fes) for config, obj, fitness, fes in
                    zip(arch_array, arch_obj, arch_fitness, arch_fes)]

        return new_arch

    def run_instance_configs(self, arch1, arch2):
        sampled_arch1 = random.sample(arch1, 10)

        arch1_array = np.array([ind.x for ind in sampled_arch1])
        arch2_array = np.array([ind.x for ind in arch2])
        print("CA size:{}".format(len(arch1_array)))
        print("DA size:{}".format(len(arch2_array)))

        configs_array = np.array([config.get_array() for config in self.configs])
        arch1_index = np.where((arch1_array[:, None, :] == configs_array).all(axis=-1))[1]
        arch2_index = np.where((arch2_array[:, None, :] == configs_array).all(axis=-1))[1]
        run_configs_index = np.union1d(arch1_index, arch2_index)
        run_configs = [self.configs[idx] for idx in run_configs_index]
        return run_configs, run_configs_index

    def change_instance_value(self, run_configs_index, instance_fitness, instance_fes, instance_idx):
        for i, idx in enumerate(run_configs_index):
            self.fitness[idx, instance_idx] = instance_fitness[i, :]
            self.fes[idx, instance_idx] = instance_fes[i, :]

        self.min_max = self.fitness_min_max()
        self.change_key = True

        self.fitness_norm = self.normalize(self.fitness)

        self.quality_score = np.nanmean(self.fitness_norm, axis=1).reshape((-1, 1))
        self.fes_score = np.nanmean(self.fes, axis=1).reshape((-1, 1))
        self.configs_obj = np.hstack((self.quality_score, self.fes_score))




