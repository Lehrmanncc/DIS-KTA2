import numpy as np


class TA:
    def __init__(self):
        self.max_fes = 10000

        self.fitness = 0
        self.fes_score = 0
        self.seed = None
        self.instance = None
        self.dim = None
        self.low = None
        self.up = None
        self.toolbox = None

        self.alpha = 0.4
        self.beta = 200
        self.window_size = 10
        self.fes_threshold = 500
        self.best_values = []
        self.f_value = []

        # 监控变量
        self.t = -1
        self.t_es = 50
        self.t_r = 400
        self.e_mean = np.inf
        self.dsd_value = np.inf
        self.break_flag = False

    def stop_strategy(self, fe, arch_fitness):

        if fe > self.fes_threshold:
            best_fitness = np.min(arch_fitness)
            self.t += 1
            mean_value = np.mean(arch_fitness[-self.window_size:])
            self.best_values.append(mean_value * self.alpha + best_fitness * (1 - self.alpha))

            if self.t > 1:
                f = ((self.best_values[self.t] - self.best_values[self.t - 1]) -
                     (self.best_values[self.t - 1] - self.best_values[self.t - 2])) / 2
                self.f_value.append(f)

                if self.t == self.t_es + 1:
                    self.e_mean = np.sum(np.abs(self.f_value)) / self.t_es
                    if self.e_mean == 0:
                        self.t_es += 1
                        self.t_r += 1

                if self.t > self.t_r:
                    f_hat = [f / self.e_mean for f in self.f_value]
                    self.dsd_value = np.sum(np.abs(f_hat[self.t - self.t_r - 1:]))

            if self.dsd_value < self.beta:
                self.break_flag = True
