import numpy as np
import scipy.stats as stats
import copy
from ta.base_ta import TA
from ioh import problem


class MadDE(TA):
    def __init__(self, p_qbx=0.01, p=0.18, a_rate=2.3, hm=10, npm=2, f0=0.2, cr0=0.2):
        super().__init__()
        self._p = p
        self._p_qbx = p_qbx
        self._f0 = f0
        self._cr0 = cr0
        self._a_rate = a_rate
        self._npm = npm
        self._hm = hm

        self._pm = np.ones(3) / 3
        self._fe = 0
        self._pop_num = None
        self._arch_num = None
        self._pop = None
        self._archive = None
        self.g_best = None

    @staticmethod
    def _ctb_w_arc(group, best, archive, Fs):
        NP, dim = group.shape
        NB = best.shape[0]
        NA = archive.shape[0]

        count = 0
        rb = np.random.randint(NB, size=NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size=duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + NA, size=NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size=duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        xb = best[rb]
        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]
        v = group + Fs * (xb - group) + Fs * (x1 - x2)

        return v

    @staticmethod
    def _ctr_w_arc(group, archive, Fs):
        NP, dim = group.shape
        NA = archive.shape[0]

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + NA, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]
        v = group + Fs * (x1 - x2)

        return v

    @staticmethod
    def _weighted_rtb(group, best, Fs, Fas):
        NP, dim = group.shape
        NB = best.shape[0]

        count = 0
        rb = np.random.randint(NB, size=NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size=duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP, size=NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        xb = best[rb]
        x1 = group[r1]
        x2 = group[r2]
        v = Fs * x1 + Fs * Fas * (xb - x2)

        return v

    @staticmethod
    def _binomial(x, v, Crs):
        NP, dim = x.shape
        jrand = np.random.randint(dim, size=NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    def _sort(self):
        # new index after sorting
        ind = np.argsort(self.fitness)
        self.fitness = self.fitness[ind]
        self._pop = self._pop[ind]

    def _update_archive(self, old_id):
        if self._archive.shape[0] < self._arch_num:
            self._archive = np.append(self._archive, self._pop[old_id]).reshape(-1, self.dim)
        else:
            self._archive[np.random.randint(self._archive.shape[0])] = self._pop[old_id]

    @staticmethod
    def _mean_wL(df, s):
        w = df / np.sum(df)
        if np.sum(w * s) > 0.000001:
            return np.sum(w * (s ** 2)) / np.sum(w * s)
        else:
            return 0.5

    # randomly choose step length nad crossover rate from MF and MCr
    def _choose_F_Cr(self):
        # generate Cr can be done simutaneously
        gs = self._pop_num
        ind_r = np.random.randint(0, self._MF.shape[0], size=gs)  # index
        c_r = np.minimum(1, np.maximum(0, np.random.normal(loc=self._MCr[ind_r], scale=0.1, size=gs)))
        # as for F, need to generate 1 by 1
        cauchy_locs = self._MF[ind_r]
        f = stats.cauchy.rvs(loc=cauchy_locs, scale=0.1, size=gs)
        err = np.where(f < 0)[0]
        f[err] = 2 * cauchy_locs[err] - f[err]
        return c_r, np.minimum(1, f)

    # update MF and MCr, join new value into the set if there are some successful changes or set it to initial value
    def _update_M_F_Cr(self, s_f, s_cr, df):
        # 更新F与CR
        if s_f.shape[0] > 0:
            mean_wL = self._mean_wL(df, s_f)
            self._MF[self._k] = mean_wL
            mean_wL = self._mean_wL(df, s_cr)
            self._MCr[self._k] = mean_wL
            self._k = (self._k + 1) % self._MF.shape[0]
        else:
            self._MF[self._k] = 0.5
            self._MCr[self._k] = 0.5

    def evaluate(self, ind):
        fit_value = self.instance(ind)
        return (fit_value,)

    def _init_population(self):
        self._pop_num_min = 4
        self._pop_num_max = int(np.round(self._npm * self.dim * self.dim))
        self._H = int(np.round(self._hm * self.dim))

        self._pop_num = self._pop_num_max
        self._arch_num = int(self._a_rate * self._pop_num)
        self._pop = np.random.rand(self._pop_num, self.dim) * (self.up - self.low) + self.low
        self.fitness = np.array(self.instance(self._pop))

        self._fe = self._pop_num
        self._archive = np.array([])
        self._MF = np.ones(self._H) * self._f0
        self._MCr = np.ones(self._H) * self._cr0
        self._k = 0
        self.g_best = np.min(self.fitness)

    def evolve(self, seed, instance):
        np.random.seed(seed)
        self.instance = instance

        if isinstance(self.instance, problem.BBOB):
            self.dim = self.instance.meta_data.n_variables
            self.low = self.instance.bounds.lb[0]
            self.up = self.instance.bounds.ub[0]
        else:
            print("Instance type not supported!")

        self._init_population()
        arch_fitness = copy.deepcopy(self.fitness)
        while self._fe < self.max_fes:
            self._sort()
            pop_num, dim = self._pop_num, self.dim

            q = 2 * self._p - self._p * self._fe / self.max_fes
            fa = 0.5 + 0.5 * self._fe / self.max_fes
            cr, f = self._choose_F_Cr()

            mu = np.random.choice(3, size=pop_num, p=self._pm)
            p1 = self._pop[mu == 0]
            p2 = self._pop[mu == 1]
            p3 = self._pop[mu == 2]
            p_best = self._pop[:max(int(self._p * pop_num), 2)]
            q_best = self._pop[:max(int(q * pop_num), 2)]

            fs = f.repeat(dim).reshape(pop_num, dim)
            v1 = self._ctb_w_arc(p1, p_best, self._archive, fs[mu == 0])
            v2 = self._ctr_w_arc(p2, self._archive, fs[mu == 1])
            v3 = self._weighted_rtb(p3, q_best, fs[mu == 2], fa)

            v = np.zeros((pop_num, dim))
            v[mu == 0] = v1
            v[mu == 1] = v2
            v[mu == 2] = v3

            v[v < self.low] = (self._pop[v < self.low] + self.low) / 2
            v[v > self.up] = (self._pop[v > self.up] + self.up) / 2

            rvs = np.random.rand(pop_num)
            crs = cr.repeat(dim).reshape(pop_num, dim)
            offspring = np.zeros((pop_num, dim))
            if np.any(rvs <= self._p_qbx):
                qu = v[rvs <= self._p_qbx]
                if self._archive.shape[0] > 0:
                    q_best = np.concatenate((self._pop, self._archive), 0)[
                             :max(int(q * (pop_num + self._archive.shape[0])), 2)]
                cross_q_best = q_best[np.random.randint(q_best.shape[0], size=qu.shape[0])]
                qu = self._binomial(cross_q_best, qu, crs[rvs <= self._p_qbx])
                offspring[rvs <= self._p_qbx] = qu
            bu = v[rvs > self._p_qbx]
            bu = self._binomial(self._pop[rvs > self._p_qbx], bu, crs[rvs > self._p_qbx])
            offspring[rvs > self._p_qbx] = bu

            for i, ind in enumerate(offspring):
                fitness = self.instance(ind)

                self._fe += 1
                arch_fitness = np.append(arch_fitness, fitness)

                self.stop_strategy(self._fe, arch_fitness)
                if self.break_flag:
                    fes_norm = (self._fe - self.fes_threshold) / (self.max_fes - self.fes_threshold)
                    return np.min(arch_fitness), fes_norm

            offspring_cost = arch_fitness[-len(offspring):,]

            optim_idx = np.where(offspring_cost < self.fitness)[0]
            non_optim_idx = np.where(offspring_cost >= self.fitness)[0]
            for i in non_optim_idx:
                self._update_archive(i)

            s_f = f[optim_idx]
            s_cr = cr[optim_idx]
            df = np.maximum(0, self.fitness - offspring_cost)
            self._update_M_F_Cr(s_f, s_cr, df[optim_idx])

            count_s = np.zeros(3)
            for i in range(3):
                if len(self.fitness[mu == i]):
                    count_s[i] = np.mean(df[mu == i] / self.fitness[mu == i])
            if np.sum(count_s) > 0:
                self._pm = np.maximum(0.1, np.minimum(0.9, count_s / np.sum(count_s)))
                self._pm /= np.sum(self._pm)
            else:
                self._pm = np.ones(3) / 3

            self._pop[optim_idx] = offspring[optim_idx]
            self.fitness = np.minimum(self.fitness, offspring_cost)

            self._pop_num = int(
                np.round(self._pop_num_max + (self._pop_num_min - self._pop_num_max) * self._fe / self.max_fes))
            self._arch_num = int(self._a_rate * self._pop_num)
            self._sort()
            self._pop = self._pop[:self._pop_num]
            self.fitness = self.fitness[:self._pop_num]
            self._archive = self._archive[:self._arch_num]

            if np.min(self.fitness) < self.g_best:
                self.g_best = np.min(self.fitness)

        fes_norm = (self._fe - self.fes_threshold) / (self.max_fes - self.fes_threshold)
        return self.g_best, fes_norm


if __name__ == '__main__':
    from ioh import get_problem, ProblemClass

    fun_instances = [get_problem(1, i, 10, ProblemClass.BBOB) for i in range(2)]
    instance = fun_instances[0]

    ta = MadDE()
    res = ta.evolve(1, instance)
    print(res)
    print("instance optimum value:{}".format(instance.optimum.y))
