import os
import numpy as np
import random
from pyDOE import lhs
import ConfigSpace.hyperparameters as csh
from itertools import chain
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.optimize import OptimizeResult
from pymoo.indicators.hv import Hypervolume
import skopt
import matplotlib.pyplot as plt


def LHDesign(config_space, sample_num, seed):
    np.random.seed(seed=seed)
    lhs_samples = lhs(len(config_space), samples=sample_num, criterion='maximin')
    configs = []
    for i in range(sample_num):
        config = config_space.sample_configuration()
        for j, key in enumerate(config_space):
            if isinstance(config_space[key], csh.UniformIntegerHyperparameter):
                config[key] = int(lhs_samples[i, j] * (config_space[key].upper - config_space[key].lower)
                                  + config_space[key].lower)
            elif isinstance(config_space[key], csh.UniformFloatHyperparameter):
                config[key] = (lhs_samples[i, j] * (config_space[key].upper - config_space[key].lower)
                               + config_space[key].lower)
            else:
                print("ConfigSpace Hyperparameters TypeError!")
        configs.append(config)

    return configs


def get_cfs_low_up(config_space):
    lower = np.empty(len(config_space))
    upper = np.empty(len(config_space))
    for i, key in enumerate(config_space):
        lower[i] = config_space[key].lower
        upper[i] = config_space[key].upper
    return lower, upper


def convert_config_array(config):
    array = np.empty(shape=len(config.config_space))
    for i, (key, hp) in enumerate(config.items()):
        array[i] = hp
    return array


def convert_configs_to_array(configs):
    configs_array = np.array([convert_config_array(config) for config in configs])
    return configs_array


def array_to_configs(config_space, array):
    configs = []
    for config_array in array:
        config = config_space.sample_configuration()
        for j, key in enumerate(config_space):
            if isinstance(config_space[key], csh.UniformIntegerHyperparameter):
                config[key] = int(config_array[j])
            else:
                config[key] = config_array[j]
        configs.append(config)
    return configs


# ta/rand/1
def mutDE(y, a, b, c, f):
    # size = len(y)
    for i in range(len(y)):
        y[i] = a[i] + f * (b[i] - c[i])
    return y


# 指数交叉
def cxExponential(x, y, cr):
    size = len(x)
    index = random.randrange(size)
    # Loop on the indices index -> end, then on 0 -> index
    for i in chain(range(index, size), range(0, index)):
        x[i] = y[i]
        if random.random() < cr:
            break
    return x


def non_dominated_sort(pop_obj, front_num):
    unique_pop_obj, unique_indices = np.unique(pop_obj, axis=0, return_inverse=True)
    table, _ = np.histogram(unique_indices, bins=(max(unique_indices) + 1))

    unique_pop_num, obj_num = unique_pop_obj.shape
    front_flag = np.full(unique_pop_num, np.inf)
    max_front_flag = 0

    while np.sum(table[front_flag < np.inf]) < min(front_num, len(unique_indices)):
        max_front_flag += 1
        for i in range(unique_pop_num):
            if front_flag[i] == np.inf:
                dominated = False
                for j in range(i - 1, -1, -1):
                    if front_flag[j] == max_front_flag:
                        if np.sum(unique_pop_obj[i, :] >= unique_pop_obj[j, :]) == obj_num:
                            dominated = True
                        if dominated or obj_num == 2:
                            break
                if not dominated:
                    front_flag[i] = max_front_flag

    non_dominated_ind = front_flag[unique_indices]
    return non_dominated_ind


def statsrexact(v, w):
    n = len(v)
    v = np.sort(v)

    max_w = n * (n + 1) / 2
    folded = w > max_w / 2
    if folded:
        w = max_w - w

    doubled = np.any(v != np.floor(v))
    if doubled:
        v = np.round(2 * v)
        w = np.round(2 * w)

    C = np.zeros(w + 1)
    C[0] = 1
    top = 1

    for vj in v[v <= w]:
        new_top = min(top + vj, w + 1)
        hi = np.arange(min(vj, w + 1), new_top).astype("int64")
        lo = np.arange(0, len(hi)).astype("int64")

        C[hi] = C[hi] + C[lo]
        top = new_top

    C = C / (2 ** n)
    p_val = np.sum(C)

    all_w = np.arange(0, w + 1)
    if doubled:
        all_w = all_w / 2
    if folded:
        all_w = n * (n + 1) / 2 - all_w

    P = np.column_stack((all_w, C))

    return p_val


def sign_rank(x, y, alpha=0.05, method="auto"):
    diff_xy = x - y
    eps_diff = np.finfo(float).eps * (np.abs(x) + np.abs(y))

    t = np.isnan(diff_xy)
    diff_xy = diff_xy[~t]
    eps_diff = eps_diff[~t]

    t = np.abs(diff_xy) < eps_diff
    diff_xy = diff_xy[~t]

    n = len(diff_xy)
    if n == 0:
        p, h = 1, 0
        return p, h

    if method == "auto":
        if n < 15:
            method = 'exact'
        else:
            method = 'approximate'

    neg = diff_xy < 0
    tie_rank = stats.rankdata(np.abs(diff_xy), method='average')

    w = np.sum(tie_rank[neg]).astype("int64")
    r1 = w
    r2 = (n * (n + 1) / 2 - w).astype("int64")
    w = min(w, n * (n + 1) / 2 - w).astype("int64")

    if method == 'approximate':
        z = (w - n * (n + 1) / 4) / np.sqrt((n * (n + 1) * (2 * n + 1)) / 24)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    else:  # method == 'exact'
        p = statsrexact(tie_rank, w)
        p = min(1, 2 * p)

    h = int(p <= alpha)

    return p, h, r1, r2


def cal_convergence(pop1_obj, pop2_obj, min_point):
    if pop1_obj.shape[0] != pop2_obj.shape[0]:
        flag = 0
    else:
        pop_obj = np.vstack((pop1_obj, pop2_obj))
        pop_obj_norm = (pop_obj - min_point) / (pop_obj.max(axis=0) - min_point)

        distance1 = np.sqrt(np.sum(pop_obj_norm[:pop1_obj.shape[0], :], axis=1))
        distance2 = np.sqrt(np.sum(pop_obj_norm[pop1_obj.shape[0]:, :], axis=1))

        _, flag, r1, r2 = sign_rank(distance1, distance2)
        if flag == 1 and (r1 - r2 < 0):
            flag = 0

    return flag


def pure_diversity(pop_obj):
    if pop_obj.size == 0:
        score = np.nan
    else:
        n = pop_obj.shape[0]
        c = np.zeros((n, n), dtype=bool)
        np.fill_diagonal(c, True)

        dist = cdist(pop_obj, pop_obj, metric='minkowski', p=0.1)
        np.fill_diagonal(dist, np.inf)
        score = 0

        for _ in range(n - 1):
            while True:
                d = np.min(dist, axis=1)
                j = np.argmin(dist, axis=1)
                i = np.argmax(d)

                if dist[j[i], i] != -np.inf:
                    dist[j[i], i] = np.inf
                if dist[i, j[i]] != -np.inf:
                    dist[i, j[i]] = np.inf
                p = np.any(c[i, :].reshape((1, -1)), axis=0)
                while not p[j[i]]:
                    new_p = np.any(c[p, :], axis=0)
                    if np.array_equal(p, new_p):
                        break
                    else:
                        p = new_p
                if not p[j[i]]:
                    break
            c[i, j[i]] = True
            c[j[i], i] = True
            dist[i, :] = -np.inf
            score += d[i]

    return score


def mating_selection(ca_dec, ca_obj, da_dec, da_obj, parent_size):
    ca_parent1_index = np.random.randint(0, ca_obj.shape[0], size=int(np.ceil(parent_size / 2)))
    ca_parent2_index = np.random.randint(0, ca_obj.shape[0], size=int(np.ceil(parent_size / 2)))

    a1 = 0 + np.any(ca_obj[ca_parent1_index, :] < ca_obj[ca_parent2_index, :], axis=1)
    a2 = 0 + np.any(ca_obj[ca_parent1_index, :] > ca_obj[ca_parent2_index, :], axis=1)
    dominate = a1 - a2

    choose_index = np.concatenate((ca_parent1_index[dominate == 1], ca_parent2_index[dominate != 1]))
    parent_c_obj = ca_obj[choose_index, :]
    parent_c_dec = ca_dec[choose_index, :]

    parent_c_obj = np.vstack((parent_c_obj, da_obj[np.random.randint(0, da_obj.shape[0],
                                                                     size=int(np.ceil(parent_size / 2))), :]))
    parent_c_dec = np.vstack((parent_c_dec, da_dec[np.random.randint(0, da_dec.shape[0],
                                                                     size=int(np.ceil(parent_size / 2))), :]))
    parent_m_obj = ca_obj[np.random.randint(0, ca_obj.shape[0], size=parent_size), :]
    parent_m_dec = ca_dec[np.random.randint(0, ca_dec.shape[0], size=parent_size), :]

    return parent_c_dec, parent_m_dec


def cross_mutation(parent, config_space, pc, pm, yita1, yita2):
    parent1 = parent[:len(parent) // 2, :]
    parent2 = parent[len(parent) // 2:(len(parent) // 2) * 2, :]
    n, d = parent1.shape

    # Simulated binary crossover
    beta = np.zeros((n, d))
    mu = np.random.rand(n, d)

    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (yita1 + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (yita1 + 1))
    beta = beta * ((-1) ** np.random.randint(0, 2, size=(n, d)))

    beta[np.random.rand(n, d) < 0.5] = 1
    beta[np.random.rand(n, d) > pc] = 1

    off_cross = np.vstack([(parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2,
                           (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2])

    # Polynomial mutation
    xl = np.array([0.0] * len(config_space))
    xu = np.array([1.0] * len(config_space))
    lower = np.tile(xl, (2 * n, 1))
    upper = np.tile(xu, (2 * n, 1))
    site = np.random.rand(2 * n, d) < (pm / d)
    mu = np.random.rand(2 * n, d)

    temp = site & (mu <= 0.5)
    off_mu = np.minimum(np.maximum(off_cross, lower), upper)
    off_mu[temp] = off_mu[temp] + (upper[temp] - lower[temp]) * (
            (2 * mu[temp] + (1 - 2 * mu[temp]) *
             (1 - (off_mu[temp] - lower[temp]) / (upper[temp] - lower[temp])) ** (yita2 + 1)) ** (1 / (yita2 + 1)) - 1)

    temp = site & (mu > 0.5)
    off_mu[temp] = off_mu[temp] + (upper[temp] - lower[temp]) * (
            1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) *
                 (1 - (upper[temp] - off_mu[temp]) / (upper[temp] - lower[temp])) ** (yita2 + 1)) ** (1 / (yita2 + 1)))

    return off_mu


def mk_dir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        pass


def optimize_result(best_arch, best_configs, test_fitness, test_fes, configs_history,
                    sampling_method, da_history, run_times, run_instances):
    res = OptimizeResult()
    res.test_fitness = test_fitness
    res.test_fes = test_fes
    res.configs_history = configs_history
    res.sampling_method = sampling_method
    res.run_times = run_times
    res.run_instances = run_instances

    configs_array_norm = np.array([config.get_array() for config in configs_history.configs])
    if isinstance(best_configs, list):
        best_configs_array = np.array([config.get_array() for config in best_configs])
        res.best_configs = best_configs
    else:
        best_configs_array = np.array([best_configs.get_array()])
        res.best_configs = [best_configs]

    arch_index = np.where((best_configs_array[:, None, :] == configs_array_norm).all(axis=-1))[1]
    res.arch = best_arch
    res.arch_history = da_history

    res.arch_index = arch_index
    res.arch_fitness = configs_history.fitness[arch_index, :]
    res.arch_fes = configs_history.fes[arch_index, :]
    res.arch_obj = configs_history.configs_obj[arch_index, :]

    return res


def optimize_nsga2_result(best_arch, best_configs, test_fitness, test_fes, configs_history):
    res = OptimizeResult()
    res.test_fitness = test_fitness
    res.test_fes = test_fes
    res.configs_history = configs_history

    configs_array_norm = np.array([config.get_array() for config in configs_history.configs])
    arch_dec = np.array([ind for ind in best_arch])
    # print(arch_dec)
    arch_index = np.where((arch_dec[:, None, :] == configs_array_norm).all(axis=-1))[1]

    res.arch = best_arch
    res.best_configs = best_configs

    res.arch_index = arch_index
    res.arch_fitness = configs_history.fitness[arch_index, :]
    res.arch_fes = configs_history.fes[arch_index, :]
    res.arch_obj = configs_history.configs_obj[arch_index, :]
    res.arch_obj1 = np.array([ind.fitness.values for ind in best_arch])

    return res


def remove_similar_rows(array, threshold=5e-3):
    if array.shape[0] == 1:
        return array
    elif np.all(np.isnan(array).any(axis=1)):
        return array
    else:
        threshold_array = np.full((array.shape[0], array.shape[0]), threshold)
        dist = cdist(array, array, metric='euclidean')
        mask = dist > threshold_array
        id_r, id_c = np.triu_indices_from(dist, k=1)
        remove_id_r = [j for (i, j) in zip(id_r, id_c) if not mask[i, j]]
        array_filter = np.delete(array, remove_id_r, axis=0)
        return array_filter


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


def plot_pareto_front(exp_num, obj_list, labels, save_path, save_name, title_name):
    plt.style.use(['science', 'no-latex', 'ieee'])
    # 创建图表和子图
    for obj, label in zip(obj_list, labels):
        plt.scatter(obj[:, 0], obj[:, 1], label=label, alpha=1)

    plt.xlabel('F1(fitness score)')
    plt.ylabel('F2(fes score)')
    plt.title('{}-th: {}'.format(exp_num, title_name))

    plt.legend()

    plt.savefig("{}/{}th-{}".format(save_path, exp_num, save_name))
    plt.show()


def plot_mean_curve(data_lst, save_path, save_name, title_name, colors, labels, linestyle):
    plt.style.use(['science', 'no-latex', 'ieee'])
    plt.figure(figsize=(3, 2.5))
    for i, data in enumerate(data_lst):
        mean_budget, mean_train_hv, std_train_hv = data

        plt.plot(mean_budget, mean_train_hv, color=colors[i], label=labels[i], linestyle=linestyle[i])
        plt.xlabel('Maximum Budget')
        plt.ylabel('Average Hypervolume')
        plt.title('{}'.format(title_name))
    #

    plt.legend(loc='best', frameon=True, fontsize=8)
    plt.tight_layout(pad=0.5)
    plt.savefig("{}/{}.pdf".format(save_path, save_name), bbox_inches='tight')
    plt.show()


def plot_bar(data_list, save_path, save_name, labels):
    plt.style.use(['science', 'no-latex', 'ieee'])
    dist1, dist2 = data_list
    x1, y1 = zip(*sorted(dist1.items()))
    x2, y2 = zip(*sorted(dist2.items()))
    bar_width = 0.4

    x1_positions = np.array(x1) - bar_width / 2
    x2_positions = np.array(x2) + bar_width / 2

    plt.figure(figsize=(3, 2.5))
    plt.bar(x1_positions, y1, width=bar_width, color='skyblue', edgecolor='black', linewidth=0.6, label=labels[0])
    plt.bar(x2_positions, y2, width=bar_width, color='orange', edgecolor='black', linewidth=0.6, label=labels[1])

    plt.legend()
    plt.title("Evaluation Counts Distribution")
    plt.xlabel("Evaluation Counts")
    plt.ylabel("Number of Configurations")

    plt.xticks(range(min(min(x1), min(x2)), max(max(x1), max(x2)) + 1))
    plt.yticks()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("{}/{}.pdf".format(save_path, save_name), bbox_inches='tight')
    plt.show()


def plot_curve(exp_num, x, y, save_path, save_name, title_name):
    plt.style.use(['science', 'no-latex', 'ieee'])

    plt.plot(x, y)

    plt.xlabel('F1(fitness score)')
    plt.ylabel('F2(fes score)')
    plt.title('{}'.format(title_name))

    plt.show()


def calculate_da_hv(da_list_t1, da_list_t2, config_history):

    ref_point = np.array([1, 1])
    hv = Hypervolume(ref_point)

    norm_fitness_list_t1 = [config_history.normalize(np.array([ind.fitness for ind in da_t1]))
                            for da_t1 in da_list_t1]
    quality_obj_list_t1 = [np.nanmean(norm_fitness_t1, axis=1).reshape((-1, 1))
                           for norm_fitness_t1 in norm_fitness_list_t1]

    fe_obj_list_t1 = [np.array([ind.obj for ind in da_t1])[:, 1].reshape((-1, 1)) for da_t1 in da_list_t1]
    obj_list_t1 = [np.hstack((quality_obj_t1, fe_obj_t1)) for quality_obj_t1, fe_obj_t1 in
                   zip(quality_obj_list_t1, fe_obj_list_t1)]

    norm_fitness_list_t2 = [config_history.normalize(np.array([ind.fitness for ind in da_t2])) for da_t2 in da_list_t2]
    quality_obj_list_t2 = [np.nanmean(norm_fitness_t2, axis=1).reshape((-1, 1))
                           for norm_fitness_t2 in norm_fitness_list_t2]

    fe_obj_list_t2 = [np.array([ind.obj for ind in da_t2])[:, 1].reshape((-1, 1)) for da_t2 in da_list_t2]
    obj_list_t2 = [np.hstack((quality_obj_t2, fe_obj_t2)) for quality_obj_t2, fe_obj_t2 in
                   zip(quality_obj_list_t2, fe_obj_list_t2)]

    hv_t1 = np.mean(np.array([hv.do(obj_t1) for obj_t1 in obj_list_t1]))
    hv_t2 = np.mean(np.array([hv.do(obj_t2) for obj_t2 in obj_list_t2]))
    return hv_t1, hv_t2

