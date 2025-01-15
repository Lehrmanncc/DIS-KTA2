from scipy.spatial.distance import cdist
from history.individual import Individual
from history.history import ConfigHistory
from model_fit.train_model import *
from util.util import mating_selection
from util.util import cross_mutation
from multi_process import run_target_algorithm_multiprocess
from util.util import calculate_da_hv
from util.util import remove_similar_rows
from util.util import LHDesign
from update.update import *
from sampling.adaptive_sampling import adaptive_sampling

from tqdm import tqdm
from copy import deepcopy
import ConfigSpace
from random import choice, seed
import logging
from pymoo.indicators.hv import Hypervolume


def dis_kta2(algorithm_name, run_id, config_space, fun_ids, train_instances_num=5, multi_instance=True,
             max_budget=250, init_budget=20, pop_size=20, phi=0.1, w_max=10, mu=1, eps=1e-4, ins_dim=10,
             fix_dim=True, benchmark="BBOB"):
    pbar = tqdm(desc="{}-th run".format(run_id), total=max_budget)
    init_configs = LHDesign(config_space, init_budget, run_id)

    sum_train_instances_num = len(fun_ids) * train_instances_num
    random_indices = [choice(range(sum_train_instances_num))]

    init_fitness, init_fes, budget = run_target_algorithm_multiprocess(run_id, algorithm_name, init_configs, fun_ids,
                                                                       random_indices, train_instances_num, None,
                                                                       mode="train", ins_dim=ins_dim, fix_dim=fix_dim,
                                                                       benchmark=benchmark)
    use_budget = budget
    use_instance_budget = budget
    pbar.update(use_budget)

    init_fitness_pad = np.pad(init_fitness, ((0, 0), (0, sum_train_instances_num - init_fitness.shape[1])),
                              mode='constant', constant_values=np.nan)
    init_fes_pad = np.pad(init_fes, ((0, 0), (0, sum_train_instances_num - init_fes.shape[1])),
                          mode='constant', constant_values=np.nan)

    configs_history = ConfigHistory(init_configs, init_fitness_pad, init_fes_pad, multi_instance)

    train_x = np.array([config.get_array() for config in configs_history.configs])
    train_y = configs_history.configs_obj
    n_var, n_obj = train_x.shape[1], train_y.shape[1]

    init_arch = [Individual(config, obj, fit_pad, fes_pad)
                 for config, obj, fit_pad, fes_pad in zip(train_x, train_y, init_fitness_pad, init_fes_pad)]
    ca = update_ca([], new_pop=init_arch, ca_size=pop_size)
    da = deepcopy(init_arch)

    p = 1 / n_obj
    theta = 5 * np.ones((n_obj, n_var))

    sampling_method = []
    uncertain_flag = 0

    run_configs = deepcopy(init_configs)
    run_configs_obj = deepcopy(configs_history.configs_obj)
    chose_indices = deepcopy(random_indices)

    t = -1
    da_history = []
    da_window_list = [da]
    da_history_length = 0
    instance_budget_time = []
    use_instance_budget_list = []
    ref_point = np.array([1, 1])
    hv = Hypervolume(ref_point)

    while use_budget < max_budget:

        da_fitness = np.array([ind.fitness for ind in da])
        da_fitness_norm = configs_history.normalize(da_fitness)
        da_fes = np.array([ind.fes for ind in da])

        instance_obj = [np.hstack((da_fitness_norm[:, i].reshape(-1, 1), da_fes[:, i].reshape(-1, 1)))
                        for i in range(sum_train_instances_num)]
        instance_hv = [hv.do(obj) for obj in instance_obj]
        instance_hv_dict = dict(zip(chose_indices, instance_hv))
        format_dict = {key: f"{value:.4f}" for key, value in instance_hv_dict.items()}

        da_obj = np.array([ind.obj for ind in da])
        logging.info(f'budget={use_budget}, {format_dict},  da_hv={hv.do(da_obj):.4f},  da_num:{len(da_obj)}, '
                     f'change_key:{configs_history.change_key}')
        model_list, theta = train_model(n_obj, train_x, train_y, theta)

        cca_random_configs = config_space.sample_configuration(100 - len(ca))
        cda_random_configs = config_space.sample_configuration(100 - len(da))

        cca_random_array = np.array([config.get_array() for config in cca_random_configs])
        cda_random_array = np.array([config.get_array() for config in cda_random_configs])

        cca_random_obj = np.zeros((100 - len(ca), n_obj))
        cda_random_obj = np.zeros((100 - len(da), n_obj))
        for i in range(len(cca_random_array)):
            for j in range(n_obj):
                cca_random_obj[i, j] = model_list[j].predict_values(cca_random_array[i, :].reshape((1, -1)))

        for i in range(len(cda_random_array)):
            for j in range(n_obj):
                cda_random_obj[i, j] = model_list[j].predict_values(cda_random_array[i, :].reshape((1, -1)))

        cca_obj = np.vstack((np.array([s.obj for s in ca]), cca_random_obj))
        cca_dec = np.vstack((np.array([s.x for s in ca]), cca_random_array))
        cda_obj = np.vstack((np.array([s.obj for s in da]), cda_random_obj))
        cda_dec = np.vstack((np.array([s.x for s in da]), cda_random_array))

        for _ in range(w_max):
            parent_c_dec, parent_m_dec = mating_selection(cca_dec, cca_obj, cda_dec, cda_obj, 100)
            offspring1_dec = cross_mutation(parent_c_dec, config_space, pc=1, pm=0, yita1=20, yita2=0)
            offspring2_dec = cross_mutation(parent_m_dec, config_space, pc=0, pm=1, yita1=0, yita2=20)

            sum_pop_dec = np.vstack((cda_dec, cca_dec, offspring1_dec, offspring2_dec))
            sum_pop_dec = remove_similar_rows(sum_pop_dec, 1e-6)

            pre_pop_obj = np.zeros((sum_pop_dec.shape[0], n_obj))
            var = np.zeros((sum_pop_dec.shape[0], n_obj))

            for i in range(sum_pop_dec.shape[0]):
                for j in range(n_obj):
                    pre_pop_obj[i, j] = model_list[j].predict_values(sum_pop_dec[i, :].reshape((1, -1)))
                    var[i, j] = model_list[j].predict_variances(sum_pop_dec[i, :].reshape((1, -1)))

            cca_dec, cca_obj = update_cca(sum_pop_dec, pre_pop_obj, 100)
            cda_dec, cda_obj, cda_var = update_cda(sum_pop_dec, pre_pop_obj, var, 100, p)

        new_pop_dec, sampling_str = adaptive_sampling(cca_obj, cca_dec, cda_obj, cda_dec, cda_var, da,
                                                      mu, p, phi, uncertain_flag)
        sampling_method.append(sampling_str)
        print(sampling_str)

        new_pop_dec, _ = np.unique(new_pop_dec, axis=0, return_index=True)
        pop_filter = []
        for i in range(new_pop_dec.shape[0]):
            dist = cdist(np.real(new_pop_dec[i, :].reshape((1, -1))), np.real(train_x), 'euclidean')
            if np.min(dist) > 1e-5:
                pop_filter.append(new_pop_dec[i, :])

        new_configs_dec = np.array(pop_filter)
        if len(new_configs_dec) > 0:

            new_configs = [ConfigSpace.Configuration(config_space, vector=dec) for dec in new_configs_dec]
            new_fitness, new_fes, budget = run_target_algorithm_multiprocess(run_id, algorithm_name, new_configs,
                                                                             fun_ids,
                                                                             chose_indices, train_instances_num, None,
                                                                             mode="train", ins_dim=ins_dim,
                                                                             fix_dim=fix_dim,
                                                                             benchmark=benchmark)
            use_budget += budget
            use_instance_budget += budget
            pbar.update(budget)

            new_fitness_pad = np.pad(new_fitness, ((0, 0), (0, sum_train_instances_num - new_fitness.shape[1])),
                                     mode='constant', constant_values=np.nan)
            new_fes_pad = np.pad(new_fes, ((0, 0), (0, sum_train_instances_num - new_fes.shape[1])),
                                 mode='constant', constant_values=np.nan)
            configs_history.add(new_configs, new_fitness_pad, new_fes_pad, multi_instance)
            new_configs_obj = configs_history.new_configs_obj
            offspring = [Individual(config, obj, fitness, fes) for config, obj, fitness, fes in
                         zip(new_configs_dec, new_configs_obj, new_fitness_pad, new_fes_pad)]

            run_configs = run_configs + new_configs
            run_configs_obj = np.vstack((run_configs_obj, new_configs_obj))

            train_x = np.array([config.get_array() for config in run_configs])
            train_y = run_configs_obj

            if configs_history.change_key:
                ca = configs_history.update_obj(ca)
                da = configs_history.update_obj(da)

            ca = update_ca(ca, offspring, pop_size)
            da = update_da(da, offspring, pop_size, p)

            da_window_list.append(da)
            da_history_length = len(da_history)

            if len(da_window_list) % 3 == 0:
                da_history.append(da_window_list)
                da_window_list = []
                t += 1
        else:
            print("This round of sampling is empty!")

        if t > 0 and da_history_length != len(da_history):
            hv_t1, hv_t2 = calculate_da_hv(da_history[t - 1], da_history[t], configs_history)
            diff_hv = np.abs(hv_t1 - hv_t2) / np.abs(hv_t1)
            if diff_hv < eps and (len(chose_indices) < sum_train_instances_num):
                use_instance_budget_list.append(use_instance_budget)
                use_instance_budget = 0
                remain_indices = [i for i in range(sum_train_instances_num) if i not in chose_indices]

                seed(run_id)
                add_instance_idx = choice(remain_indices)
                chose_indices.append(add_instance_idx)

                add_instances_ith = len(chose_indices) - 1

                run_configs, run_configs_index = configs_history.run_instance_configs(ca, da)
                instances_fitness, instances_fes, budget = run_target_algorithm_multiprocess(run_id, algorithm_name,
                                                                                             run_configs,
                                                                                             fun_ids,
                                                                                             [add_instance_idx],
                                                                                             train_instances_num,
                                                                                             None,
                                                                                             mode="train",
                                                                                             ins_dim=ins_dim,
                                                                                             fix_dim=fix_dim,
                                                                                             benchmark=benchmark)
                instance_budget_time.append(use_budget)
                use_budget += budget
                pbar.update(budget)
                use_instance_budget += budget

                configs_history.change_instance_value(run_configs_index, instances_fitness,
                                                      instances_fes, add_instances_ith)
                run_configs_obj = configs_history.configs_obj[run_configs_index, :]

                ca = configs_history.update_obj(ca)
                da = configs_history.update_obj(da)
                ca = update_ca(ca, [], pop_size)
                da = update_da(da, [], pop_size, p)

                train_x = np.array([config.get_array() for config in run_configs])
                train_y = run_configs_obj

                da_history = []
                da_window_list = [da]
                t = -1

    return da, configs_history, sampling_method, da_history, chose_indices
