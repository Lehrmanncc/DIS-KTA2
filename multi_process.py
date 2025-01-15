from setup import get_train_instance, get_test_instance
from ta.search_algorithm import create_algorithm
import functools
import numpy as np
import multiprocessing


def worker(seed, exp_seed, ta, fun_ids, instance_idx, train_ins_num, test_ins_num, mode, ins_dim, fix_dim, benchmark):
    if mode == "train":
        run_instance = get_train_instance(exp_seed, fun_ids, instance_idx, train_ins_num, ins_dim, fix_dim, benchmark)
    elif mode == "test":
        run_instance = get_test_instance(exp_seed, fun_ids, instance_idx, test_ins_num, ins_dim, fix_dim, benchmark)
    else:
        raise ValueError('mode error!')
    fitness, fes = ta.evolve(seed, run_instance)
    return fitness, fes


def cal_configuration_score(exp_seed, algorithm_name, run_num, fun_ids, instance_idx,
                            train_ins_num, test_ins_num, config, mode,
                            ins_dim, fix_dim, benchmark):
    kwargs = dict(config)
    ta = create_algorithm(algorithm_name, **kwargs)

    pool = multiprocessing.Pool()
    worker_partial = functools.partial(worker, exp_seed=exp_seed, ta=ta, fun_ids=fun_ids, instance_idx=instance_idx,
                                       train_ins_num=train_ins_num, test_ins_num=test_ins_num,
                                       mode=mode, ins_dim=ins_dim, fix_dim=fix_dim, benchmark=benchmark)
    sum_score = pool.map(worker_partial, range(run_num))
    pool.close()
    pool.join()

    mean_fitness = np.mean(np.array(sum_score)[:, 0])
    mean_fes_score = np.mean(np.array(sum_score)[:, 1])

    return mean_fitness, mean_fes_score


def config_worker(exp_seed, algorithm_name, config_idx, fun_ids, instances_idx, train_ins_num, test_ins_num, config,
                  configs_fitness, configs_fes, calling_count, mode="train",
                  ins_dim=10, fix_dim=True, benchmark="BBOB"):
    for j, idx in enumerate(instances_idx):
        fitness, fes = cal_configuration_score(exp_seed, algorithm_name, 10, fun_ids, idx,
                                               train_ins_num, test_ins_num, config, mode,
                                               ins_dim, fix_dim, benchmark)

        idx = config_idx * len(instances_idx) + j

        configs_fitness[idx] = fitness
        configs_fes[idx] = fes
        calling_count.value += 1


def run_target_algorithm_multiprocess(exp_seed, algorithm_name, configs, fun_ids, instances_idx,
                                      train_ins_num=None, test_ins_num=None, mode="train",
                                      ins_dim=10, fix_dim=True, benchmark="BBOB"):
    print("run instances idx: {}".format(instances_idx))
    configs_fitness = multiprocessing.Array('d', len(configs) * len(instances_idx))
    configs_fes = multiprocessing.Array('d', len(configs) * len(instances_idx))
    calling_count = multiprocessing.Value('i', 0)
    max_processes = 16

    running_processes = []
    for i, config in enumerate(configs):
        formatted_strings = [f"{key}: {value:.4f}" for key, value in dict(config).items()]
        value_str = ', '.join(formatted_strings)
        print("Configuration {}: ".format(i) + value_str)

        process = multiprocessing.Process(target=config_worker,
                                          args=(exp_seed, algorithm_name, i, fun_ids, instances_idx, train_ins_num, test_ins_num,
                                                config, configs_fitness, configs_fes, calling_count, mode,
                                                ins_dim, fix_dim, benchmark))
        running_processes.append(process)
        if len(running_processes) >= max_processes:
            running_processes[0].start()
            running_processes[0].join()
            running_processes.pop(0)

    for process in running_processes:
        process.start()

    for process in running_processes:
        process.join()

    return np.array(configs_fitness).reshape(len(configs), len(instances_idx)), \
        np.array(configs_fes).reshape(len(configs), len(instances_idx)), \
        calling_count.value

