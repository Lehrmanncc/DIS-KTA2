import os
import ioh
import time
import ConfigSpace
import logging

from util.util import mk_dir
from setup import get_configspace
from algorithm.DIS_KTA2 import dis_kta2


if __name__ == "__main__":

    # low_opt = "GA"
    low_opt = "MadDE"

    abs_folder = os.getcwd()
    res_save_dir = os.path.join(abs_folder, "result", low_opt)
    mk_dir(res_save_dir)

    max_budget = 500
    init_budget = 50

    instance_dim = 10
    bbob_problems = ioh.ProblemClass.BBOB.problems

    train_func_key = [1, 2, 3, 4, 5]
    train_func_name = [bbob_problems[key] for key in train_func_key]
    train_num = 2

    cs = get_configspace(low_opt)
    start_time = time.perf_counter()

    logging.basicConfig(filename=f'{res_save_dir}/optimize.log', level=logging.INFO, format='%(message)s')

    (best_pop, configs_history, sampling_method,
     da_history, run_instances_idx) = dis_kta2(low_opt, 0, cs, train_func_name, train_num,
                                               True, max_budget, init_budget,
                                               20, mu=2, eps=0.01, w_max=10, ins_dim=instance_dim)

    run_time = time.perf_counter() - start_time

    best_configs = [ConfigSpace.Configuration(cs, vector=ind.x) for ind in best_pop]
    print(f"best configs:{best_configs}, cost time:{run_time}")
