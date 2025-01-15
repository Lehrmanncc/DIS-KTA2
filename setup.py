from ConfigSpace.read_and_write import json as cs_json
from optproblems.cec2005 import F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12
from ioh import get_problem, ProblemClass
import numpy as np


def get_configspace(algorithm_name):
    with open(f'../../configspace/{algorithm_name}.json', 'r') as f:
        json_string = f.read()
        configspace = cs_json.read(json_string)

    return configspace


def get_train_instance(exp_seed, fun_ids, ins_idx, train_instances_num, ins_dim, fix_dim, benchmark="BBOB"):
    np.random.seed(seed=exp_seed)
    train_instances = []
    if benchmark == "CEC":
        if train_instances_num == 1:
            train_instances = [F3(10)]

        elif train_instances_num == 3:
            train_instances = [F1(10), F4(10), F10(10)]

        elif train_instances_num == 5:
            train_instances = [F1(10), F3(10), F4(10), F6(10), F10(10)]

        elif train_instances_num == 7:
            train_instances = [F1(10), F3(10), F4(10), F5(10), F6(10), F8(10), F10(10)]

        elif train_instances_num == 9:
            train_instances = [F1(10), F3(10), F4(10), F5(10), F6(10), F8(10), F10(10), F11(10), F12(10)]

        else:
            print("train instances size error !")

    elif benchmark == "BBOB":
        if train_instances_num is None:
            raise ValueError("Train mode requires train_instances_num parameter.")
        else:
            if fix_dim:
                for fid in fun_ids:
                    fun_instances = [get_problem(fid, i, ins_dim, ProblemClass.BBOB) for i in range(train_instances_num)]
                    train_instances += fun_instances
            else:
                for fid in fun_ids:
                    dim = np.random.randint(low=2, high=ins_dim+1)
                    fun_instances = [get_problem(fid, i, dim, ProblemClass.BBOB) for i in range(train_instances_num)]
                    train_instances += fun_instances
    else:
        print("benchmark name error!")
    run_instance = train_instances[ins_idx]

    return run_instance


def get_test_instance(exp_seed, test_func_ids, ins_idx, test_instances_num, ins_dim, fix_dim, benchmark="BBOB"):
    np.random.seed(seed=exp_seed)
    test_instances = []
    if benchmark == "CEC":
        test_instances = [F2(10), F9(10)]

    elif benchmark == "BBOB":
        if test_instances_num is None:
            raise ValueError("Test mode requires test_instances_num parameter.")
        else:
            if fix_dim:
                for fid in test_func_ids:
                    fun_instances = [get_problem(fid, i+10, ins_dim, ProblemClass.BBOB)
                                     for i in range(test_instances_num)]
                    test_instances += fun_instances
            else:
                for fid in test_func_ids:
                    dim = np.random.randint(low=2, high=ins_dim+1)
                    fun_instances = [get_problem(fid, i, dim, ProblemClass.BBOB) for i in range(test_instances_num)]
                    test_instances += fun_instances
    else:
        print("benchmark name error!")
    run_instance = test_instances[ins_idx]
    return run_instance


def get_train_instances_id(fun_ids, train_instances_num, benchmark="BBOB"):
    train_instances = []
    if benchmark == "BBOB":
        if train_instances_num is None:
            raise ValueError("Train mode requires test_instances_num parameter.")
        else:
            for fid in fun_ids:
                fun_instance = [get_problem(fid, i, 10, ProblemClass.BBOB) for i in range(train_instances_num)]
                train_instances += fun_instance
    else:
        print("benchmark name error!")

    instance_name = []
    for instance in train_instances:
        instance_name.append(f"F{instance.meta_data.problem_id}-{instance.meta_data.instance}")
    return instance_name


def get_test_instances_id(fun_ids, test_instances_num, benchmark="BBOB"):
    test_instances = []
    if benchmark == "BBOB":
        if test_instances_num is None:
            raise ValueError("Test mode requires test_instances_num parameter.")
        else:
            for fid in fun_ids:
                fun_instance = [get_problem(fid, i+10, 10, ProblemClass.BBOB) for i in range(test_instances_num)]
                test_instances += fun_instance
    else:
        print("benchmark name error!")

    instance_name = []
    for instance in test_instances:
        instance_name.append(("F{}".format(instance.meta_data.problem_id)))
    return instance_name


