import skopt
import copy
import numpy as np
import matplotlib.pyplot as plt
from KPI import InitializeVector, identificationCurvature
from util.util import mk_dir
import os
from tqdm import tqdm
from indicator.HV import sign_test
from indicator.NDR import find_nd_points, cal_ndr


def calculate_test_obj(res):
    test_min_max = np.array([np.min(res.test_fitness, axis=0), np.max(res.test_fitness, axis=0)])
    quality_obj = np.mean((res.test_fitness - test_min_max[0, :])
                          / (test_min_max[1, :] - test_min_max[0, :]), axis=1).reshape((-1, 1))
    fe_score = np.mean(res.test_fes, axis=1).reshape((-1, 1))

    res_test_obj = np.hstack((quality_obj, fe_score))
    return res_test_obj


def calculate_instance_test_obj(res):
    test_res_instance_obj = []
    for i in range(res.test_fitness.shape[1]):
        test_instance_fitness_norm = ((res.test_fitness[:, i] - np.min(res.test_fitness[:, i], axis=0))
                                      / (np.max(res.test_fitness[:, i], axis=0) - np.min(res.test_fitness[:, i],
                                                                                         axis=0)))
        test_instance_obj = np.hstack((test_instance_fitness_norm.reshape(-1, 1), res.test_fes[:, i].reshape(-1, 1)))
        test_res_instance_obj.append(test_instance_obj)

    return test_res_instance_obj


def find_outliers_idx(configs_array):
    q1, q3 = np.percentile(configs_array, 25, axis=0), np.percentile(configs_array, 75, axis=0)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    outliers_idx = np.where((configs_array > upper) | (configs_array < lower))[0]
    unique_outliers_idx = list(set(outliers_idx))
    return unique_outliers_idx


def find_pareto_endpoints(pf):
    sorted_pf = sorted(pf, key=lambda x: (x[0], x[1]))

    endpoint1 = sorted_pf[0]
    endpoint2 = sorted_pf[-1]

    return endpoint1, endpoint2


def find_prefer_point(configs, idx, points):
    if len(points) > 1:
        norm_points = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0))
        diff = np.abs(norm_points[:, 0] - norm_points[:, 1])
        k = 1
        prefer_idx = np.argsort(diff)[:k]

        best_point = points[prefer_idx, :]
        best_config = configs[prefer_idx[0]]
        best_idx = idx[prefer_idx[0]]

    elif len(points) == 1:
        best_point = points
        best_config = configs
        best_idx = idx
    else:
        print("Error !")
    return best_idx, best_point, best_config


def find_preference_point(points, train_obj, k):
    weights = [0.5, 0.5]
    if len(points) > 1:
        norm_points = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0))
        asf = np.max(norm_points / weights, axis=1)
        local_idx = np.argsort(asf)[:k]
        prefer_point = points[local_idx, :]
        prefer_idx = np.where((prefer_point[:, None, :] == train_obj).all(axis=-1))[1]
    elif len(points) == 1:
        prefer_point = points
        prefer_idx = np.where((prefer_point[:, None, :] == train_obj).all(axis=-1))[1]
    else:
        print("Error !")
        prefer_idx = None
    return prefer_idx


def find_random_point(points, train_obj, k, seed):
    np.random.seed(seed)
    if len(points) > 1:
        random_points = points[np.random.choice(points.shape[0], k, replace=False)]
        random_idx = np.where((random_points[:, None, :] == train_obj).all(axis=-1))[1]
    elif len(points) == 1:
        random_point = points
        random_idx = np.where((random_point[:, None, :] == train_obj).all(axis=-1))[1]
    else:
        print("Error !")
        random_idx = None
    return random_idx


def find_knee_points(filter_data, train_obj):
    vsep = InitializeVector(2)
    vectors = np.array([vsep[i].vector.tolist() for i in range(len(vsep))])

    pf = copy.copy(filter_data)
    end_point1, end_point2 = find_pareto_endpoints(pf)

    ind0 = np.where(np.all(pf == end_point1, axis=1))[0]
    ind1 = np.where(np.all(pf == end_point2, axis=1))[0]
    endpoint_idx = np.unique(np.concatenate((ind0, ind1)))
    pf_new = np.delete(pf, endpoint_idx, axis=0)

    knee_idx, knee_points = identificationCurvature(pf_new, vectors)
    knee_idx = np.where((knee_points[:, None, :] == train_obj).all(axis=-1))[1]

    return knee_idx, knee_points


def find_knee_points_old(filter_data, train_obj, k):
    vsep = InitializeVector(2)
    vectors = np.array([vsep[i].vector.tolist() for i in range(len(vsep))])

    pf = copy.copy(filter_data)
    end_point1, end_point2 = find_pareto_endpoints(pf)

    ind0 = np.where(np.all(pf == end_point1, axis=1))[0]
    ind1 = np.where(np.all(pf == end_point2, axis=1))[0]
    endpoint_idx = np.unique(np.concatenate((ind0, ind1)))
    pf_new = np.delete(pf, endpoint_idx, axis=0)

    knee_idx, knee_points = identificationCurvature(pf_new, vectors, k)
    knee_idx = np.where((knee_points[:, None, :] == train_obj).all(axis=-1))[1]

    return knee_idx, knee_points


def analysis_kpi_select(file, num_experiments=5, outliers_filter=True, prefer_select=True, no_kpi=True):
    ins_ndr_lst = []
    for i in tqdm(range(num_experiments)):
        res = skopt.load(f"D:/Project_Code/AC_multi_obj/experiment/res/{file}/res{i}.pkl")
        configs_array = np.array([config.get_array() for config in res.best_configs])
        test_instance_obj_lst = calculate_instance_test_obj(res)

        ins_ndr = np.empty((1, len(test_instance_obj_lst)))
        for j, test_obj in enumerate(test_instance_obj_lst):
            test_nd_points, test_dominated_points, test_nd_idx = find_nd_points(test_obj)

            if outliers_filter:
                outliers_idx = find_outliers_idx(configs_array)
                filter_idx = np.setdiff1d(np.arange(len(res.arch_obj)), outliers_idx)
                filter_points = res.arch_obj[filter_idx, :]

                if len(filter_points) > 2:
                    knee_idx, knee_points = find_knee_points(filter_points, res.arch_obj)
                    ins_ndr[0][j] = cal_ndr(knee_idx, test_nd_idx)

                    if no_kpi:
                        random_idx = find_random_point(filter_points, res.arch_obj, len(knee_idx), i)
                        ins_ndr[0][j] = cal_ndr(random_idx, test_nd_idx)

                    if prefer_select:
                        prefer_idx = find_preference_point(filter_points, res.arch_obj, len(knee_idx))
                        ins_ndr[0][j] = cal_ndr(prefer_idx, test_nd_idx)
                else:
                    print("Error !")

            else:
                outliers_idx = find_outliers_idx(configs_array)
                filter_idx = np.setdiff1d(np.arange(len(res.arch_obj)), outliers_idx)
                filter_points = res.arch_obj[filter_idx, :]
                knee_idx_new, _ = find_knee_points(filter_points, res.arch_obj)

                filter_points = res.arch_obj[:, :]
                if len(filter_points) > 2:
                    knee_idx, knee_points = find_knee_points_old(filter_points, res.arch_obj, len(knee_idx_new))
                    ins_ndr[0][j] = cal_ndr(knee_idx, test_nd_idx)
                    if prefer_select:
                        prefer_idx = find_preference_point(filter_points, res.arch_obj, len(knee_idx_new))
                        ins_ndr[0][j] = cal_ndr(prefer_idx, test_nd_idx)

                else:
                    print("Error !")
        ins_ndr_lst.append(ins_ndr)
    return np.vstack(ins_ndr_lst)


def cal_mean_ndr(res_file):
    alg_flag = [[False, False, False], [True, False, True],
                [False, True, False], [True, False, False]]
    alg_ins_ndr_lst = []
    for i, flag in enumerate(alg_flag):
        ins_ndr_array = analysis_kpi_select(res_file, 30, flag[0], flag[1], flag[2])
        alg_ins_ndr_lst.append(ins_ndr_array)

    alg_mean_ndr, alg_std_ndr = [], []
    sign_lst = []
    alg_ins_ndr_array = np.transpose(np.asarray(alg_ins_ndr_lst), (2, 1, 0))
    for ndr_array in alg_ins_ndr_array:
        alg_mean_ndr.append(np.mean(ndr_array, axis=0))
        alg_std_ndr.append(np.std(ndr_array, axis=0))

        sign_lst.append([sign_test(ndr_array[:, i], ndr_array[:, -1], obj="max")
                         for i in range(ndr_array.shape[1] - 1)])

    alg_mean_ndr, alg_std_ndr = np.asarray(alg_mean_ndr), np.asarray(alg_std_ndr)
    sign_test_array = np.asarray(sign_lst)
    return alg_mean_ndr, alg_std_ndr, sign_test_array


def find_best_config(file, num_experiments=30):
    best_idx = None
    res_best_config, res_best_config_performance = [], []

    for i in range(num_experiments):
        res = skopt.load(f"D:/Project_Code/AC_multi_obj/experiment/res/{file}/res{i}.pkl")
        configs_array = np.array([config.get_array() for config in res.best_configs])
        outliers_idx = find_outliers_idx(configs_array)

        filter_idx = np.setdiff1d(np.arange(len(res.arch_obj)), outliers_idx)
        filter_points = res.arch_obj[filter_idx, :]

        if len(filter_points) > 2:
            knee_idx, knee_points = find_knee_points(filter_points, res.arch_obj)
            best_idx = find_preference_point(knee_points, res.arch_obj, 1)[0]
            best_config = res.best_configs[best_idx]

        elif len(filter_points) == 2:
            best_idx = find_preference_point(filter_points, res.arch_obj, 1)[0]
            best_config = res.best_configs[best_idx]

        elif len(filter_points) == 1:
            best_idx = filter_idx[0]
            best_config = res.best_configs[best_idx]
        else:
            print("Error !")

        best_config_performance = np.empty((res.test_fitness.shape[1], res.arch_obj.shape[1]))
        for j in range(res.test_fitness.shape[1]):
            test_fe = int(res.test_fes[best_idx, j] * (10000 - 500) + 500)
            best_config_performance[j, :] = np.asarray([res.test_fitness[best_idx, j], test_fe])

        res_best_config.append(best_config)
        res_best_config_performance.append(best_config_performance)
    return res_best_config, np.mean(np.asarray(res_best_config_performance), axis=0)


def get_so_test_instance_performance(file, num_experiments=30):
    res_best_config, res_best_config_performance = [], []
    for i in range(num_experiments):
        res = skopt.load(f"D:/Project_Code/AC_multi_obj/experiment/res/{file}/res{i}.pkl")
        best_config_performance = np.empty((res.test_fitness.shape[1], 2))
        for j in range(res.test_fitness.shape[1]):
            test_fe = int(res.test_fes[0, j] * (10000 - 500) + 500)
            best_config_performance[j, :] = np.asarray([res.test_fitness[0, j], test_fe])

        res_best_config.append(res.best_configs[0])
        res_best_config_performance.append(best_config_performance)
    return res_best_config, np.mean(np.asarray(res_best_config_performance), axis=0)


def plot_train_test_pf(exp_num, train_points, test_points, labels, markers, save_path, save_name, color):
    plt.style.use('default')
    plt.style.use(['science', 'no-latex', 'ieee'])
    fig, axs = plt.subplots(1, 2, figsize=(4, 2))  # 创建包含两个子图的画布

    for i, obj in enumerate(train_points):
        if len(obj) > 0:
            axs[0].scatter(obj[:, 0], obj[:, 1], label=labels[i], alpha=1,
                           marker=markers[i], color=color[i], s=50)

    for i, obj in enumerate(test_points):
        if len(obj) > 0:
            axs[1].scatter(obj[:, 0], obj[:, 1], label=labels[i], alpha=1,
                           marker=markers[i], color=color[i], s=50)

    axs[0].set_xlabel('$F_1$(Quality Score)')
    axs[0].set_ylabel('$F_2$(FEs score)')
    axs[0].set_title('{}-th: Train Instances'.format(exp_num))

    axs[1].set_xlabel('$F_1$(Quality Score)')
    axs[1].set_ylabel('$F_2$(FEs score)')
    axs[1].set_title('{}-th: Test Instances'.format(exp_num))

    axs[0].legend(fontsize=6)
    axs[1].legend(fontsize=6)

    plt.tight_layout()
    plt.savefig("{}/{}th-{}".format(save_path, exp_num, save_name), dpi=600)
    plt.show()


def plot_test_pf(test_points, labels, markers, save_path, save_name, color):
    plt.style.use('default')
    plt.style.use(['science', 'no-latex', 'ieee'])
    plt.figure(figsize=(2.4, 2))

    for i, obj in enumerate(test_points):
        if len(obj) > 0:
            plt.scatter(obj[:, 1], obj[:, 0], label=labels[i], alpha=1,
                        marker=markers[i], color=color[i], s=20)

    plt.xlabel('$f_1$(Efficiency Score)')
    plt.ylabel('$f_2$(Quality Score)')

    plt.legend(fontsize=6)

    plt.tight_layout()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig("{}/{}".format(save_path, save_name))
    plt.show()


def plot_instance_pf(test_points, labels, markers, save_path, instance_name, color):
    plt.style.use('default')
    plt.style.use(['science', 'no-latex', 'ieee'])
    plt.figure(figsize=(2.4, 2))

    for i, obj in enumerate(test_points):
        if len(obj) > 0:
            plt.scatter(obj[:, 1], obj[:, 0], label=labels[i], alpha=1,
                        marker=markers[i], c=color[i], s=20)

    plt.xlabel('$f_1$(Efficiency Score)')
    plt.ylabel('$f_2$(Quality Score)')

    plt.legend(fontsize=6)
    save_path = os.path.join("./experiment/res/", save_path)
    mk_dir(save_path)
    plt.tight_layout()
    plt.savefig("{}/{}.pdf".format(save_path, instance_name))
    plt.show()
