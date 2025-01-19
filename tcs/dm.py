import copy
import numpy as np
from KPI import InitializeVector, identificationCurvature


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


def find_tradeoff_config(best_configs, configs_history):

    configs_array = np.array([config.get_array() for config in best_configs])
    configs_array_norm = np.array([config.get_array() for config in configs_history.configs])
    arch_index = np.where((configs_array[:, None, :] == configs_array_norm).all(axis=-1))[1]
    arch_obj = configs_history.configs_obj[arch_index, :]

    outliers_idx = find_outliers_idx(configs_array)

    filter_idx = np.setdiff1d(np.arange(len(arch_obj)), outliers_idx)
    filter_points = arch_obj[filter_idx, :]

    if len(filter_points) > 2:
        knee_idx, knee_points = find_knee_points(filter_points, arch_obj)
        tradeoff_idx = find_preference_point(knee_points, arch_obj, 1)[0]
        tradeoff_config = best_configs[tradeoff_idx]

    elif len(filter_points) == 2:
        tradeoff_idx = find_preference_point(filter_points, arch_obj, 1)[0]
        tradeoff_config = best_configs[tradeoff_idx]

    elif len(filter_points) == 1:
        tradeoff_idx = filter_idx[0]
        tradeoff_config = best_configs[tradeoff_idx]
    else:
        print("Error !")

    return tradeoff_config