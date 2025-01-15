import numpy as np
from copy import deepcopy
from util.util import non_dominated_sort
from scipy.spatial.distance import pdist


def update_ca(ca, new_pop, ca_size):
    kappa = 0.05
    total_pop = deepcopy(ca + new_pop)
    i_eps = np.zeros((len(total_pop), len(total_pop)))
    choose_index = list(range(len(total_pop)))

    if len(total_pop) <= ca_size:
        return total_pop

    pop_obj = np.array([ind.obj for ind in total_pop])
    # print(pop_obj)
    pop_obj = ((pop_obj - pop_obj.min(axis=0)) /
               (pop_obj.max(axis=0) - pop_obj.min(axis=0)))

    for i in range(i_eps.shape[0]):
        for j in range(i_eps.shape[1]):
            i_eps[i][j] = np.max(pop_obj[i, :] - pop_obj[j, :])

    c = np.max(np.abs(i_eps), axis=0)
    f = np.sum(-np.exp(-i_eps / (c * kappa)), 0) + 1

    while len(choose_index) > ca_size:
        min_index = np.argmin(f[choose_index])
        f = f + np.exp(-i_eps[choose_index[min_index], :] / (c[choose_index[min_index]] * kappa))
        choose_index.pop(min_index)

    ca_new = [total_pop[i] for i in choose_index]
    ca_obj = pop_obj[choose_index, :]
    for s, obj_norm in zip(ca_new, ca_obj):
        s.obj = obj_norm
    return ca_new


def update_cca(pop_dec, pop_pre_obj, max_size):
    pop_num = len(pop_dec)
    if pop_num <= max_size:
        return pop_dec, pop_pre_obj

    print(pop_pre_obj)
    pop_obj_norm = ((pop_pre_obj - pop_pre_obj.min(axis=0)) /
                    (pop_pre_obj.max(axis=0) - pop_pre_obj.min(axis=0)))

    i_eps = np.zeros((pop_num, pop_num))
    for i in range(pop_num):
        for j in range(pop_num):
            i_eps[i, j] = np.max(pop_obj_norm[i, :] - pop_obj_norm[j, :])

    c = np.max(np.abs(i_eps), axis=0)
    f = np.sum(-np.exp(-i_eps / (c * 0.05)), axis=0) + 1

    choose_index = list(range(pop_num))
    while len(choose_index) > max_size:
        min_index = np.argmin(f[choose_index])
        f = f + np.exp(-i_eps[choose_index[min_index], :] / (c[choose_index[min_index]] * 0.05))
        choose_index.pop(min_index)

    cca_dec = pop_dec[choose_index, :]
    cca_pre_obj = pop_pre_obj[choose_index, :]

    return cca_dec, cca_pre_obj


def update_da(da, new_pop, da_size, p):
    # find the non-dominated solution
    total_pop = deepcopy(da + new_pop)
    total_pop_obj = np.array([s.obj for s in total_pop])

    nd_index = non_dominated_sort(total_pop_obj, 1)
    nd_pop = [total_pop[i] for i, flag in enumerate(nd_index) if flag == 1]

    nd_num = len(nd_pop)
    if nd_num <= da_size:
        return nd_pop
    nd_obj = [s.obj for s in nd_pop]

    # select the extreme solution first
    choose = np.zeros(nd_num, dtype=bool)
    choose[np.argmin(nd_obj, axis=0)] = True
    choose[np.argmax(nd_obj, axis=0)] = True

    if np.sum(choose) > da_size:
        chosen_index = np.where(choose)[0]
        k = np.random.choice(chosen_index, size=(np.sum(choose) - da_size), replace=False)
        choose[k] = False
    elif np.sum(choose) < da_size:
        distance = np.full((nd_num, nd_num), np.inf)
        for i in range(nd_num - 1):
            for j in range(i + 1, nd_num):
                distance[i, j] = pdist([nd_obj[i], nd_obj[j]], 'minkowski', p=p)[0]
                distance[j, i] = distance[i, j]
        while np.sum(choose) < da_size:
            remain_index = np.where(~choose)[0]
            chosen_index = np.where(choose)[0]

            x = np.argmax(np.min(distance[remain_index.reshape(-1, 1), chosen_index.reshape(1, -1)], axis=1))
            choose[remain_index[x]] = True

    nd_pop_new = [nd_pop[i] for i in range(nd_num) if choose[i]]
    return nd_pop_new


def update_cda(pop_dec, pop_pre_obj, davar, max_size, p):
    print(pop_pre_obj)
    pop_obj_norm = ((pop_pre_obj - np.min(pop_pre_obj, axis=0)) /
                    (np.max(pop_pre_obj, axis=0) - np.min(pop_pre_obj, axis=0)))

    # Find the non-dominated solutions
    nd_index = non_dominated_sort(pop_pre_obj, 1)

    nd_pop_dec = pop_dec[nd_index == 1, :]
    nd_pop_obj = pop_pre_obj[nd_index == 1, :]
    nd_pop_obj_norm = pop_obj_norm[nd_index == 1, :]
    nd_davar = davar[nd_index == 1, :]

    nd_num = len(nd_pop_dec)
    if nd_num <= max_size:
        return nd_pop_dec, nd_pop_obj, nd_davar

    # Select the extreme solutions first
    choose = np.zeros(nd_num, dtype=bool)

    select = np.random.permutation(nd_pop_obj_norm.shape[1])
    choose[select[0]] = True

    if np.sum(choose) > max_size:
        chosen_index = np.where(choose)[0]
        k = np.random.choice(chosen_index, size=(np.sum(choose) - max_size), replace=False)
        choose[k] = False
    elif np.sum(choose) < max_size:
        distance = np.full((nd_num, nd_num), np.inf)
        for i in range(nd_num - 1):
            for j in range(i + 1, nd_num):
                distance[i, j] = pdist([nd_pop_obj_norm[i], nd_pop_obj_norm[j]], 'minkowski', p=p)[0]
                distance[j, i] = distance[i, j]
        while np.sum(choose) < max_size:
            remain_index = np.where(~choose)[0]
            chosen_index = np.where(choose)[0]

            x = np.argmax(np.min(distance[remain_index.reshape(-1, 1), chosen_index.reshape(1, -1)], axis=1))
            choose[remain_index[x]] = True

    cda_dec = nd_pop_dec[choose, :]
    cda_obj = nd_pop_obj[choose, :]
    cda_var = nd_davar[choose, :]

    return cda_dec, cda_obj, cda_var
