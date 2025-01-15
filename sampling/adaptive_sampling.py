import numpy as np
from scipy.spatial.distance import pdist
from util.util import cal_convergence
from util.util import pure_diversity
import logging


def adaptive_sampling(cca_obj, cca_dec, cda_obj, cda_dec, davar, da, mu, p, phi, uncertain_flag=0):
    da_obj = np.array([s.obj for s in da])
    da_dec = np.array([s.x for s in da])
    ideal_point = np.min(np.vstack((cca_obj, cda_obj)), axis=0)
    flag = cal_convergence(cca_obj, cda_obj, ideal_point)

    if flag == 1 and uncertain_flag == 0:
        # convergence sampling strategy
        cca_num = cca_obj.shape[0]
        cca_obj_norm = (cca_obj - np.min(cca_obj, axis=0)) / (np.max(cca_obj, axis=0) - np.min(cca_obj, axis=0))

        i_eps = np.zeros((cca_num, cca_num))
        for i in range(cca_num):
            for j in range(cca_num):
                i_eps[i, j] = np.max(cca_obj_norm[i, :] - cca_obj_norm[j, :])

        c = np.max(np.abs(i_eps), axis=0)
        f = np.sum(-np.exp(-i_eps / (c * 0.05)), axis=0) + 1

        choose = list(range(cca_num))
        while len(choose) > mu:
            min_index = np.argmin(f[choose])
            f = f + np.exp(-i_eps[choose[min_index], :] / (c[choose[min_index]] * 0.05))
            choose.pop(min_index)

        offspring_dec = cca_dec[choose, :]
        return offspring_dec, "convergence sampling"
    else:
        if (pure_diversity(cda_obj) < pure_diversity(da_obj)) or uncertain_flag == 1:
            # uncertainty sampling strategy
            cda_num = davar.shape[0]
            choose = np.zeros(mu)
            for i in range(mu):
                cda_random_index = np.random.permutation(cda_num)
                uncertainty = np.mean(davar[cda_random_index[0: int(np.ceil(phi * cda_num))], :], axis=1)
                best = np.argmax(uncertainty)
                choose[i] = cda_random_index[best]

            offspring_dec = cda_dec[choose.astype('int64'), :]
            return offspring_dec, "uncertainty sampling"
        else:
            # diversity sampling strategy
            pop_dec = np.vstack((da_dec, cda_dec))
            pop_obj = np.vstack((da_obj, cda_obj))
            pop_obj_norm = ((pop_obj - np.min(pop_obj, axis=0)) /
                            (np.max(pop_obj, axis=0) - np.min(pop_obj, axis=0)))

            da_num = da_obj.shape[0]
            pop_num = pop_obj_norm.shape[0]

            distance = np.full((pop_num, pop_num), np.inf)
            for i in range(pop_num - 1):
                for j in range(i + 1, pop_num):
                    distance[i, j] = pdist([pop_obj_norm[i], pop_obj_norm[j]], 'minkowski', p=p)[0]
                    distance[j, i] = distance[i, j]

            choose = np.zeros(pop_num, dtype=bool)
            choose[:da_num] = True
            max_size = da_num + mu
            offspring_dec = []
            while np.sum(choose) < max_size:
                remain_index = np.where(~choose)[0]
                chosen_index = np.where(choose)[0]
                x = np.argmax(np.min(distance[remain_index.reshape(-1, 1), chosen_index.reshape(1, -1)], axis=1))
                choose[remain_index[x]] = True
                offspring_dec.append(pop_dec[remain_index[x], :])

            offspring_dec = np.array(offspring_dec)

            return offspring_dec, "diversity sampling"
