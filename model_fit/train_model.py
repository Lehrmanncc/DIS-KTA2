import warnings
import numpy as np
from smt.surrogate_models import KRG


def train_model(obj_num, train_x, train_y, theta):
    model_list = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(obj_num):
            model = KRG(poly='constant', corr='squar_exp', theta0=theta[i, :], theta_bounds=[1e-5, 20],
                        print_global=False)
            model.set_training_values(train_x, train_y[:, i])
            model.train()

            theta[i, :] = model.optimal_theta
            model_list.append(model)

    return model_list, theta
