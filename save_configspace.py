from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.read_and_write import json as cs_json


def save_configspace(algorithm_name, cs):
    cs_string = cs_json.write(cs)
    with open(f"./configspace/{algorithm_name}.json", 'w') as f:
        f.write(cs_string)


def create_ga_configspace():
    cs = ConfigurationSpace()
    eta_c = UniformFloatHyperparameter(name='eta_c', lower=1, upper=100)
    eta_m = UniformFloatHyperparameter(name='eta_m', lower=1, upper=100)
    # mutpb = UniformFloatHyperparameter(name='mutate_prob', lower=0.01, upper=0.1)
    pop_size = UniformIntegerHyperparameter(name='pop_num', lower=10, upper=500)
    cs.add_hyperparameters([eta_c, eta_m, pop_size])
    return cs


def create_de_configspace():
    cs = ConfigurationSpace()
    cr = UniformFloatHyperparameter(name='cr', lower=0, upper=1)
    f = UniformFloatHyperparameter(name='f', lower=0, upper=2)
    pop_size = UniformIntegerHyperparameter(name='pop_num', lower=10, upper=500)
    cs.add_hyperparameters([cr, f, pop_size])
    return cs


def create_MadDE_configspace():
    cs = ConfigurationSpace()
    # default p_qbx = 0.01, p = 0.18, a_rate = 2.3, hm = 10, npm = 2, f0 = 0.2, cr0 = 0.2
    p_qbx = UniformFloatHyperparameter(name='p_qbx', lower=0.01, upper=0.05)
    p = UniformFloatHyperparameter(name='p', lower=0.05, upper=0.25)
    a_rate = UniformFloatHyperparameter(name='a_rate', lower=1, upper=3)
    hm = UniformFloatHyperparameter(name='hm', lower=1, upper=10)
    npm = UniformFloatHyperparameter(name='npm', lower=1, upper=5)
    f0 = UniformFloatHyperparameter(name='f0', lower=0.1, upper=0.9)
    cr0 = UniformFloatHyperparameter(name='cr0', lower=0.1, upper=0.9)

    cs.add_hyperparameters([p_qbx, p, a_rate, hm, npm, f0, cr0])
    return cs


if __name__ == "__main__":
    algorithm_names = ["DE", "GA", "MadDE"]
    configspace_list = [create_de_configspace(), create_ga_configspace(), create_MadDE_configspace()]

    for name, configsapce in zip(algorithm_names, configspace_list):
        save_configspace(name, configsapce)
