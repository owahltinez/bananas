''' Local Datasets Module '''

import os
from pathlib import Path
from ..dataset.dataset import DataSet

def load_dataset_csv(name: str, target_column: int = -1, random_seed: int = None):
    '''
    Parameters
    ----------
    TODO
    '''
    cwd = os.path.dirname(os.path.realpath(__file__))
    train = DataSet.from_csv(Path(cwd) / name / 'train.csv', target_column=target_column,
                             random_seed=random_seed, name=name)
    test = DataSet.from_csv(Path(cwd) / name / 'test.csv', target_column=target_column,
                            random_seed=random_seed, name=name)
    return train, test

def load_bike(random_seed: int = None):
    return load_dataset_csv('bike', random_seed=random_seed)

def load_boston(random_seed: int = None):
    return load_dataset_csv('boston', random_seed=random_seed)

def load_california(random_seed: int = None):
    return load_dataset_csv('california', random_seed=random_seed)

def load_titanic(random_seed: int = None):
    return load_dataset_csv('titanic', random_seed=random_seed)
