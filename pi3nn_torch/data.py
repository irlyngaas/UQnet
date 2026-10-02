"""
Minimal data loader for the boston-housing UCI benchmark, reading the data
file already shipped in this repo (datasets/UCI_datasets/boston-housing/)
rather than duplicating it -- unlike the copy of this module kept in
hydrogen-emulator-configurable, which copies the data file locally since
that's a separate repo and shouldn't depend on this one's location.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_boston():
    path = os.path.join(HERE, '..', 'datasets', 'UCI_datasets', 'boston-housing', 'boston_housing.txt')
    raw = np.loadtxt(path)
    X = raw[:, :-1]
    Y = raw[:, -1]
    return X, Y
