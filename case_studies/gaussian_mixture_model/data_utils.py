import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
from sklearn.datasets import make_blobs
import time
from enum import Enum

class DatasetSize(Enum):
    LARGE = 1
    MEDIUM = 2
    SMALL = 3

def make_ds(centers, dataset_size: DatasetSize):
    if dataset_size == DatasetSize.LARGE:
        points_in_classes = [50_000, 20_000, 15_000, 5_500]
    elif dataset_size == DatasetSize.MEDIUM:
        points_in_classes = [15_000, 8_000, 6_000, 1_500]
    else:
        points_in_classes = [5_000, 3_500, 2_000, 500]
    cluster_stds = [1.3, 1.4, 1.5, 0.8]
    ################################################
    # Initial Guesses
    ################################################
    # Randomly increase/ decrease by 25% each way
    scale = (np.random.randint(low=-80, high=80, size=centers.shape)) / 100

    initial_mu_guesses = centers + (centers * scale)
    return make_blobs(
        points_in_classes, centers=centers, cluster_std=cluster_stds), initial_mu_guesses
