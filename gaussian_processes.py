import pandas as pd
import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt


def get_distr_params(gamma, ind, sigma=1, mean=0):
    cov = sigma ** 2 * np.exp(-(np.add.outer(ind, -ind)) ** 2 / (2 * gamma ** 2))
    means = np.ones(len(ind)) * mean
    return means, cov


def make_paths(sigma, gamma, paths_num, point1, point2, ind, paths_selected=25, mean=0):
    paths = np.ones(shape=(0, len(ind)))
    paths_chunk_size=500_000
    for i in range(0, paths_num, paths_chunk_size):
        paths_chunk = multivariate_normal(*get_distr_params(gamma, ind, sigma, mean), 
                                          min(paths_chunk_size, paths_num - i))
        paths = np.append(paths, paths_chunk, axis=0)
        paths = paths[
            np.maximum(np.abs(paths[:, 0] - point1), np.abs(paths[:, -1] - point2)).argsort()
        ][:paths_selected]
    return paths


def plot_different_GP(sigmas, gammas, point1, point2, gradularity, paths_selected=25, paths_num=500_000, **kwargs):
    ind = np.linspace(0, 1, gradularity + 1)
    fig, ax = plt.subplots(nrows=len(gammas), ncols=len(sigmas), figsize=(5 * len(sigmas), 5 * len(gammas)))

    for col, sigma in enumerate(sigmas):
        all_paths = {}
        ymax, ymin = -sigma * 5, sigma * 5
        for gamma in gammas:
            all_paths[gamma] = make_paths(sigma, gamma, paths_num, point1, point2, ind, paths_selected)
            ymax = max(ymax, all_paths[gamma].max())
            ymin = min(ymin, all_paths[gamma].min())
        for row, gamma in enumerate(gammas):
            pd.DataFrame(all_paths[gamma], columns=ind).T.plot(
                legend=False, 
                title=f'σ={sigma}, γ={gamma}', 
                ax=ax[row, col], 
                ylim=(ymin, ymax), 
                **kwargs
            )
    plt.show()
