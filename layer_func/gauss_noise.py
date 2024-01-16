import numpy as np


def gauss_noise(X, y, mean=0, sigma=0.5):
    X_noise = X + np.random.normal(mean, sigma, X.shape)
    y_noise = y.copy()
    return X_noise, y_noise
