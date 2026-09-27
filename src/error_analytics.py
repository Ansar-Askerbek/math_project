import numpy as np

def calculate_mse(y_approx, y_true):
    """
    Вычисляет среднеквадратичную ошибку (Mean Squared Error - MSE).
    MSE = (1/N) * sum((y_approx - y_true)^2)
    """
    return np.mean((y_approx - y_true) ** 2)

def calculate_mae(y_approx, y_true):
    """
    Вычисляет среднюю абсолютную ошибку (Mean Absolute Error - MAE).
    MAE = (1/N) * sum(|y_approx - y_true|)
    """
    return np.mean(np.abs(y_approx - y_true))

def calculate_relative_error(y_approx, y_true):
    """
    Вычисляет среднюю относительную погрешность в процентах (%).
    """
    # Добавляем 1e-10, чтобы избежать деления на ноль, если y_true == 0
    return np.mean(np.abs((y_approx - y_true) / (y_true + 1e-10))) * 100