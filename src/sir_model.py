import numpy as np

def beta_function(t, beta = 0.3, type_dynamics = 'constant'):
    """
    Функция для вычисления коэффициента передачи инфекции (beta) в зависимости от времени.
    
    Параметры:
    - t: Время (дни)
    - beta: Базовое значение коэффициента передачи инфекции
    - type_dynamics: Тип динамики ('constant', 'linear', 'exponential')
    
    Возвращает:
    - Значение beta в момент времени t
    """
    if type_dynamics == 'constant':
        return beta
    elif type_dynamics == 'quarantine':
        return beta * np.exp(-0.05 * t)  # Линейное уменьшение на 1% в день
    elif type_dynamics == 'seasonal':
        return beta * (1 + 0.2 * np.cos(2 * np.pi * t / 365))  # Сезонное изменение
    return beta  # По умолчанию возвращаем базовое значение

def gamma_fucnction(t, gamma0 = 0.1):
    return gamma0  # В данной модели предполагаем, что коэффициент выздоровления постоянен

def sir_derivatives(t, S, I, R, N, beta_type='constant'):
    """
    Вычисляет производные для модели SIR.
    
    Параметры:
    - t: Время (дни)
    - S: Количество восприимчивых
    - I: Количество инфицированных
    - R: Количество выздоровевших
    - N: Общая численность населения
    - beta: Коэффициент передачи инфекции
    - gamma: Коэффициент выздоровления
    
    Возвращает:
    - dS_dt: Производная по времени для S
    - dI_dt: Производная по времени для I
    - dR_dt: Производная по времени для R
    """

    beta = beta_function(t, type_dynamics=beta_type)
    gamma = gamma_fucnction(t)

    dS_dt = -beta * S * I / N
    dI_dt = beta * S * I / N - gamma * I
    dR_dt = gamma * I
    return dS_dt, dI_dt, dR_dt