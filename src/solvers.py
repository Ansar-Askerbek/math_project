import numpy as np
import src.sir_model as sir_derivatives

def solve_euler(S0, I0, R0, N, t_max, dt, beta_type='constant'):
    """
    Решает систему уравнений SIR методом Эйлера.
    
    Параметры:
    - S0: Начальное количество восприимчивых
    - I0: Начальное количество инфицированных
    - R0: Начальное количество выздоровевших
    - N: Общая численность населения
    - t_max: Максимальное время (дни)
    - dt: Шаг по времени (дни)
    - beta_type: Тип динамики коэффициента передачи инфекции ('constant', 'quarantine', 'seasonal')
    
    Возвращает:
    - t: Массив времени
    - S: Массив количества восприимчивых
    - I: Массив количества инфицированных
    - R: Массив количества выздоровевших
    """
    
    steps = int(t_max / dt)
    
    S = np.zeros(steps)
    I = np.zeros(steps)
    R = np.zeros(steps)
    t = np.linspace(0, t_max, steps)

    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(steps - 1):
        dS_dt, dI_dt, dR_dt = sir_derivatives.sir_derivatives(t[i], S[i], I[i], R[i], N, beta_type=beta_type)

        S[i + 1] = S[i] + dS_dt * dt
        I[i + 1] = I[i] + dI_dt * dt
        R[i + 1] = R[i] + dR_dt * dt

    return t, S, I, R

def solve_rk4(S0, I0, R0, N, t_max, dt, beta_type='constant'):
    """
    Решает систему уравнений SIR методом Рунге-Кутты 4-го порядка.
    
    Параметры:
    - S0: Начальное количество восприимчивых
    - I0: Начальное количество инфицированных
    - R0: Начальное количество выздоровевших
    - N: Общая численность населения
    - t_max: Максимальное время (дни)
    - dt: Шаг по времени (дни)
    - beta_type: Тип динамики коэффициента передачи инфекции ('constant', 'quarantine', 'seasonal')
    
    Возвращает:
    - t: Массив времени
    - S: Массив количества восприимчивых
    - I: Массив количества инфицированных
    - R: Массив количества выздоровевших
    """
    
    steps = int(t_max / dt)
    t = np.linspace(0, t_max, steps)
    
    S = np.zeros(steps)
    I = np.zeros(steps)
    R = np.zeros(steps)
    
    S[0], I[0], R[0] = S0, I0, R0
    
    for i in range(0, steps - 1):
        ti = t[i]
        Si, Ii, Ri = S[i], I[i], R[i]
        
        # --- Шаг K1 ---
        k1_S, k1_I, k1_R = sir_derivatives.sir_derivatives(ti, Si, Ii, Ri, N, beta_type)
        
        # --- Шаг K2 ---
        k2_S, k2_I, k2_R = sir_derivatives.sir_derivatives(ti + 0.5*dt, 
                                           Si + 0.5*dt*k1_S, 
                                           Ii + 0.5*dt*k1_I, 
                                           Ri + 0.5*dt*k1_R, N, beta_type)
        
        # --- Шаг K3 ---
        k3_S, k3_I, k3_R = sir_derivatives.sir_derivatives(ti + 0.5*dt, 
                                           Si + 0.5*dt*k2_S, 
                                           Ii + 0.5*dt*k2_I, 
                                           Ri + 0.5*dt*k2_R, N, beta_type)
        
        # --- Шаг K4 ---
        k4_S, k4_I, k4_R = sir_derivatives.sir_derivatives(ti + dt, 
                                           Si + dt*k3_S, 
                                           Ii + dt*k3_I, 
                                           Ri + dt*k3_R, N, beta_type)
        
        # Пересчет на следующий шаг
        S[i+1] = Si + (dt / 6.0) * (k1_S + 2*k2_S + 2*k3_S + k4_S)
        I[i+1] = Ii + (dt / 6.0) * (k1_I + 2*k2_I + 2*k3_I + k4_I)
        R[i+1] = Ri + (dt / 6.0) * (k1_R + 2*k2_R + 2*k3_R + k4_R)
        
    return t, S, I, R