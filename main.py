import numpy as np
import matplotlib.pyplot as plt

# Импортируем модули проекта
from src.solvers import solve_euler, solve_rk4
from src.error_analytics import calculate_mse, calculate_relative_error

#quarantine, seasonal, constant

# 1. Параметры модели SIR
N = 1000            # Общая популяция (Y-intercept max = Population size)
I0, R0 = 1, 0       # 1 инфицированный, 0 выздоровевших
S0 = N - I0         # Восприимчивые
t_max = 100         # Время моделирования в днях (X-intercept max = Days)

# 2. Расчет True Value (RK4 с очень маленьким шагом dt = 0.0001)
print("1. Вычисление True Value (RK4, dt = 0.0001)...")
t_true, S_true, I_true, R_true = solve_rk4(
    S0, I0, R0, N, t_max, dt=0.0001, beta_type='constant'
)

# 3. Расчет методов с тестовым крупным шагом dt = 1.0 дня
dt_test = 1.0
print(f"2. Расчет методов с шагом dt = {dt_test} дня...")
t_eul, S_eul, I_eul, R_eul = solve_euler(
    S0, I0, R0, N, t_max, dt=dt_test, beta_type='constant'
)
t_rk4, S_rk4, I_rk4, R_rk4 = solve_rk4(
    S0, I0, R0, N, t_max, dt=dt_test, beta_type='constant'
)

# 4. Вычисление погрешностей по инфицированным (I)
I_true_interp = np.interp(t_rk4, t_true, I_true)
mse_euler = calculate_mse(I_eul, I_true_interp)
mse_rk4 = calculate_mse(I_rk4, I_true_interp)
rel_err_euler = calculate_relative_error(I_eul, I_true_interp)
rel_err_rk4 = calculate_relative_error(I_rk4, I_true_interp)

print("\n--- РЕЗУЛЬТАТЫ АНАЛИЗА ПОГРЕШНОСТИ (по кривой I) ---")
print(f"Метод Эйлера (dt={dt_test}): MSE = {mse_euler:.6f}, Относительная ошибка = {rel_err_euler:.2f}%")
print(f"Метод RK4    (dt={dt_test}): MSE = {mse_rk4:.6f}, Относительная ошибка = {rel_err_rk4:.4f}%")

# ==============================================================================
# 5. ПОЛНЫЙ ОБЪЕДИНЕННЫЙ ГРАФИК ПОПУЛЯЦИИ
# ==============================================================================
plt.figure(figsize=(12, 7))

# --- ЭТАЛОННЫЕ КРИВЫЕ (True Value: S, I, R) ---
plt.plot(t_true, S_true, color='gray', linestyle='--', linewidth=1.5, label='S True (Восприимчивые)')
plt.plot(t_true, I_true, color='black', linestyle='--', linewidth=2.0, label='I True (Эталон Инфицированные)')
plt.plot(t_true, R_true, color='lightgreen', linestyle='--', linewidth=1.5, label='R True (Выздоровевшие)')

# --- МЕТОД ЭЙЛEРА (dt = 1.0) ---
plt.plot(t_eul, S_eul, color='cyan', linestyle='-.', marker='v', markevery=5, alpha=0.7, label=f'S Эйлер (dt={dt_test})')
plt.plot(t_eul, I_eul, color='red', linestyle='-', marker='o', markevery=4, linewidth=2, label=f'I Эйлер (dt={dt_test})')
plt.plot(t_eul, R_eul, color='orange', linestyle='-.', marker='^', markevery=5, alpha=0.7, label=f'R Эйлер (dt={dt_test})')

# --- МЕТОД РУНГЕ-КУТТЫ 4-ГО ПОРЯДКА (dt = 1.0) ---
plt.plot(t_rk4, S_rk4, color='blue', linestyle='-.', alpha=0.6, label=f'S RK4 (dt={dt_test})')
plt.plot(t_rk4, I_rk4, color='magenta', linestyle='-', marker='s', markevery=4, linewidth=2, label=f'I RK4 (dt={dt_test})')
plt.plot(t_rk4, R_rk4, color='green', linestyle='-.', alpha=0.6, label=f'R RK4 (dt={dt_test})')

# --- ЛИНИЯ ПОЛНОЙ ПОПУЛЯЦИИ (N) ---
plt.axhline(y=N, color='purple', linestyle='-', linewidth=1, label=f'Max Population (N = {N})')

# Оформление осей и заголовка
plt.title('Полная динамика SIR-модели: Сравнение методов Эйлера и RK4 с True Value', fontsize=13)
plt.xlabel('Days / Время (Дни)', fontsize=11)
plt.ylabel('Population Size / Размер популяции (Человек)', fontsize=11)

# Ограничения осей
plt.xlim(0, t_max)
plt.ylim(0, N + 50)

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), fontsize=9) # Легенда вынесена сбоку
plt.tight_layout()

plt.show()