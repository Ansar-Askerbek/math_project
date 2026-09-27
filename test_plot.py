import numpy as np
import matplotlib.pyplot as plt

gamma = 0.1
beta = 0.3

N = 1000
I0 = 1
s0 = N - I0
R0 = 0

dt = 0.1
t_max = 100
steps = int(t_max / dt)

S = np.zeros(steps)
I = np.zeros(steps)
R = np.zeros(steps)
t = np.linspace(0, t_max, steps)

S[0] = s0
I[0] = I0
R[0] = R0

for i in range(0, steps - 1):
    dS = -beta * S[i] * I[i] / N * dt
    dI = (beta * S[i] * I[i] / N - gamma * I[i]) * dt
    dR = gamma * I[i] * dt

    S[i + 1] = S[i] + dS
    I[i + 1] = I[i] + dI
    R[i + 1] = R[i] + dR

plt.figure(figsize=(10, 6))
plt.plot(t, S, label='S (Восприимчивые)', color='blue')
plt.plot(t, I, label='I (Инфицированные)', color='red')
plt.plot(t, R, label='R (Выздоровевшие)', color='green')

plt.title('Простая модель SIR (Базовый тест)')
plt.xlabel('Время (Дни)')
plt.ylabel('Количество людей')
plt.grid(True)
plt.legend()
plt.show()