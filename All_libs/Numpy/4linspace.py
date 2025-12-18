from random import randint
import numpy as np
from matplotlib import pyplot as plt

matrix = np.array([np.linspace(0, 10, 50), np.linspace(0, 10, 50),
                   np.linspace(0, 10, 50), np.linspace(0, 10, 50)])
matrix[0] = matrix[0] ** 2 * randint(-5, 5)
matrix[1] = matrix[1] ** 2.5 * randint(-5, 5)
matrix[2] = matrix[2] ** 3 * randint(-5, 5)
matrix[3] = matrix[3] ** 3.5 * randint(-5, 5)

plt.style.use('seaborn-dark')

for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#212946'  # bluish dark grey

for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey


plt.plot(matrix[0], color='blue')
plt.plot(matrix[0], 'r--.', color='blue')
plt.plot(matrix[1], color='green')
plt.plot(matrix[1], 'r--.', color='green')
plt.plot(matrix[2], color='red')
plt.plot(matrix[2], 'r--.', color='red')
plt.plot(matrix[2] - matrix[1], color='orange')
plt.plot(matrix[2] - matrix[1], 'r--,', color='orange')
plt.plot(matrix[3], color='purple')
plt.plot(matrix[3], 'r--.', color='purple')
plt.plot(matrix[3] - matrix[2], color='pink')
plt.plot(matrix[3] - matrix[2], 'r--.', color='pink')
plt.show()
