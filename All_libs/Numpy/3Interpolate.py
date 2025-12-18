from scipy import interpolate
import numpy as np
from matplotlib import pyplot as plt

prices = np.full(100, fill_value=np.nan)
prices[[0, 25, 60, -1]] = [80., 30., 75., 50.]
days = np.arange(len(prices))
is_valid = ~np.isnan(prices)
prices = np.interp(x=days, xp=days[is_valid], fp=prices[is_valid])  # val, val i
prices += np.random.randn(len(prices)) * 2

fig, ax = plt.subplots()
ax.plot(prices)
ax.plot(prices, 'b')
# plt.show()


y = np.random.random(20)
x = np.arange(0, y.size)

new_length = 100
new_x = np.linspace(x.min(), x.max(), new_length)
new_y = interpolate.interp1d(x, y, kind='linear')(new_x)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, y, 'ro-')
plt.subplot(2, 1, 2)
plt.plot(new_x, new_y, 'bo-')
plt.show()
