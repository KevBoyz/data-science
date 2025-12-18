import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime


# vendas = pd.read_excel('assets/Vendas.xlsx')
# norte_shopping = vendas.loc[vendas['ID Loja'] == 'Norte Shopping', ['Data', 'Quantidade', 'Valor Final']]
# norte_shopping.dropna()
# norte_shopping = norte_shopping.groupby('Data').sum()
# norte_shopping.to_csv('assets/norte_shopping.csv', index=False)

norte_shopping = pd.read_csv('assets/norte_shopping.csv')
norte_shopping = norte_shopping.groupby('Data').sum()

#plt.style.use("seaborn-dark")

for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#212946'  # bluish dark grey

for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey

fig, ax = plt.subplots()
ax.grid(color='#3B456A')  # bluish dark grey, but slightly lighter than background
colors = [
    '#08F7FE',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41'   # matrix green
]

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

ax.xaxis.set_minor_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))

plt.ylim(0, 17500)

ax.plot(norte_shopping['Quantidade'] * 30, '--.', label='Produtos vendidos', color=colors[3])
ax.plot(norte_shopping['Valor Final'], '--.', label='Valor arrecadado', color=colors[0])
ax.plot((norte_shopping['Valor Final'] / norte_shopping['Quantidade']) * 20, linestyle='dotted', color=colors[1], label='Vendas/faturamento', marker='.')
ax.legend(loc="upper left")
plt.title('Norte Shopping - Vendas 2019')

plt.savefig("grafico_vendas_norte_shopping.png", dpi=300)

