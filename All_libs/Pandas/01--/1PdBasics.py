import pandas as pd
from math import trunc

vendas_test = {
    'data': ['22/2', '23/3'],  # DataFrame by Python dict
    'produto': ['Canets', 'Canets'],
    'valor': [30, 40],
    'qtde': [3, 4]
}
vendas_df = pd.DataFrame(vendas_test)
vendas_df = pd.concat([vendas_df, pd.DataFrame(vendas_test)])
print(vendas_df)

# vendas_df = pd.read_excel('assets/Vendas.xlsx')  # pip install openpyxl

# print(vendas_df.head(5), vendas_df.shape)  # Header of df and shape
# print(vendas_df.tail(1))  # Last element
# print(vendas_df.describe())  # Statistics of df: min, max, mean,

# Getting columns
# print(vendas_df['Produto'])  # Pandas series
# print(vendas_df[['ID Loja', 'Produto']])  # DataFrame

# Getting Items
# print(vendas_df.loc[1:5])
# print(vendas_df.loc[1, 'ID Loja'])  # Specific item            # Rows, Columns
# norte_Shopping_df = vendas_df.loc[vendas_df['ID Loja'] == 'Norte Shopping', ['Produto', 'Quantidade', 'Valor Final']]
# print(norte_Shopping_df.describe())
# print('Valor total arrecadado:', trunc(norte_Shopping_df['Valor Final'].sum()))  # Sum of values

# Editing DataFrames
"""norte_Shopping_df['Comissão'] = norte_Shopping_df['Valor Final'] * 0.05  # Column by existing column
# norte_Shopping_df['Imposto'] = 0  # New column bad way (slow)
norte_Shopping_df.loc[:, 'Imposto'] = 0  # New column Good way -> : = all

vendas_dez = pd.read_excel('assets/Vendas-Dez.xlsx')
vendas_df = pd.concat([vendas_df, vendas_dez])  # Adding new rows (merging dfs)
vendas_df = pd.concat([vendas_df, vendas_df.tail(1)])

vendas_df.drop('Imposto', axis=1)  # Remove column"""
