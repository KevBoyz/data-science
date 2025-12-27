import pandas as pd

vendas = pd.read_excel('./assets/Vendas-Dez.xlsx')
vendas['Comissão'] = vendas['Valor Final'] * 0.05
vendas2 = pd.read_excel('./assets/Vendas.xlsx')
vendas2 = vendas2.dropna()

vendas = pd.concat([vendas, vendas2])

# Deleting empty values
# vendas.dropna(how='all', axis=0)  # If all v in r/c are empty, delete
# vendas = vendas.dropna()  # Delete if one v is empty

# Fill empty values
vendas['Comissão'] = vendas['Comissão'].fillna(vendas['Comissão'].mean())
# vendas = vendas.ffill()  # Values to the bottom

transacoes = vendas['ID Loja'].value_counts()

faturamento_loja = vendas[['ID Loja', 'Valor Final']].groupby('ID Loja').sum()
faturamento_loja.sort_values(by=['Valor Final'], inplace=True)  # Sorting

gerentes = pd.read_excel('assets/Gerentes.xlsx')
vendas = vendas.merge(gerentes)
gerentes_loja = vendas[['ID Loja', 'Gerente']]
gerentes_loja = gerentes_loja.drop_duplicates()

print(transacoes)

