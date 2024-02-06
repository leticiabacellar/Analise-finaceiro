# Instale as bibliotecas necessárias:
# pip install pandas matplotlib yfinance

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import cvxpy as cp
import plotly.express as px


# 1. Coleta de Dados
ticker = 'AAPL'  # Símbolo da Apple, mas pode ser substituído por outra ação
start_date = '2020-01-01'
end_date = '2021-01-01'

data = yf.download(ticker, start=start_date, end=end_date)

# 2. Exploração Inicial
print(data.head())

# 3. Visualização de Tendências
plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Preço de Fechamento')
plt.title(f'Tendência do Preço de {ticker}')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento (USD)')
plt.legend()
plt.show()

# 4. Análise de Retorno e Risco
data['Daily Return'] = data['Close'].pct_change()
data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()

plt.figure(figsize=(10, 5))
plt.plot(data['Cumulative Return'], label='Retorno Cumulativo')
plt.title(f'Retorno Cumulativo de {ticker}')
plt.xlabel('Data')
plt.ylabel('Retorno Cumulativo')
plt.legend()
plt.show()

# 5. Correlação com Índices de Mercado
index_ticker = '^GSPC'  # S&P 500, pode ser alterado para outro índice
index_data = yf.download(index_ticker, start=start_date, end=end_date)

correlation = data['Close'].corr(index_data['Close'])
print(f"Correlação entre {ticker} e {index_ticker}: {correlation}")

# 6. Modelagem de Risco e Retorno (Otimização de Portfólio) - Requer bibliotecas adicionais

# Função para obter dados de ações
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

# Lista de ações no portfólio
portfolio_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
start_date = '2020-01-01'
end_date = '2021-01-01'

# Obtendo dados de preços de fechamento
prices = pd.DataFrame({ticker: get_stock_data(ticker, start_date, end_date) for ticker in portfolio_tickers})

# Calcular os retornos diários
returns = prices.pct_change().dropna()

# Parâmetros para a otimização
n_assets = len(portfolio_tickers)
weights = cp.Variable(n_assets)
expected_returns = returns.mean()
cov_matrix = returns.cov()

# Problema de otimização
objective = cp.Maximize(expected_returns @ weights - 0.5 * cp.quad_form(weights, cov_matrix))
constraints = [cp.sum(weights) == 1, weights >= 0]
problem = cp.Problem(objective, constraints)

# Resolvendo o problema de otimização
result = problem.solve()

# Obtendo os pesos otimizados
optimal_weights = weights.value

# Imprimir os pesos otimizados
for ticker, weight in zip(portfolio_tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

# Visualizar o portfólio otimizado
portfolio_return = expected_returns @ optimal_weights
portfolio_std_dev = cp.sqrt(cp.quad_form(optimal_weights, cov_matrix)).value

print(f"Retorno do Portfólio: {portfolio_return:.4f}")
print(f"Desvio Padrão do Portfólio: {portfolio_std_dev:.4f}")

# Gráfico de Dispersão dos Portfólios simulados
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
for i in range(num_portfolios):
    weights = np.random.random(n_assets)
    weights /= np.sum(weights)
    portfolio_return = expected_returns @ weights
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    results[0, i] = portfolio_return
    results[1, i] = portfolio_std_dev
    results[2, i] = portfolio_return / portfolio_std_dev

# Plotar os resultados
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
plt.colorbar(label='Índice de Sharpe')
plt.title('Fronteira Eficiente')
plt.xlabel('Desvio Padrão (Risco)')
plt.ylabel('Retorno')
plt.show()


# 7. Visualizações Interativas - Pode ser feito com Plotly


# Obtendo dados de preços de fechamento
prices = pd.DataFrame({ticker: get_stock_data(ticker, start_date, end_date) for ticker in portfolio_tickers})

# Calcular os retornos diários
returns = prices.pct_change().dropna()

# Gráfico de Tendência Interativo
fig_trend = px.line(prices, x=prices.index, y=portfolio_tickers, labels={'value': 'Preço de Fechamento (USD)'}, title='Tendência de Preços')
fig_trend.update_xaxes(title_text='Data')
fig_trend.show()

# Gráfico de Retorno Cumulativo Interativo
cumulative_returns = (1 + returns).cumprod()
fig_cumulative = px.line(cumulative_returns, x=cumulative_returns.index, y=portfolio_tickers, labels={'value': 'Retorno Cumulativo'}, title='Retorno Cumulativo')
fig_cumulative.update_xaxes(title_text='Data')
fig_cumulative.show()

# Gráfico de Dispersão da Correlação com o Índice de Mercado
index_ticker = '^GSPC'  # S&P 500
index_data = yf.download(index_ticker, start=start_date, end=end_date)
correlation_data = pd.concat([returns, index_data['Close']], axis=1).dropna()

fig_correlation = px.scatter(correlation_data, x=index_ticker, y=portfolio_tickers, title='Correlação com o Índice de Mercado')
fig_correlation.update_xaxes(title_text=f'Índice de Mercado ({index_ticker})')
fig_correlation.update_yaxes(title_text='Retorno Diário')
fig_correlation.show()

# Obtendo dados de preços de fechamento
prices = pd.DataFrame({ticker: get_stock_data(ticker, start_date, end_date) for ticker in portfolio_tickers})

# Calcular os retornos diários
returns = prices.pct_change().dropna()

# Gráfico de Tendência Interativo
fig_trend = px.line(prices, x=prices.index, y=portfolio_tickers, labels={'value': 'Preço de Fechamento (USD)'}, title='Tendência de Preços')
fig_trend.update_xaxes(title_text='Data')
fig_trend.show()

# Gráfico de Retorno Cumulativo Interativo
cumulative_returns = (1 + returns).cumprod()
fig_cumulative = px.line(cumulative_returns, x=cumulative_returns.index, y=portfolio_tickers, labels={'value': 'Retorno Cumulativo'}, title='Retorno Cumulativo')
fig_cumulative.update_xaxes(title_text='Data')
fig_cumulative.show()

# Gráfico de Dispersão da Correlação com o Índice de Mercado
index_ticker = '^GSPC'  # S&P 500
index_data = yf.download(index_ticker, start=start_date, end=end_date)
correlation_data = pd.concat([returns, index_data['Close']], axis=1).dropna()

fig_correlation = px.scatter(correlation_data, x=index_ticker, y=portfolio_tickers, title='Correlação com o Índice de Mercado')
fig_correlation.update_xaxes(title_text=f'Índice de Mercado ({index_ticker})')
fig_correlation.update_yaxes(title_text='Retorno Diário')
fig_correlation.show()


# 8. Relatório de Insights - Documentar suas conclusões

# Jupyter Notebook para Documentação
from IPython.display import display, Markdown

# Insights e Conclusões
insights = """
## Análise Financeira - Relatório de Insights

### 1. Tendência de Preços
- Observamos uma tendência de alta nos preços das ações ao longo do período analisado, especialmente para a Apple (AAPL) e Amazon (AMZN).

### 2. Retorno Cumulativo
- O retorno cumulativo mostra desempenho positivo ao longo do tempo, indicando resultados favoráveis para o portfólio.

### 3. Correlação com o Índice de Mercado
- A correlação entre as ações do portfólio e o índice de mercado (S&P 500) revela forte dependência, sugerindo que o desempenho do portfólio está alinhado com o mercado.

### 4. Otimização de Portfólio
- A otimização de portfólio indicou uma alocação ideal de ativos para maximizar o retorno ajustado ao risco.
- Os pesos otimizados sugerem uma distribuição equilibrada entre as ações selecionadas.

### 5. Fronteira Eficiente
- A fronteira eficiente destaca as combinações de portfólio que oferecem o melhor equilíbrio entre retorno e risco.

### 6. Conclusões Gerais
- O portfólio analisado demonstra bom desempenho, mas é importante monitorar as condições de mercado e reavaliar a estratégia regularmente.

---

Este relatório é uma análise inicial e deve ser complementado com informações adicionais e contextuais.
"""

# Exibir o Relatório
display(Markdown(insights))