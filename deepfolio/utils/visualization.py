import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

class AdvancedVisualizer:

    @staticmethod
    def plot_returns_heatmap(returns):
        plt.figure(figsize=(12, 10))
        sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Asset Returns Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_efficient_frontier(returns, covariance, n_portfolios=10000):
        def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
            returns = np.sum(mean_returns * weights) * 252
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return std, returns

        results = np.zeros((3, n_portfolios))
        weights_record = []
        for i in range(n_portfolios):
            weights = np.random.random(returns.shape[1])
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_std_dev, portfolio_return = portfolio_annualized_performance(weights, returns.mean(), covariance)
            results[0,i] = portfolio_std_dev
            results[1,i] = portfolio_return
            results[2,i] = (results[1,i] - 0.01) / results[0,i]  # Sharpe ratio
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results[0,:], y=results[1,:], mode='markers',
                                 marker=dict(size=5, color=results[2,:], colorscale='Viridis', showscale=True),
                                 text=[f'Sharpe: {s:.2f}' for s in results[2,:]], hoverinfo='text+x+y'))
        fig.update_layout(title='Efficient Frontier',
                          xaxis_title='Annualized Risk',
                          yaxis_title='Annualized Return',
                          showlegend=False)
        fig.show()

    @staticmethod
    def plot_drawdown(returns):
        wealth_index = (1 + returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=returns.index, y=drawdowns, fill='tozeroy', name='Drawdown'))
        fig.update_layout(title='Portfolio Drawdown',
                          xaxis_title='Date',
                          yaxis_title='Drawdown',
                          yaxis_tickformat='%')
        fig.show()

    @staticmethod
    def plot_rolling_beta(returns, benchmark_returns, window=252):
        excess_returns = returns - 0.03/252  # Assuming risk-free rate of 3%
        excess_market_returns = benchmark_returns - 0.03/252
        rolling_cov = excess_returns.rolling(window=window).cov(excess_market_returns)
        rolling_market_var = excess_market_returns.rolling(window=window).var()
        rolling_beta = rolling_cov / rolling_market_var

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=returns.index, y=rolling_beta, mode='lines', name='Rolling Beta'))
        fig.update_layout(title=f'{window}-day Rolling Beta',
                          xaxis_title='Date',
                          yaxis_title='Beta')
        fig.show()

    @staticmethod
    def plot_asset_allocation(weights, title='Current Asset Allocation'):
        fig = px.pie(values=weights.values, names=weights.index, title=title)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.show()

    @staticmethod
    def plot_return_distribution(returns):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            subplot_titles=('Return Distribution', 'QQ Plot'))

        fig.add_trace(go.Histogram(x=returns, name='Returns'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sorted(returns), y=np.random.normal(size=len(returns)), 
                                 mode='markers', name='QQ Plot'), row=2, col=1)

        fig.update_layout(title='Return Distribution Analysis',
                          showlegend=False)
        fig.show()

    @staticmethod
    def plot_network_graph(G, title='Network Graph'):
        pos = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                                marker=dict(showscale=True, colorscale='YlGnBu', size=10, colorbar=dict(thickness=15, title='Node Connections')))

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(f'# of connections: {len(adjacencies[1])}')

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title=title, showlegend=False, hovermode='closest',
                                         margin=dict(b=20,l=5,r=5,t=40),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        fig.show()
