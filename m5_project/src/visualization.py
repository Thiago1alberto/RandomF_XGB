"""
Módulo para visualizações e análises
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class M5Visualizer:
    """Classe para visualizações do projeto M5"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_sales_overview(self, data: pd.DataFrame) -> None:
        """Visualização geral das vendas"""
        # Agrupa por data
        daily_sales = data.groupby('date')['demand'].sum().reset_index()
        daily_sales['date'] = pd.to_datetime(daily_sales['date'])
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=['Vendas Diárias Totais', 'Vendas por Estado',
                                         'Vendas por Categoria', 'Vendas por Store'])
        
        # Vendas diárias totais
        fig.add_trace(
            go.Scatter(x=daily_sales['date'], y=daily_sales['demand'],
                      mode='lines', name='Vendas Totais'),
            row=1, col=1
        )
        
        # Vendas por estado
        state_sales = data.groupby(['date', 'state_id'])['demand'].sum().reset_index()
        for state in state_sales['state_id'].unique():
            state_data = state_sales[state_sales['state_id'] == state]
            fig.add_trace(
                go.Scatter(x=pd.to_datetime(state_data['date']), 
                          y=state_data['demand'],
                          mode='lines', name=f'State {state}'),
                row=1, col=2
            )
        
        # Vendas por categoria
        cat_sales = data.groupby(['date', 'cat_id'])['demand'].sum().reset_index()
        for cat in cat_sales['cat_id'].unique():
            cat_data = cat_sales[cat_sales['cat_id'] == cat]
            fig.add_trace(
                go.Scatter(x=pd.to_datetime(cat_data['date']), 
                          y=cat_data['demand'],
                          mode='lines', name=f'Cat {cat}'),
                row=2, col=1
            )
        
        # Vendas por store (top 5)
        store_total = data.groupby('store_id')['demand'].sum().sort_values(ascending=False).head(5)
        store_sales = data[data['store_id'].isin(store_total.index)].groupby(['date', 'store_id'])['demand'].sum().reset_index()
        
        for store in store_total.index:
            store_data = store_sales[store_sales['store_id'] == store]
            fig.add_trace(
                go.Scatter(x=pd.to_datetime(store_data['date']), 
                          y=store_data['demand'],
                          mode='lines', name=f'Store {store}'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Overview das Vendas M5")
        fig.show()
    
    def plot_seasonal_patterns(self, data: pd.DataFrame) -> None:
        """Análise de sazonalidade"""
        data_copy = data.copy()
        data_copy['date'] = pd.to_datetime(data_copy['date'])
        data_copy['dayofweek'] = data_copy['date'].dt.dayofweek
        data_copy['month'] = data_copy['date'].dt.month
        data_copy['quarter'] = data_copy['date'].dt.quarter
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=['Vendas por Dia da Semana', 'Vendas por Mês',
                                         'Vendas por Trimestre', 'Vendas com SNAP'])
        
        # Por dia da semana
        dow_sales = data_copy.groupby('dayofweek')['demand'].mean()
        fig.add_trace(
            go.Bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 
                  y=dow_sales.values, name='Dia da Semana'),
            row=1, col=1
        )
        
        # Por mês
        month_sales = data_copy.groupby('month')['demand'].mean()
        fig.add_trace(
            go.Bar(x=month_sales.index, y=month_sales.values, name='Mês'),
            row=1, col=2
        )
        
        # Por trimestre
        quarter_sales = data_copy.groupby('quarter')['demand'].mean()
        fig.add_trace(
            go.Bar(x=quarter_sales.index, y=quarter_sales.values, name='Trimestre'),
            row=2, col=1
        )
        
        # SNAP effect
        snap_cols = ['snap_CA', 'snap_TX', 'snap_WI']
        snap_data = []
        for col in snap_cols:
            if col in data_copy.columns:
                snap_effect = data_copy.groupby(col)['demand'].mean()
                snap_data.append(go.Bar(x=[f'{col}_0', f'{col}_1'], 
                                       y=snap_effect.values, 
                                       name=col))
        
        for trace in snap_data:
            fig.add_trace(trace, row=2, col=2)
        
        fig.update_layout(height=800, title_text="Padrões Sazonais")
        fig.show()
    
    def plot_item_analysis(self, data: pd.DataFrame, top_n: int = 10) -> None:
        """Análise de itens específicos"""
        # Top itens por volume
        top_items = data.groupby('item_id')['demand'].sum().sort_values(ascending=False).head(top_n)
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=['Top Itens por Volume', 'Distribuição de Demanda',
                                         'Itens com Zero Vendas', 'Séries Temporais Top 5'])
        
        # Top itens
        fig.add_trace(
            go.Bar(x=top_items.index, y=top_items.values, name='Volume Total'),
            row=1, col=1
        )
        
        # Distribuição de demanda
        fig.add_trace(
            go.Histogram(x=data['demand'], nbinsx=50, name='Distribuição'),
            row=1, col=2
        )
        
        # Zero vendas por item
        zero_sales = data.groupby('item_id')['demand'].apply(lambda x: (x == 0).sum()).sort_values(ascending=False).head(top_n)
        fig.add_trace(
            go.Bar(x=zero_sales.index, y=zero_sales.values, name='Dias Zero'),
            row=2, col=1
        )
        
        # Séries temporais top 5
        top_5_items = top_items.head(5).index
        for item in top_5_items:
            item_data = data[data['item_id'] == item].groupby('date')['demand'].sum().reset_index()
            fig.add_trace(
                go.Scatter(x=pd.to_datetime(item_data['date']), 
                          y=item_data['demand'],
                          mode='lines', name=f'Item {item}'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Análise de Itens")
        fig.show()
    
    def plot_price_analysis(self, data: pd.DataFrame) -> None:
        """Análise de preços"""
        if 'sell_price' not in data.columns:
            print("Dados de preço não disponíveis")
            return
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=['Distribuição de Preços', 'Preço vs Demanda',
                                         'Evolução de Preços', 'Correlação Preço-Demanda'])
        
        # Distribuição de preços
        fig.add_trace(
            go.Histogram(x=data['sell_price'], nbinsx=50, name='Preços'),
            row=1, col=1
        )
        
        # Preço vs Demanda (scatter)
        sample_data = data.sample(n=min(10000, len(data)))  # Sample para performance
        fig.add_trace(
            go.Scatter(x=sample_data['sell_price'], y=sample_data['demand'],
                      mode='markers', name='Preço vs Demanda', opacity=0.5),
            row=1, col=2
        )
        
        # Evolução de preços (média por categoria)
        if 'cat_id' in data.columns:
            price_evolution = data.groupby(['date', 'cat_id'])['sell_price'].mean().reset_index()
            for cat in price_evolution['cat_id'].unique():
                cat_data = price_evolution[price_evolution['cat_id'] == cat]
                fig.add_trace(
                    go.Scatter(x=pd.to_datetime(cat_data['date']), 
                              y=cat_data['sell_price'],
                              mode='lines', name=f'Cat {cat}'),
                    row=2, col=1
                )
        
        # Correlação por categoria
        if 'cat_id' in data.columns:
            corr_data = []
            for cat in data['cat_id'].unique():
                cat_subset = data[data['cat_id'] == cat]
                if len(cat_subset) > 1:
                    corr = cat_subset['sell_price'].corr(cat_subset['demand'])
                    corr_data.append({'category': cat, 'correlation': corr})
            
            if corr_data:
                corr_df = pd.DataFrame(corr_data)
                fig.add_trace(
                    go.Bar(x=corr_df['category'], y=corr_df['correlation'], 
                          name='Correlação'),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, title_text="Análise de Preços")
        fig.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20) -> None:
        """Visualiza importância das features"""
        top_features = importance_df.head(top_n)
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=top_features['importance'], 
                  y=top_features['feature'],
                  orientation='h',
                  name='Feature Importance')
        )
        
        fig.update_layout(
            title=f'Top {top_n} Features Mais Importantes',
            xaxis_title='Importância',
            yaxis_title='Features',
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        fig.show()
    
    def plot_model_performance(self, cv_results: List[Dict]) -> None:
        """Visualiza performance do modelo"""
        cv_df = pd.DataFrame(cv_results)
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=['RMSE por Fold', 'MAE por Fold'])
        
        # RMSE
        fig.add_trace(
            go.Bar(x=cv_df['fold'], y=cv_df['train_rmse'], name='Train RMSE'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=cv_df['fold'], y=cv_df['val_rmse'], name='Val RMSE'),
            row=1, col=1
        )
        
        # MAE
        fig.add_trace(
            go.Bar(x=cv_df['fold'], y=cv_df['train_mae'], name='Train MAE'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=cv_df['fold'], y=cv_df['val_mae'], name='Val MAE'),
            row=1, col=2
        )
        
        fig.update_layout(height=500, title_text="Performance do Modelo por Fold")
        fig.show()
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  sample_size: int = 10000) -> None:
        """Compara predições vs valores reais"""
        # Sample para performance
        if len(y_true) > sample_size:
            indices = np.random.choice(len(y_true), sample_size, replace=False)
            y_true_sample = y_true[indices]
            y_pred_sample = y_pred[indices]
        else:
            y_true_sample = y_true
            y_pred_sample = y_pred
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=['Predições vs Real', 'Distribuição dos Erros'])
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(x=y_true_sample, y=y_pred_sample,
                      mode='markers', name='Predições', opacity=0.6),
            row=1, col=1
        )
        
        # Linha perfeita
        min_val = min(y_true_sample.min(), y_pred_sample.min())
        max_val = max(y_true_sample.max(), y_pred_sample.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Predição Perfeita', 
                      line=dict(dash='dash')),
            row=1, col=1
        )
        
        # Distribuição dos erros
        errors = y_pred_sample - y_true_sample
        fig.add_trace(
            go.Histogram(x=errors, nbinsx=50, name='Erros'),
            row=1, col=2
        )
        
        fig.update_layout(height=500, title_text="Análise de Predições")
        fig.show()
    
    def create_dashboard_summary(self, data: pd.DataFrame, model_results: Dict = None) -> None:
        """Cria um dashboard resumo"""
        print("=== DASHBOARD M5 FORECASTING ===\n")
        
        # Estatísticas básicas
        print("📊 ESTATÍSTICAS GERAIS:")
        print(f"• Total de registros: {len(data):,}")
        print(f"• Período: {data['date'].min()} a {data['date'].max()}")
        print(f"• Número de itens únicos: {data['item_id'].nunique():,}")
        print(f"• Número de lojas: {data['store_id'].nunique()}")
        print(f"• Demanda total: {data['demand'].sum():,.0f}")
        print(f"• Demanda média diária: {data['demand'].mean():.2f}")
        print(f"• Dias com zero vendas: {(data['demand'] == 0).sum():,} ({(data['demand'] == 0).mean()*100:.1f}%)")
        
        # Top performers
        print(f"\n🏆 TOP PERFORMERS:")
        top_items = data.groupby('item_id')['demand'].sum().sort_values(ascending=False).head(5)
        print("• Top 5 itens por volume:")
        for item, volume in top_items.items():
            print(f"  - {item}: {volume:,.0f}")
        
        top_stores = data.groupby('store_id')['demand'].sum().sort_values(ascending=False).head(5)
        print("• Top 5 lojas por volume:")
        for store, volume in top_stores.items():
            print(f"  - {store}: {volume:,.0f}")
        
        # Informações do modelo
        if model_results:
            print(f"\n🤖 PERFORMANCE DO MODELO:")
            print(f"• RMSE médio: {np.mean([r['val_rmse'] for r in model_results]):.4f}")
            print(f"• MAE médio: {np.mean([r['val_mae'] for r in model_results]):.4f}")
            print(f"• Melhor iteração: {np.mean([r['best_iteration'] for r in model_results]):.0f}")
        
        print("\n" + "="*50)
