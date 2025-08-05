"""
Módulo para engenharia de features
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import LabelEncoder

class M5FeatureEngineer:
    """Classe para criação de features para o modelo M5"""
    
    def __init__(self):
        self.label_encoders = {}
        
    def create_melted_data(self, sales_train: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
        """Converte dados de wide para long format"""
        print("Convertendo dados para formato long...")
        
        # Identifica colunas de dias
        day_cols = [col for col in sales_train.columns if col.startswith('d_')]
        
        # Melt dos dados
        sales_melted = pd.melt(
            sales_train,
            id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
            value_vars=day_cols,
            var_name='d',
            value_name='demand'
        )
        
        # Merge com calendar
        sales_melted = sales_melted.merge(calendar, on='d', how='left')
        
        print(f"Melted data shape: {sales_melted.shape}")
        return sales_melted
    
    def add_price_features(self, data: pd.DataFrame, sell_prices: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de preço"""
        print("Adicionando features de preço...")
        
        # Merge com preços
        data = data.merge(
            sell_prices,
            on=['store_id', 'item_id', 'wm_yr_wk'],
            how='left'
        )
        
        # Features de preço
        data['sell_price'] = data['sell_price'].fillna(0)
        
        # Price change features
        data = data.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)
        data['price_change'] = data.groupby(['item_id', 'store_id'])['sell_price'].diff()
        data['price_change_pct'] = data.groupby(['item_id', 'store_id'])['sell_price'].pct_change()
        
        # Price momentum (últimos 7 dias)
        data['price_momentum'] = data.groupby(['item_id', 'store_id'])['sell_price'].rolling(7).mean().reset_index(drop=True)
        
        # Relative price (price vs category average)
        data['avg_price_cat'] = data.groupby(['cat_id', 'date'])['sell_price'].transform('mean')
        data['relative_price'] = data['sell_price'] / (data['avg_price_cat'] + 1e-8)
        
        return data
    
    def add_lag_features(self, data: pd.DataFrame, lags: List[int] = [1, 7, 14, 28]) -> pd.DataFrame:
        """Adiciona features de lag"""
        print(f"Adicionando lags: {lags}")
        
        data = data.sort_values(['id', 'date']).reset_index(drop=True)
        
        for lag in lags:
            data[f'demand_lag_{lag}'] = data.groupby('id')['demand'].shift(lag)
            
        return data
    
    def add_rolling_features(self, data: pd.DataFrame, windows: List[int] = [7, 14, 28]) -> pd.DataFrame:
        """Adiciona features de médias móveis"""
        print(f"Adicionando rolling features: {windows}")
        
        data = data.sort_values(['id', 'date']).reset_index(drop=True)
        
        for window in windows:
            # Rolling mean
            data[f'rolling_mean_{window}'] = data.groupby('id')['demand'].rolling(window).mean().reset_index(drop=True)
            
            # Rolling std
            data[f'rolling_std_{window}'] = data.groupby('id')['demand'].rolling(window).std().reset_index(drop=True)
            
            # Rolling min/max
            data[f'rolling_min_{window}'] = data.groupby('id')['demand'].rolling(window).min().reset_index(drop=True)
            data[f'rolling_max_{window}'] = data.groupby('id')['demand'].rolling(window).max().reset_index(drop=True)
            
        return data
    
    def add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features estatísticas"""
        print("Adicionando features estatísticas...")
        
        # Features por item/store
        group_stats = data.groupby(['item_id', 'store_id'])['demand'].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).reset_index()
        
        group_stats.columns = ['item_id', 'store_id'] + [f'item_store_{col}' for col in group_stats.columns[2:]]
        
        data = data.merge(group_stats, on=['item_id', 'store_id'], how='left')
        
        # Features por categoria/store
        cat_stats = data.groupby(['cat_id', 'store_id'])['demand'].agg([
            'mean', 'std'
        ]).reset_index()
        
        cat_stats.columns = ['cat_id', 'store_id'] + [f'cat_store_{col}' for col in cat_stats.columns[2:]]
        
        data = data.merge(cat_stats, on=['cat_id', 'store_id'], how='left')
        
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Codifica features categóricas"""
        print("Codificando features categóricas...")
        
        categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
                           'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        
        for col in categorical_cols:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data[col].astype(str))
                else:
                    data[f'{col}_encoded'] = self.label_encoders[col].transform(data[col].astype(str))
        
        return data
    
    def create_demand_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cria features baseadas na demanda histórica"""
        print("Criando features de demanda...")
        
        # Zero demand features
        data['zero_demand'] = (data['demand'] == 0).astype(int)
        data['zero_demand_lag_1'] = data.groupby('id')['zero_demand'].shift(1)
        
        # Consecutive zero days
        data['consecutive_zeros'] = data.groupby('id')['zero_demand'].apply(
            lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        ).reset_index(drop=True)
        
        # Days since last sale
        data['days_since_last_sale'] = np.nan
        data.loc[data['demand'] > 0, 'days_since_last_sale'] = 0
        
        for id_val in data['id'].unique():
            mask = data['id'] == id_val
            subset = data.loc[mask].copy()
            subset['days_since_last_sale'] = subset['days_since_last_sale'].fillna(method='ffill')
            subset['days_since_last_sale'] = subset.groupby(subset['days_since_last_sale'].isna().cumsum())['days_since_last_sale'].apply(lambda x: x.fillna(method='ffill') + np.arange(len(x)))
            data.loc[mask, 'days_since_last_sale'] = subset['days_since_last_sale']
        
        return data
    
    def create_all_features(self, sales_train: pd.DataFrame, calendar: pd.DataFrame, 
                           sell_prices: pd.DataFrame) -> pd.DataFrame:
        """Pipeline completo de criação de features"""
        print("Iniciando pipeline de features...")
        
        # 1. Converte para formato long
        data = self.create_melted_data(sales_train, calendar)
        
        # 2. Adiciona features de preço
        data = self.add_price_features(data, sell_prices)
        
        # 3. Adiciona lags
        data = self.add_lag_features(data)
        
        # 4. Adiciona rolling features
        data = self.add_rolling_features(data)
        
        # 5. Adiciona features estatísticas
        data = self.add_statistical_features(data)
        
        # 6. Adiciona features de demanda
        data = self.create_demand_features(data)
        
        # 7. Codifica categóricas
        data = self.encode_categorical_features(data)
        
        print(f"Dataset final shape: {data.shape}")
        print("Pipeline de features concluído!")
        
        return data
    
    def get_feature_list(self, data: pd.DataFrame) -> List[str]:
        """Retorna lista de features para o modelo"""
        exclude_cols = ['id', 'date', 'd', 'demand', 'item_id', 'dept_id', 'cat_id', 
                       'store_id', 'state_id', 'event_name_1', 'event_type_1', 
                       'event_name_2', 'event_type_2']
        
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        return feature_cols
