"""
Módulo para carregamento e limpeza inicial dos dados M5
"""
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict

class M5DataLoader:
    """Classe para carregar e processar os dados básicos do M5"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.sales_train = None
        self.calendar = None
        self.sell_prices = None
        self.sample_submission = None
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Carrega todos os arquivos de dados"""
        print("Carregando dados...")
        
        # Carrega sales_train_evaluation (mais completo que validation)
        self.sales_train = pd.read_csv(os.path.join(self.data_path, 'sales_train_evaluation.csv'))
        print(f"Sales train shape: {self.sales_train.shape}")
        
        # Carrega calendar
        self.calendar = pd.read_csv(os.path.join(self.data_path, 'calendar.csv'))
        print(f"Calendar shape: {self.calendar.shape}")
        
        # Carrega sell_prices
        self.sell_prices = pd.read_csv(os.path.join(self.data_path, 'sell_prices.csv'))
        print(f"Sell prices shape: {self.sell_prices.shape}")
        
        # Carrega sample_submission
        self.sample_submission = pd.read_csv(os.path.join(self.data_path, 'sample_submission.csv'))
        print(f"Sample submission shape: {self.sample_submission.shape}")
        
        return {
            'sales_train': self.sales_train,
            'calendar': self.calendar,
            'sell_prices': self.sell_prices,
            'sample_submission': self.sample_submission
        }
    
    def get_basic_info(self) -> Dict:
        """Retorna informações básicas dos dados"""
        if self.sales_train is None:
            self.load_all_data()
            
        info = {
            'n_items': self.sales_train.shape[0],
            'n_days': self.sales_train.shape[1] - 6,  # Excluindo colunas id, item_id, dept_id, cat_id, store_id, state_id
            'date_range': (self.calendar['date'].min(), self.calendar['date'].max()),
            'n_stores': self.sales_train['store_id'].nunique(),
            'n_categories': self.sales_train['cat_id'].nunique(),
            'n_departments': self.sales_train['dept_id'].nunique(),
            'n_states': self.sales_train['state_id'].nunique()
        }
        
        return info
    
    def preprocess_calendar(self) -> pd.DataFrame:
        """Preprocessa o calendar com features temporais"""
        calendar = self.calendar.copy()
        
        # Converte data
        calendar['date'] = pd.to_datetime(calendar['date'])
        
        # Features temporais
        calendar['year'] = calendar['date'].dt.year
        calendar['month'] = calendar['date'].dt.month
        calendar['day'] = calendar['date'].dt.day
        calendar['dayofweek'] = calendar['date'].dt.dayofweek
        calendar['dayofyear'] = calendar['date'].dt.dayofyear
        calendar['week'] = calendar['date'].dt.isocalendar().week
        calendar['quarter'] = calendar['date'].dt.quarter
        
        # Weekend flag
        calendar['is_weekend'] = calendar['dayofweek'].isin([5, 6]).astype(int)
        
        # Month beginning/end
        calendar['is_month_start'] = calendar['date'].dt.is_month_start.astype(int)
        calendar['is_month_end'] = calendar['date'].dt.is_month_end.astype(int)
        
        # Event flags
        event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        for col in event_cols:
            calendar[f'{col}_null'] = calendar[col].isnull().astype(int)
        
        # SNAP flags já estão prontos
        
        return calendar
    
    def get_memory_usage(self) -> Dict:
        """Retorna o uso de memória dos DataFrames"""
        memory_usage = {}
        
        if self.sales_train is not None:
            memory_usage['sales_train'] = f"{self.sales_train.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        if self.calendar is not None:
            memory_usage['calendar'] = f"{self.calendar.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        if self.sell_prices is not None:
            memory_usage['sell_prices'] = f"{self.sell_prices.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            
        return memory_usage
