"""
Módulo para modelagem com LightGBM e XGBoost
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Any
import pickle
import os

class M5Model:
    """Classe base para modelos M5"""
    
    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.cv_scores = []
        
    def wrmsse_score(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     weights: np.ndarray = None) -> float:
        """
        Calcula WRMSSE (Weighted Root Mean Squared Scaled Error)
        """
        if weights is None:
            weights = np.ones(len(y_true))
            
        # Evita divisão por zero
        y_true_shifted = np.roll(y_true, 1)
        y_true_shifted[0] = y_true[0]
        
        denominator = np.mean((y_true - y_true_shifted) ** 2)
        if denominator == 0:
            denominator = 1e-8
            
        rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.sqrt(denominator)
        
        return rmsse
    
    def get_lightgbm_params(self) -> Dict:
        """Parâmetros otimizados para LightGBM"""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 128,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,
            'max_depth': -1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.3,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def get_xgboost_params(self) -> Dict:
        """Parâmetros otimizados para XGBoost"""
        return {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 50,
            'reg_alpha': 0.1,
            'reg_lambda': 0.3,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    def prepare_training_data(self, data: pd.DataFrame, feature_cols: List[str],
                            target_col: str = 'demand') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara dados para treinamento"""
        # Remove linhas com NaN nas features críticas
        data_clean = data.dropna(subset=feature_cols + [target_col])
        
        X = data_clean[feature_cols]
        y = data_clean[target_col]
        
        # Converte apenas colunas numéricas para float32
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            X[col] = X[col].astype(np.float32)
        
        y = y.astype(np.float32)
        
        return X, y
    
    def time_series_split_train(self, X: pd.DataFrame, y: pd.Series, 
                               n_splits: int = 3) -> List[Dict]:
        """Treina modelo com Time Series Cross Validation"""
        print(f"Iniciando Time Series CV com {n_splits} splits...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"Fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if self.model_type == 'lightgbm':
                result = self._train_lightgbm(X_train, y_train, X_val, y_val)
            else:
                result = self._train_xgboost(X_train, y_train, X_val, y_val)
            
            result['fold'] = fold + 1
            cv_results.append(result)
            
        self.cv_scores = cv_results
        return cv_results
    
    def _train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Treina modelo LightGBM"""
        params = self.get_lightgbm_params()
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Predições
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Métricas
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        self.model = model
        self.feature_importance = importance
        
        return {
            'model': model,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'feature_importance': importance,
            'best_iteration': model.best_iteration
        }
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Treina modelo XGBoost"""
        params = self.get_xgboost_params()
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=evallist,
            early_stopping_rounds=100,
            verbose_eval=False
        )
        
        # Predições
        y_pred_train = model.predict(dtrain)
        y_pred_val = model.predict(dval)
        
        # Métricas
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': list(model.get_score(importance_type='gain').values())
        }).sort_values('importance', ascending=False)
        
        self.model = model
        self.feature_importance = importance
        
        return {
            'model': model,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'feature_importance': importance,
            'best_iteration': model.best_iteration
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faz predições com o modelo treinado"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda!")
        
        if self.model_type == 'lightgbm':
            return self.model.predict(X)
        else:
            dtest = xgb.DMatrix(X)
            return self.model.predict(dtest)
    
    def save_model(self, filepath: str):
        """Salva o modelo"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda!")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'cv_scores': self.cv_scores
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str):
        """Carrega o modelo"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data['feature_importance']
        self.cv_scores = model_data['cv_scores']
        
        print(f"Modelo carregado de: {filepath}")
    
    def get_cv_summary(self) -> pd.DataFrame:
        """Retorna resumo dos resultados de CV"""
        if not self.cv_scores:
            return pd.DataFrame()
        
        summary = pd.DataFrame(self.cv_scores)
        
        # Estatísticas resumidas
        metrics = ['train_rmse', 'val_rmse', 'train_mae', 'val_mae']
        summary_stats = {}
        
        for metric in metrics:
            summary_stats[f'{metric}_mean'] = summary[metric].mean()
            summary_stats[f'{metric}_std'] = summary[metric].std()
        
        return pd.DataFrame([summary_stats])

class M5Ensemble:
    """Classe para ensemble de modelos"""
    
    def __init__(self):
        self.models = []
        self.weights = []
        
    def add_model(self, model: M5Model, weight: float = 1.0):
        """Adiciona modelo ao ensemble"""
        self.models.append(model)
        self.weights.append(weight)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predição com ensemble ponderado"""
        if not self.models:
            raise ValueError("Nenhum modelo no ensemble!")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Média ponderada
        weights = np.array(self.weights) / np.sum(self.weights)
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
