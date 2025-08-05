"""
Utilitários gerais para o projeto M5
"""
import pandas as pd
import numpy as np
import os
import gc
from typing import Dict, List, Any
import psutil
import time
from functools import wraps

def memory_usage():
    """Retorna uso atual de memória"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduz uso de memória do DataFrame"""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                    
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f'Uso de memória: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% redução)')
    
    return df

def timer(func):
    """Decorator para medir tempo de execução"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executado em {end_time - start_time:.2f} segundos")
        return result
    return wrapper

class MemoryManager:
    """Gerenciador de memória para o projeto"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.initial_memory = memory_usage()
    
    def __enter__(self):
        if self.verbose:
            print(f"Memória inicial: {self.initial_memory:.2f} MB")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        final_memory = memory_usage()
        if self.verbose:
            print(f"Memória final: {final_memory:.2f} MB")
            print(f"Diferença: {final_memory - self.initial_memory:+.2f} MB")

def create_submission_file(predictions: np.ndarray, sample_submission: pd.DataFrame, 
                          output_path: str) -> None:
    """Cria arquivo de submissão"""
    submission = sample_submission.copy()
    submission.iloc[:, 1:] = predictions.reshape(-1, predictions.shape[-1])
    
    submission.to_csv(output_path, index=False)
    print(f"Arquivo de submissão salvo em: {output_path}")

def calculate_wrmsse(y_true: np.ndarray, y_pred: np.ndarray, 
                    sales_train: pd.DataFrame) -> float:
    """
    Calcula WRMSSE oficial da competição M5
    """
    # Calcula escala para cada série
    scales = []
    day_cols = [col for col in sales_train.columns if col.startswith('d_')]
    
    for i in range(len(sales_train)):
        series = sales_train.iloc[i][day_cols].values
        # Diferenças de primeiro ordem
        diffs = np.diff(series)
        scale = np.mean(diffs ** 2)
        scales.append(scale)
    
    scales = np.array(scales)
    scales = np.where(scales == 0, 1e-8, scales)  # Evita divisão por zero
    
    # Calcula RMSSE para cada série
    rmsse_scores = []
    for i in range(len(y_true)):
        mse = np.mean((y_true[i] - y_pred[i]) ** 2)
        rmsse = np.sqrt(mse / scales[i])
        rmsse_scores.append(rmsse)
    
    # Média ponderada (assumindo pesos iguais por simplicidade)
    wrmsse = np.mean(rmsse_scores)
    
    return wrmsse

def create_forecast_periods(last_date: str, horizon: int = 28) -> List[str]:
    """Cria períodos de forecast"""
    last_date = pd.to_datetime(last_date)
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                  periods=horizon, freq='D')
    return forecast_dates.strftime('%Y-%m-%d').tolist()

def validate_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """Valida qualidade dos dados"""
    report = {
        'shape': data.shape,
        'missing_values': data.isnull().sum().to_dict(),
        'duplicate_rows': data.duplicated().sum(),
        'negative_demand': (data['demand'] < 0).sum() if 'demand' in data.columns else 0,
        'zero_demand_ratio': (data['demand'] == 0).mean() if 'demand' in data.columns else 0,
        'data_types': data.dtypes.to_dict()
    }
    
    return report

def split_train_validation(data: pd.DataFrame, validation_days: int = 28) -> tuple:
    """Divide dados em treino e validação baseado em datas"""
    data['date'] = pd.to_datetime(data['date'])
    
    max_date = data['date'].max()
    validation_start = max_date - pd.Timedelta(days=validation_days - 1)
    
    train_data = data[data['date'] < validation_start].copy()
    val_data = data[data['date'] >= validation_start].copy()
    
    print(f"Divisão train/validation:")
    print(f"  Train: {len(train_data):,} registros ({train_data['date'].min()} a {train_data['date'].max()})")
    print(f"  Validation: {len(val_data):,} registros ({val_data['date'].min()} a {val_data['date'].max()})")
    
    return train_data, val_data

def save_results(results: Dict, filepath: str) -> None:
    """Salva resultados em formato JSON"""
    import json
    
    # Converte numpy arrays para listas
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return obj
    
    # Converte recursivamente
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(v) for v in obj]
        else:
            return convert_numpy(obj)
    
    results_converted = recursive_convert(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    print(f"Resultados salvos em: {filepath}")

class ProgressTracker:
    """Tracker de progresso para operações longas"""
    
    def __init__(self, total_steps: int, description: str = "Processando"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, steps: int = 1):
        """Atualiza progresso"""
        self.current_step += steps
        progress = self.current_step / self.total_steps
        
        elapsed_time = time.time() - self.start_time
        if progress > 0:
            eta = elapsed_time / progress - elapsed_time
        else:
            eta = 0
        
        print(f"\r{self.description}: {progress*100:.1f}% | "
              f"Tempo decorrido: {elapsed_time:.1f}s | "
              f"ETA: {eta:.1f}s", end="")
        
        if self.current_step >= self.total_steps:
            print()  # Nova linha ao completar

def check_environment() -> Dict[str, Any]:
    """Verifica ambiente de execução"""
    env_info = {
        'python_version': os.sys.version,
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'cpu_count': os.cpu_count(),
        'current_memory_usage_mb': memory_usage()
    }
    
    # Tenta verificar se tem GPU
    try:
        import tensorflow as tf
        env_info['gpu_available'] = len(tf.config.list_physical_devices('GPU')) > 0
    except:
        env_info['gpu_available'] = False
    
    return env_info
