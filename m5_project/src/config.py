"""
Configurações do projeto M5
"""

# Caminhos
DATA_PATH = r"c:\Users\thiago.santos\Desktop\PESSOAL\RandomF_XGB\m5-forecasting-accuracy"
OUTPUT_PATH = r"c:\Users\thiago.santos\Desktop\PESSOAL\RandomF_XGB\m5_project\outputs"
MODEL_PATH = r"c:\Users\thiago.santos\Desktop\PESSOAL\RandomF_XGB\m5_project\models"

# Parâmetros de feature engineering
LAG_FEATURES = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]
VALIDATION_DAYS = 28
FORECAST_HORIZON = 28

# Parâmetros de modelagem
MODEL_TYPE = 'lightgbm'  # 'lightgbm' ou 'xgboost'
CV_SPLITS = 3
EARLY_STOPPING_ROUNDS = 100
MAX_BOOST_ROUNDS = 2000

# Parâmetros LightGBM
LIGHTGBM_PARAMS = {
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

# Parâmetros XGBoost
XGBOOST_PARAMS = {
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

# Configurações de visualização
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (12, 8)
DPI = 100

# Configurações de memória
MEMORY_OPTIMIZATION = True
CHUNK_SIZE = 50000

# Random seed para reprodutibilidade
RANDOM_SEED = 42
