"""
Script simplificado para demonstração do projeto M5
"""
import sys
import os

# Adiciona src ao path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

from data_loader import M5DataLoader
from feature_engineering import M5FeatureEngineer
from modeling import M5Model
from utils import MemoryManager
import config
import pandas as pd
import numpy as np

def main():
    """Execução simplificada do projeto M5"""
    print("🚀 Iniciando demonstração M5 Forecasting...")
    
    # 1. Carrega dados
    print("\n📁 Carregando dados...")
    with MemoryManager():
        loader = M5DataLoader(config.DATA_PATH)
        data_dict = loader.load_all_data()
        calendar_processed = loader.preprocess_calendar()
        
        # Informações básicas
        basic_info = loader.get_basic_info()
        print(f"📊 Itens: {basic_info['n_items']:,}")
        print(f"📊 Lojas: {basic_info['n_stores']}")
        print(f"📊 Dias: {basic_info['n_days']}")
    
    # 2. Cria amostra pequena para demonstração
    print("\n⚙️ Criando features (amostra de 500 itens)...")
    feature_eng = M5FeatureEngineer()
    
    # Amostra muito pequena para demonstração
    sales_sample = data_dict['sales_train'].sample(n=500, random_state=42)
    
    with MemoryManager():
        # Converte para formato long
        sample_melted = feature_eng.create_melted_data(sales_sample, calendar_processed)
        
        # Adiciona apenas algumas features básicas
        sample_melted = feature_eng.add_price_features(sample_melted, data_dict['sell_prices'])
        sample_melted = feature_eng.add_lag_features(sample_melted, lags=[1, 7])
        sample_melted = feature_eng.add_rolling_features(sample_melted, windows=[7])
        
        # Remove linhas com NaN
        sample_melted = sample_melted.dropna()
        
        print(f"✅ Dados processados: {sample_melted.shape}")
    
    # 3. Prepara features numéricas apenas
    numeric_features = [
        'wm_yr_wk', 'sell_price', 'price_change', 'price_momentum',
        'demand_lag_1', 'demand_lag_7', 'rolling_mean_7', 'rolling_std_7',
        'year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'quarter',
        'is_weekend', 'is_month_start', 'is_month_end',
        'snap_CA', 'snap_TX', 'snap_WI'
    ]
    
    # Filtra apenas features que existem
    available_features = [f for f in numeric_features if f in sample_melted.columns]
    print(f"📝 Features disponíveis: {len(available_features)}")
    
    # Remove linhas com NaN nas features selecionadas
    clean_data = sample_melted.dropna(subset=available_features + ['demand'])
    
    if len(clean_data) < 100:
        print("❌ Dados insuficientes após limpeza")
        return
    
    print(f"📊 Dados limpos: {len(clean_data):,} registros")
    
    # 4. Treina modelo simples
    print("\n🤖 Treinando modelo simples...")
    
    X = clean_data[available_features].astype(float)
    y = clean_data['demand'].astype(float)
    
    # Divide em treino/teste
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Treina LightGBM simples
    import lightgbm as lgb
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        callbacks=[lgb.log_evaluation(0)]
    )
    
    # Predições
    y_pred = model.predict(X_test)
    
    # Métricas
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"\n📊 Resultados:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Treino: {len(X_train)} amostras")
    print(f"  Teste: {len(X_test)} amostras")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 Top 5 Features:")
    for i, (_, row) in enumerate(importance.head(5).iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.0f}")
    
    print(f"\n✅ Demonstração concluída!")
    print(f"💡 Para resultados completos, execute o notebook M5_Forecasting_Complete.ipynb")

if __name__ == "__main__":
    main()
