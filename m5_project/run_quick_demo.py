"""
Script principal para execução rápida do projeto M5
Execute este script para treinar um modelo básico rapidamente
"""
import sys
import os

# Adiciona src ao path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

from data_loader import M5DataLoader
from feature_engineering import M5FeatureEngineer
from modeling import M5Model
from visualization import M5Visualizer
from utils import MemoryManager, reduce_memory_usage
import config

def main():
    """Execução principal do projeto M5"""
    print("🚀 Iniciando projeto M5 Forecasting...")
    
    # 1. Carrega dados
    print("\n📁 Carregando dados...")
    with MemoryManager():
        loader = M5DataLoader(config.DATA_PATH)
        data_dict = loader.load_all_data()
        calendar_processed = loader.preprocess_calendar()
    
    # 2. Engenharia de features (amostra pequena para demo)
    print("\n⚙️ Criando features...")
    feature_eng = M5FeatureEngineer()
    
    # Usa amostra pequena para demonstração
    sales_sample = data_dict['sales_train'].sample(n=1000, random_state=42)
    
    with MemoryManager():
        feature_data = feature_eng.create_all_features(
            sales_sample,
            calendar_processed,
            data_dict['sell_prices']
        )
    
    # 3. Prepara dados para modelagem
    print("\n🔧 Preparando dados para modelagem...")
    feature_cols = feature_eng.get_feature_list(feature_data)
    
    # Remove NaN
    threshold = len(feature_cols) * 0.7
    feature_data_clean = feature_data.dropna(subset=feature_cols, thresh=threshold)
    
    print(f"Dados finais: {feature_data_clean.shape}")
    print(f"Features: {len(feature_cols)}")
    
    # 4. Treina modelo
    print("\n🤖 Treinando modelo LightGBM...")
    model = M5Model(model_type='lightgbm')
    
    X, y = model.prepare_training_data(feature_data_clean, feature_cols)
    cv_results = model.time_series_split_train(X, y, n_splits=2)
    
    # 5. Resultados
    print("\n📊 Resultados:")
    for i, result in enumerate(cv_results):
        print(f"  Fold {i+1}: RMSE={result['val_rmse']:.4f}, MAE={result['val_mae']:.4f}")
    
    # 6. Salva modelo
    print("\n💾 Salvando modelo...")
    os.makedirs(config.MODEL_PATH, exist_ok=True)
    model_path = os.path.join(config.MODEL_PATH, "m5_quick_model.pkl")
    model.save_model(model_path)
    
    # 7. Feature importance
    print("\n🎯 Top 10 Features:")
    print(model.feature_importance.head(10))
    
    print("\n✅ Execução concluída!")
    print(f"Modelo salvo em: {model_path}")

if __name__ == "__main__":
    main()
