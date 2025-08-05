# 🎯 M5 Forecasting - Previsão de Demanda Walmart

## 📋 Visão Geral

Este projeto implementa um sistema completo de previsão de demanda para o dataset M5 Forecasting da Walmart, utilizando uma arquitetura modular e algoritmos de machine learning avançados (LightGBM e XGBoost).

## 🏗️ Estrutura do Projeto

```
m5_project/
├── src/                          # Código fonte modular
│   ├── __init__.py              # Inicialização do pacote
│   ├── config.py                # Configurações globais
│   ├── data_loader.py           # Carregamento e pré-processamento
│   ├── feature_engineering.py   # Engenharia de features
│   ├── modeling.py              # Modelos LightGBM/XGBoost
│   ├── visualization.py         # Visualizações e análises
│   └── utils.py                 # Utilitários gerais
├── data/                        # Diretório para dados processados
├── models/                      # Modelos treinados salvos
├── outputs/                     # Resultados e submissões
├── M5_Forecasting_Complete.ipynb # Notebook principal
├── requirements.txt             # Dependências
└── README.md                    # Este arquivo
```

## 🚀 Como Executar

### 1. Instalação de Dependências

```bash
pip install -r requirements.txt
```

### 2. Configuração dos Dados

1. Baixe o dataset M5 do Kaggle
2. Coloque os arquivos CSV no diretório `m5-forecasting-accuracy/`
3. Ajuste o caminho em `src/config.py` se necessário

### 3. Execução do Notebook

Abra e execute o notebook `M5_Forecasting_Complete.ipynb` que integra todos os módulos.

## 📊 Funcionalidades Principais

### 🔍 Análise Exploratória
- Overview de vendas por categoria, estado e loja
- Análise de sazonalidade e padrões temporais
- Distribuição de demanda e análise de preços
- Identificação de itens top performers

### ⚙️ Engenharia de Features
- **Features de Lag**: t-1, t-7, t-14, t-28
- **Rolling Features**: médias móveis, desvio padrão, min/max
- **Features de Preço**: mudanças, momentum, preço relativo
- **Features Temporais**: dia da semana, mês, feriados, SNAP
- **Features Estatísticas**: agregações por item/loja/categoria

### 🤖 Modelagem
- **LightGBM**: Otimizado para datasets grandes
- **XGBoost**: Alternativa robusta para comparação
- **Cross-Validation**: Time Series Split para validação temporal
- **Ensemble**: Combinação de múltiplos modelos

### 📈 Avaliação
- Métricas: RMSE, MAE, MAPE
- Visualizações de performance
- Feature importance analysis
- Comparação predições vs valores reais

## 🎛️ Configurações

Edite `src/config.py` para personalizar:

```python
# Caminhos
DATA_PATH = "caminho/para/seus/dados"
MODEL_PATH = "caminho/para/salvar/modelos"

# Features
LAG_FEATURES = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]

# Modelo
MODEL_TYPE = 'lightgbm'  # ou 'xgboost'
CV_SPLITS = 3
```

## 📦 Módulos Principais

### 🔧 `data_loader.py`
- Carregamento eficiente dos dados M5
- Pré-processamento do calendar
- Otimização de memória
- Informações básicas dos dados

### ⚙️ `feature_engineering.py`
- Conversão para formato long
- Criação de features de lag e rolling
- Features de preço e estatísticas
- Encoding de variáveis categóricas

### 🤖 `modeling.py`
- Modelos LightGBM e XGBoost
- Time Series Cross-Validation
- Ensemble de modelos
- Salvamento/carregamento de modelos

### 📊 `visualization.py`
- Análises exploratórias interativas
- Visualizações de performance
- Feature importance plots
- Dashboard resumo

### 🛠️ `utils.py`
- Otimização de memória
- Métricas de avaliação
- Utilitários de tempo e progresso
- Validação de qualidade dos dados

## 🎯 Resultados Esperados

O modelo típico alcança:
- **RMSE**: ~2.0-3.0 (dependendo da amostra)
- **MAE**: ~1.5-2.5
- **Features importantes**: Lags, rolling means, preços, variáveis temporais

## 🚀 Próximos Passos

1. **Rolling Forecast**: Implementar previsão de 28 dias futuros
2. **Hyperparameter Tuning**: Otimização com Optuna
3. **WRMSSE**: Implementar métrica oficial da competição
4. **MLOps**: Pipeline automatizado de retreinamento
5. **Interpretabilidade**: SHAP values para explicabilidade

## 🛡️ Otimizações de Performance

- **Redução de memória**: Conversão automática de tipos
- **Amostragem inteligente**: Para datasets grandes
- **Processamento em chunks**: Para features complexas
- **Garbage collection**: Gestão automática de memória

## 📚 Dependências Principais

- `pandas>=2.0.0`: Manipulação de dados
- `numpy>=1.24.0`: Operações numéricas
- `lightgbm>=4.0.0`: Modelo principal
- `xgboost>=2.0.0`: Modelo alternativo
- `scikit-learn>=1.3.0`: Métricas e validação
- `plotly>=5.15.0`: Visualizações interativas
- `matplotlib>=3.7.0`: Plots estáticos
- `seaborn>=0.12.0`: Visualizações estatísticas

## 🏆 Arquitetura de Solução

### 1. **Modularidade**
- Código organizado em módulos especializados
- Fácil manutenção e extensão
- Reutilização de componentes

### 2. **Escalabilidade**
- Otimização de memória automática
- Processamento por chunks
- Amostragem inteligente

### 3. **Robustez**
- Validação temporal adequada
- Tratamento de dados faltantes
- Métricas múltiplas de avaliação

### 4. **Flexibilidade**
- Múltiplos algoritmos suportados
- Configurações parametrizáveis
- Ensemble de modelos

## 📞 Suporte

Para dúvidas ou melhorias:
1. Verifique a documentação nos módulos
2. Execute o notebook passo a passo
3. Ajuste configurações conforme sua capacidade computacional

---

**✨ Desenvolvido com foco em qualidade, performance e escalabilidade para previsão de demanda em varejo.**
