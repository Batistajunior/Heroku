from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/')
def index():
    return 'Bem-vindo ao meu aplicativo Flask no Heroku!'

def prepare_data():
    # Etapa 1: Carregamento dos dados
    df_sensor = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/desafio-de-dados-fieldpro/main/Sensor_FieldPRO.csv')
    df_estacao = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/desafio-de-dados-fieldpro/main/Estacao_Convencional.csv')

    # Juntando os DataFrames df_sensor e df_estacao com base nas colunas 'data' e 'Hora (Brasília)'
    df_completo = pd.merge(df_sensor, df_estacao, on=['data', 'Hora (Brasília)'], how='inner')

    # Etapa 4: Análise exploratória dos dados e visualização de gráficos
    # Verificando quais colunas têm dados válidos para plotar os gráficos
    valid_columns = ['air_humidity_100', 'air_temperature_100', 'atm_pressure_main', 'num_of_resets', 'piezo_charge', 'piezo_temperature', 'chuva']

    # Resto do código para a análise exploratória dos dados e visualização de gráficos...

    # Etapa 5: Preparação dos dados para treinamento do modelo
    # Definindo as features e o target
    features = ['air_humidity_100', 'air_temperature_100', 'atm_pressure_main', 'num_of_resets', 'piezo_charge', 'piezo_temperature']
    target = 'chuva'

    # Removendo amostras com valores ausentes
    df_completo.dropna(subset=features + [target], inplace=True)

    # Separando os dados em features (X) e target (y)
    X = df_completo[features]
    y = df_completo[target]

    # Dividindo os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preenchendo valores ausentes com a média
    imputer = SimpleImputer(strategy='mean')
    X_train_filled = imputer.fit_transform(X_train)
    X_test_filled = imputer.transform(X_test)

    # Retornando os dados preparados e o df_completo
    return df_completo, X_train_filled, X_test_filled, y_train, y_test

def train_model():
    # Obtendo os dados preparados
    df_completo, X_train_filled, X_test_filled, y_train, y_test = prepare_data()

    # Etapa 6: Treinamento e avaliação do modelo
    # Criando e treinando o modelo de regressão com HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor()
    model.fit(X_train_filled, y_train)

    # Realizando as previsões no conjunto de teste
    y_pred = model.predict(X_test_filled)

    # Avaliando o desempenho do modelo
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Salvando o modelo treinado
    modelo_treinado_path = "modelo_treinado.joblib"
    joblib.dump(model, modelo_treinado_path)

    # Retornando as métricas do modelo
    return mae, mse, r2

@app.route('/analise')
def analise():
    # Obtendo o df_sensor para a análise e visualização de gráficos
    df_completo, _, _, _, _ = prepare_data()

    # Resto do código para a análise de dados e visualização de gráficos...

    # Exemplo de visualização do DataFrame df_sensor:
    return df_completo.head().to_html()

@app.route('/treinamento')
def treinamento():
    # Chama a função de treinamento
    mae, mse, r2 = train_model()

    # Exibindo as métricas do modelo e os dados de treinamento
    return f"Erro Médio Absoluto (MAE): {mae}<br>Erro Quadrático Médio (MSE): {mse}<br>Coeficiente de Determinação (R²): {r2}"

if __name__ == '__main__':
    app.run()
