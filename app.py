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
import base64
import io
import matplotlib.pyplot as plt

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

    # Visualizando a distribuição das variáveis e plotando os gráficos
    fig, axes = plt.subplots(1, 7, figsize=(24, 5))
    for i, col in enumerate(valid_columns):
        sns.histplot(df_completo[col], bins=20, edgecolor='black', color='skyblue', alpha=0.8, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()

    # Codificando o gráfico em formato de imagem para base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    graph_html = f'<img src="data:image/png;base64,{image_base64}" alt="Gráficos de Variáveis">'

    # Verificando a correlação entre as variáveis numéricas
    correlation_matrix = df_completo[valid_columns].corr()

    # Plotando a matriz de correlação
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, linecolor='black', annot_kws={"size": 10})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Matriz de Correlação', fontsize=14)

    # Codificando o gráfico de matriz de correlação em formato de imagem para base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64_corr = base64.b64encode(buffer.getvalue()).decode()
    corr_graph_html = f'<img src="data:image/png;base64,{image_base64_corr}" alt="Matriz de Correlação">'

    return df_completo, graph_html, corr_graph_html

def train_model():
    # Resto do código para treinamento do modelo...

 @app.route('/analise')
 def analise():
    df_completo, graph_html, corr_graph_html = prepare_data()
    # Exemplo de visualização do DataFrame df_sensor e dos gráficos na página HTML
    return f"{df_completo.head().to_html()}<br>{graph_html}<br>{corr_graph_html}"

@app.route('/treinamento')
def treinamento():
    # Chama a função de treinamento
    mae, mse, r2 = train_model()

    # Exibindo as métricas do modelo e os dados de treinamento
    return f"Erro Médio Absoluto (MAE): {mae}<br>Erro Quadrático Médio (MSE): {mse}<br>Coeficiente de Determinação (R²): {r2}"

if __name__ == '__main__':
    app.run()
