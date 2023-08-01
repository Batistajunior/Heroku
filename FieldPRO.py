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

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# DataFrame para armazenar os dados
df_completo = None

@app.route('/')
def index_inicio():
    return 'Bem-vindo ao meu aplicativo Flask no Heroku!'

def prepare_data():
    global df_completo

    # Carrega os DataFrames df_sensor e df_estacao, se ainda não foram carregados
    if df_completo is None:
        df_sensor = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/desafio-de-dados-fieldpro/main/Sensor_FieldPRO.csv')
        df_estacao = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/desafio-de-dados-fieldpro/main/Estacao_Convencional.csv')

        # Juntando os DataFrames df_sensor e df_estacao com base nas colunas 'data' e 'Hora (Brasília)'
        df_completo = pd.merge(df_sensor, df_estacao, on=['data', 'Hora (Brasília)'], how='inner')

    return df_completo

@app.route('/analise')
def analise():
    # Obtendo o df_sensor para a análise e visualização de gráficos
    df_completo = prepare_data()

    # Exemplo de criação de gráficos
    plt.figure(figsize=(10, 6))
    sns.histplot(df_completo['chuva'], kde=True)
    plt.title('Distribuição dos dados de chuva')
    plt.xlabel('Chuva')
    plt.ylabel('Contagem')
    plt.tight_layout()

    # Salvar o gráfico em um buffer de bytes para inserir no template HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()

    # Renderizar o template HTML com o gráfico
    return render_template('analise.html', plot_data=plot_data)

def train_model():
    # Obtendo os dados preparados
    df_completo = prepare_data()

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

@app.route('/treinamento')
def treinamento():
    # Chama a função de treinamento
    mae, mse, r2 = train_model()

    # Exibindo as métricas do modelo e os dados de treinamento
    return f"Erro Médio Absoluto (MAE): {mae}<br>Erro Quadrático Médio (MSE): {mse}<br>Coeficiente de Determinação (R²): {r2}"



@app.route('/visualizar-graficos')
def visualizar_graficos():
    # Etapa 1: Carregamento dos dados
    df_sensor = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/Heroku/main/Sensor_FieldPRO.csv')
    df_estacao = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/Heroku/main/Estacao_Convencional.csv')

    @app.route('/visualizar-graficos')
    def visualizar_graficos():
    # Etapa 1: Carregamento dos dados
     df_sensor = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/desafio-de-dados-fieldpro/main/Sensor_FieldPRO.csv')
    df_estacao = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/desafio-de-dados-fieldpro/main/Estacao_Convencional.csv')

    # Juntando os DataFrames df_sensor e df_estacao com base nas colunas 'data' e 'Hora (Brasília)'
    df_completo = pd.merge(df_sensor, df_estacao, on=['data', 'Hora (Brasília)'], how='inner')

    # Etapa 2: Visualização de gráficos
    # Exemplo de criação de gráfico de chuva ao longo do tempo
    plt.figure(figsize=(10, 6))
    plt.plot(df_completo['data'], df_completo['chuva'], label='Chuva')
    plt.xlabel('Data')
    plt.ylabel('Chuva')
    plt.title('Gráfico de Chuva ao longo do tempo')
    plt.legend()

    # Salvar o gráfico em um buffer de bytes para inserir no template HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close()

    # Etapa 3: Matriz de correlação
    # Selecionando as colunas relevantes para a matriz de correlação
    valid_columns = ['air_humidity_100', 'air_temperature_100', 'atm_pressure_main', 'num_of_resets', 'piezo_charge', 'piezo_temperature', 'chuva']
    correlation_matrix = df_completo[valid_columns].corr()

    # Plotando a matriz de correlação
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, linecolor='black', annot_kws={"size": 10})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Matriz de Correlação', fontsize=14)

    # Salvar a matriz de correlação em um buffer de bytes para inserir no template HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    correlation_plot_data = base64.b64encode(buffer.read()).decode()
    plt.close()

    # Renderizar o template HTML com os gráficos
    return render_template('visualizar_graficos.html', plot_data=plot_data, correlation_plot_data=correlation_plot_data)


    return "Gráficos visualizados com sucesso!"

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

