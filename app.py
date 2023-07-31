from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Etapa 1: Carregamento dos dados
df_sensor = pd.read_csv('/Users/batistajunior/Heroku/Sensor_FieldPRO.csv')
df_estacao = pd.read_csv('/Users/batistajunior/Heroku/Estacao_Convencional.csv')

# Etapa 2: Verificação dos dados
df_sensor.head()
df_estacao.head()

# Etapa 3: Pré-processamento dos dados
df_sensor['Datetime – utc'] = pd.to_datetime(df_sensor['Datetime – utc'], format='ISO8601', utc=True)
df_sensor['data'] = df_sensor['Datetime – utc'].dt.date
df_sensor['Hora (Brasília)'] = df_sensor['Datetime – utc'].dt.time
df_sensor['data'] = pd.to_datetime(df_sensor['data'])
df_sensor['Hora (Brasília)'] = pd.to_datetime(df_sensor['Hora (Brasília)'], format='%H:%M:%S', errors='coerce').dt.time
df_sensor = df_sensor.drop(columns=['Datetime – utc'])

# Resto do código de pré-processamento...

# Etapa 8: Rota para exibir os gráficos
@app.route('/visualizar-graficos')
def visualizar_graficos():
    # Código da análise exploratória e visualização de gráficos...
    valid_columns = ['air_humidity_100', 'air_temperature_100', 'atm_pressure_main', 'num_of_resets', 'piezo_charge', 'piezo_temperature', 'chuva']

    if all(df_completo[col].notnull().any() for col in valid_columns):
        # Resto do código de visualização de gráficos...

        return render_template('graficos.html', plot_data=plot_data, corr_plot_data=corr_plot_data)

# Etapa 9: Rota para exibir a página inicial
@app.route('/inicio')
def index():
    return 'Bem-vindo ao meu aplicativo Flask no Heroku!'

if __name__ == '__main__':
    app.run()
