import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

import warnings
warnings.filterwarnings("ignore")

# Etapa 1: Carregamento dos dados
df_sensor = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/Heroku/main/df_sensor.csv')
df_estacao = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/Heroku/main/Estacao_Convencional.csv')

# Etapa 2: Juntando os DataFrames df_sensor e df_estacao com base nas colunas 'data' e 'Hora (Brasília)'
df_completo = pd.merge(df_sensor, df_estacao, on=['data', 'Hora (Brasília)'], how='inner')

# Etapa 4: Análise exploratória dos dados e visualização de gráficos
# Verificando quais colunas têm dados válidos para plotar os gráficos
valid_columns = ['air_humidity_100', 'air_temperature_100', 'atm_pressure_main', 'num_of_resets', 'piezo_charge', 'piezo_temperature', 'chuva']

# Verificando se há alguma coluna com dados válidos para plotar os gráficos
if all(df_completo[col].notnull().any() for col in valid_columns):
    # Visualizando a distribuição das variáveis e plotando os gráficos
    fig_hist = go.Figure()
    for col in valid_columns:
        fig_hist.add_trace(go.Histogram(x=df_completo[col], nbinsx=20, name=col))
    fig_hist.update_layout(title='Distribuição das Variáveis', barmode='overlay', xaxis_title='Valor', yaxis_title='Contagem')

    # Verificando a correlação entre as variáveis numéricas
    correlation_matrix = df_completo[valid_columns].corr()

    # Plotando a matriz de correlação
    fig_corr = px.imshow(correlation_matrix)

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

print("Erro Médio Absoluto (MAE):", mae)
print("Erro Quadrático Médio (MSE):", mse)
print("Coeficiente de Determinação (R²):", r2)

# Salvando o modelo em um arquivo joblib
modelo_treinado_path = "modelo_treinado.joblib"
joblib.dump(model, modelo_treinado_path)

# Criação do aplicativo Dash
app = dash.Dash(__name__)

# Layout do aplicativo
app.layout = html.Div(children=[
    html.H1('Análise Exploratória de Dados'),
    dcc.Graph(id='histogram-plots', figure=fig_hist),
    dcc.Graph(id='correlation-matrix-plot', figure=fig_corr)
])

if __name__ == '__main__':
    app.run_server(debug=True)
