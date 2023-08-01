from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

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
        df_sensor = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/Heroku/main/Sensor_FieldPRO.csv')
        df_estacao = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/Heroku/main/Estacao_Convencional.csv')

        # Juntando os DataFrames df_sensor e df_estacao com base nas colunas 'data' e 'Hora (Brasília)'
        df_completo = pd.merge(df_sensor, df_estacao, on=['data', 'Hora (Brasília)'], how='inner')

    return df_completo

@app.route('/visualizar-graficos')
def visualizar_graficos():
    # Etapa 1: Carregamento dos dados
    df_sensor = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/Heroku/main/Sensor_FieldPRO.csv')
    df_estacao = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/Heroku/main/Estacao_Convencional.csv')

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


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
