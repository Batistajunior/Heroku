# Importe as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, Response
import base64
import io

app = Flask(__name__)

# Carregamento dos dados
df_sensor = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/desafio-de-dados-fieldpro/main/Sensor_FieldPRO.csv')
df_estacao = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/desafio-de-dados-fieldpro/main/Estacao_Convencional.csv')

def create_plot():
    # Exemplo de criação de gráfico usando Matplotlib e Seaborn
    plt.figure(figsize=(10, 6))
    plt.plot(df_sensor['data'], df_sensor['chuva'], label='Chuva')
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

    return plot_data

@app.route('/')
def index_inicio():
    return 'Bem-vindo ao meu aplicativo Flask no Heroku!'

@app.route('/analise')
def analise():
    # Etapa 1: Carregamento dos dados
    df_sensor = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/desafio-de-dados-fieldpro/main/Sensor_FieldPRO.csv')
    df_estacao = pd.read_csv('https://raw.githubusercontent.com/Batistajunior/desafio-de-dados-fieldpro/main/Estacao_Convencional.csv')

    # Exemplo de criação de gráfico usando Matplotlib e Seaborn
    plt.figure(figsize=(10, 6))
    plt.plot(df_sensor['data'], df_sensor['chuva'], label='Chuva')
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

    # Renderizar o template HTML com o gráfico
    return render_template('analise.html', plot_data=plot_data)

@app.route('/visualizar-graficos')
def visualizar_graficos():
    # Criação dos gráficos
    plot_data_1 = create_plot()

    # Renderizar o template HTML com os gráficos
    return render_template('visualizar_graficos.html', plot_data_1=plot_data_1)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
