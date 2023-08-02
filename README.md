# Análise e Previsão de Chuvas

Este projeto tem como objetivo realizar a análise exploratória de dados e criar um modelo de previsão de chuvas usando o algoritmo HistGradientBoostingRegressor. O projeto utiliza dados de sensores climáticos e informações de uma estação convencional para realizar a previsão.

Requisitos

Antes de executar o projeto, certifique-se de ter as seguintes bibliotecas instaladas:

pandas
matplotlib
seaborn
scikit-learn
joblib
Você pode instalá-las usando o gerenciador de pacotes 'pip' com o seguinte comando:

pip install pandas matplotlib seaborn scikit-learn joblib


Executando o Projeto

Siga os passos abaixo para executar o projeto:

Clone o repositório para o seu computador:

git clone https://github.com/seu-usuario/nome-do-repositorio.git


Navegue até o diretório do projeto:
cd nome-do-repositorio

Crie e ative um ambiente virtual (opcional, mas recomendado):

python -m venv env
source env/bin/activate


Instale as dependências do projeto:

pip install -r requirements.txt


python3 FieldPRO.py


Acesse o aplicativo no navegador:
Abra o navegador e acesse http://127.0.0.1:8050/ para visualizar o aplicativo web.

Notebook de Análise e Treinamento

O notebook notebooks/Analise_e_Previsao_de_Chuvas.ipynb contém o código utilizado para realizar a análise exploratória dos dados e o treinamento do modelo de previsão de chuvas. Nele, você encontrará todas as etapas do projeto detalhadas, desde o carregamento dos dados até a avaliação do modelo.

Deploy no Heroku

O projeto também foi implantado no Heroku para que seja possível acessá-lo pela web. Para visualizar o aplicativo online, basta acessar o link abaixo:

Link para o aplicativo no Heroku

https://seu-aplicativo-heroku.herokuapp.com

Observações


Para fazer o deploy do aplicativo no Heroku, foram seguidos os passos descritos no README. Certifique-se de ter a conta no Heroku e as dependências corretas instaladas.
O aplicativo pode ser acessado publicamente pelo link fornecido no deploy no Heroku. 

Este Desafio foi desenvolvido por Batistajunior.



