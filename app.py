from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return 'Bem-vindo ao meu aplicativo Flask no Heroku!'

if __name__ == '__main__':
    app.run()
