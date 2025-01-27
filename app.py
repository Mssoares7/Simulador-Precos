# app flask

# Imprts
from flask import Flask, request
from flask import render_template
from tools.car import Car

# Cria a app
app = Flask(__name__)

# Página de entrada
@app.route("/")
def index():
    result = None
    return render_template("index.html", result = result)

# Página com resultado da previsão
@app.route("/estimate", methods=["POST"])
def estimate():
    values = request.form.getlist('new_car')
    car = Car(values)
    value_to_predict = car.prepare()
    result = car.predict(value_to_predict)
    # Formatação no padrão brasileiro
    result = "{:,.2f}".format(result).replace(',', 'X').replace('.', ',').replace('X', '.')
    return render_template('index.html', result=result)

# Executa a app
if __name__ == "__main__":
    app.run(host = 'localhost', port = 3000, debug = True)
