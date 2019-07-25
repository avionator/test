from flask import Flask, render_template, redirect, url_for, request, make_response, jsonify
from sklearn.externals import joblib
import requests
import json
import numpy as np

app = Flask(__name__)


@app.route("/")
#donne l'url de la page d'acceuil, c'est un decorateur
#c'est relié par défaut à la page index
def index():


    response = make_response(render_template("index.html"))
#la variable forecast sera dark_sky lorsque la page index_html sera utilisé
    return response

@app.route("/predict", methods=['POST'])
#methods=[POST] permet de pouvoir faire un request , voir index.html
def predict():
    if request.method=='POST':

            regressor = joblib.load("./linear_regression_model.pkl")
            data = dict(request.form.items())
#contenu de la balise form
            years_of_experience = np.array(float(data["YearsExperience"])).reshape(-1,1)
            prediction = regressor.predict(years_of_experience)
            response = make_response(render_template(
            "predicted.html",
            prediction = float(prediction)
            ))


            return response


if __name__ == '__main__':
    app.run(debug=True)
