import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from model import tfidf_vectorizer

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    # Récupérer la description du film à partir du formulaire HTML
    description = request.form["question"]
    description_tfidf = tfidf_vectorizer.transform([description])
    # Utiliser le modèle pour prédire la classe de la fleur en fonction de la description
    prediction = model.predict(description_tfidf)  # Assurez-vous que votre modèle prend en entrée la description du film
    # Renvoyer le résultat de la prédiction à la page HTML
    return render_template("index.html", prediction_text="The movie genre is {}".format(prediction))


if __name__ == "__main__":
    flask_app.run(debug=True)