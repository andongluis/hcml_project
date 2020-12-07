from flask import Flask, request, render_template
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
# sys.path.insert(0, os.path.abspath('..'))
from ML_pipeline.custom_model import Custom_Model
from ML_pipeline.sample_explainer import Explainer

"""
1. Import model
2. Define input features and match them to website input
3. Render html from here
4. Run predict function
"""

app = Flask(__name__)

# For testing I'm using default weights, have to load trained model later
explainer = Explainer()
cols =["feature columns to collect from site"]

PRED_1 = explainer.get()["recommendations"]
PRED_2 = dict()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    titles = [rec["title"] for rec in PRED_1]

    return render_template('home.html', titles_1=titles)


@app.route('/nlg', methods=['POST'])
def show_nlg():
    title = request.form.get("movie_1")
    rec = dict()
    for pred in PRED_1:
        if pred['title'] == title:
            rec = pred

    return render_template('home.html', rec=rec)


@app.route('/masked_predict', methods=['POST', 'GET'])
def masked_predict():
    PRED_2 = ["Gone with the wind", "The Matrix"]

    return render_template('final_result.html',
                           first_prediction=PRED_1,
                           second_prediction=PRED_2)


if __name__ == '__main__':
    app.run(debug=True)
