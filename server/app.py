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
app.jinja_env.filters['zip'] = zip

# For testing I'm using default weights, have to load trained model later
explainer = Explainer()
PRED_1 = explainer.get()["recommendations"]
OLD_PRED = dict()


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


@app.route('/masked_predict', methods=['POST'])
def masked_predict():
    if request.form.get("feature_mask_1"):
        explainer.mask_at_index(request.form.get("feature_mask_1"))

    if request.form.get("feature_mask_2"):
        explainer.mask_at_index(request.form.get("feature_mask_2"))

    global OLD_PRED, PRED_1
    OLD_PRED = PRED_1[:]

    PRED_1 = explainer.get()["recommendations"]

    titles_1 = [rec["title"] for rec in PRED_1]

    print(titles_1)

    return render_template('home.html', titles_1=titles_1)


@app.route('/comparison', methods=['POST'])
def side_by_side():
    global PRED_1, OLD_PRED
    titles_old = [rec["title"] for rec in OLD_PRED]
    titles_new = [rec["title"] for rec in PRED_1]

    return render_template('comparison.html', preds=[titles_old, titles_new])


@app.route('/model_reset', methods=['POST'])
def reset_model():
    explainer.reset_parameters()

    global PRED_1
    PRED_1 = explainer.get()["recommendations"]

    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
