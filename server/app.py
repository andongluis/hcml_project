from flask import Flask, request, render_template
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
# sys.path.insert(0, os.path.abspath('..'))
from ML_pipeline.custom_model import Custom_Model
import ML_pipeline.sample_explainer as explainer

"""
1. Import model
2. Define input features and match them to website input
3. Render html from here
4. Run predict function
"""

app = Flask(__name__)

# For testing I'm using default weights, have to load trained model later
model = Custom_Model()
cols =["feature columns to collect from site"]

PRED_1 = []
PRED_2 = []

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Could be multiple lists if we initialize user with many movies
    int_features = [x for x in request.form.values()]

    user_data = np.array(int_features)
    # prediction = model.predict(user_data) # Assuming it's a list
    # return render_template('home.html', pred='Predicted movies are {}'.format(prediction))

    PRED_1 = ["Gone with the wind", "The Matrix"] # temp

    return render_template('home.html', first_prediction=PRED_1)


@app.route('/nlg', methods=['POST'])
def get_nlg():
    """
    Need a couple things: a string with the nlg stuff and to generate a list of
    checkbox instructions.
    :return: IDK if possible, but a tuple [nlg, [buttons]] would probably do.
    """
    nlg = "You are seeing this selection because I have determined you have" \
          " absolutely no taste in movies whatsoever. I'd recommend watching" \
          " any of these as a start and see if your sense of curiosity and" \
          " wonder kickstarts itself and helps you get a grip. Also your age" \
          " group is 20-30 and you like scifi, don't argue."

    # This should cover all ML features we have
    features = {"scifi": "science fiction",
                "20-30": "age group"}

    masked_features = ["scifi", "20-30"] # temp
    ret = [nlg, [features[x] for x in masked_features]]

    return render_template('home.html', nlg_component=ret, first_prediction=PRED_1)


@app.route('/masked_predict', methods=['POST', 'GET'])
def masked_predict():
    feature_masks = ["scifi", "20-30"]
    mask_values = []
    for feature in feature_masks:
        mask_value = request.form.get(feature)
        if mask_value:
            mask_values.append(True)
        else:
            mask_values.append(False)

    PRED_2 = ["Gone with the wind", "The Matrix"]

    return render_template('final_result.html',
                           first_prediction=PRED_1,
                           second_prediction=mask_values)


if __name__ == '__main__':
    app.run(debug=True)
