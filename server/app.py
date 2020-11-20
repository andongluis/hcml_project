from flask import Flask, request, render_template
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
# sys.path.insert(0, os.path.abspath('..'))
from ML_pipeline.custom_model import Custom_Model

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


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Could be multiple lists if we initialize user with many movies
    int_features = [x for x in request.form.values()]

    user_data = np.array(int_features)
    prediction = model.predict(user_data) # Assuming it's a list
    return render_template('home.html', pred='Predicted movies are {}'.format(prediction))
