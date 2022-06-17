from flask import Flask, jsonify, request
import pandas as pd
import json
from xgboost import XGBClassifier

app = Flask(__name__)

def return_prediction(model, input_json):
    input_data = pd.DataFrame(input_json, index=[0])
    prediction = model.predict(input_data)[0]

    return prediction

model = XGBClassifier()
model.load_model("models/model.json")

@app.route("/")
def index():
    return """
    <h1>Welcome to the API for Hera</h1>
    Hera is an ML-based predictor that classifies fetal health based on data from Cardiotocography (CTG) exams. It has an accuracy of 0.96

    Our endpoint is at https://hera-predictor-api.herokuapp.com/predict

    Send a POST request with a JSON containing:
    <ul>
    <li>baseline value</li>
    <li>accelerations</li>
    <li>fetal_movement</li>
    <li>uterine_contractions</li>
    <li>light_decelerations</li>
    <li>severe_decelerations</li>
    <li>prolongued_decelerations</li>
    <li>abnormal_short_term_variability</li>
    <li>mean_value_of_short_term_variability</li>
    <li>percentage_of_time_with_abnormal_long_term_variability</li>
    <li>mean_value_of_long_term_variability</li>
    <li>histogram_width</li>
    <li>histogram_min</li>
    <li>histogram_max</li>
    <li>histogram_number_of_peaks</li>
    <li>histogram_number_of_zeroes</li>
    <li>histogram_mode</li>
    <li>histogram_mean</li>
    <li>histogram_median</li>
    <li>histogram_variance</li>
    <li>histogram_tendency</li>
    </ul>

    It returns 1.0 for healthy, 2.0 for suspect, and 3.0 for pathological

    Credits to the dataset this model was trained on:
    Ayres de Campos et al. (2000) SisPorto 2.0 A Program for Automated Analysis of Cardiotocograms. J Matern Fetal Med 5:311-318
    """

@app.route('/predict', methods=['POST'])
def fetal_health_prediction():
    content = request.json
    print(content)
    results = return_prediction(model, content)
    return jsonify(results)


headers = ['baseline value', 'accelerations', 'fetal_movement',
       'uterine_contractions', 'light_decelerations', 'severe_decelerations',
       'prolongued_decelerations', 'abnormal_short_term_variability',
       'mean_value_of_short_term_variability',
       'percentage_of_time_with_abnormal_long_term_variability',
       'mean_value_of_long_term_variability', 'histogram_width',
       'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
       'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
       'histogram_median', 'histogram_variance', 'histogram_tendency']

if __name__ == "__main__":
    app.run(debug=True)