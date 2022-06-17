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
    <h1>fetus deletus</h1>
    yeetus that fetus:
    <ul>
    <li>baseline value</li>
    <li>accelerations</li>
    </ul>
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