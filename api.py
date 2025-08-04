import pickle
from flask import request, jsonify, Flask
import pandas as pd


app = Flask(__name__)

filename = "lr_model.pkl"
model = pickle.load(open(filename, 'rb'))

@app.route("/",methods=['POST'])
def main_page():
    return "<p>Loan Prediction API !</p>"

@app.route('/predict', methods=['POST'])
def login():
    if request.method == 'POST':
        data = request.json
        sample = pd.DataFrame(data, index=[0])
        prediction = model.predict(sample)
        # print("PREDICTION", prediction)
        response = jsonify({"class": str(prediction[0])})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    

if __name__ == "__main__":
    app.run(debug=True)
