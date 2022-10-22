from flask import Flask, request, jsonify
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

appFlask = Flask(__name__)
appFlask = Flask("credit_card")

with open("model2.bin", "rb") as f:
    model = pickle.load(f)
with open("dv.bin", "rb") as f:
    dv = pickle.load(f)

@appFlask.route('/index')
def index():
    return "Hello World!"

@appFlask.route('/predict', methods=["POST"])
def predict():
    customer = request.get_json()
    x = dv.transform(customer)
    print(x)
    y_pred = model.predict_proba(x)[0,1]
    print(y_pred)

    result = {"credit_card_prob": float(y_pred),
            "credit_card": bool(y_pred > 0.5)}

    return jsonify(result)

if __name__ == "__main__":
    appFlask.run(debug=True, host='0.0.0.0', port=9696)