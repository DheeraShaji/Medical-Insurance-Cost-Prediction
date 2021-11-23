from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

#create flask app
app = Flask(__name__)

#load pickle
model = pickle.load(open("neural_network_regression_model.pkl", "rb"))

#create homepage
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST','GET'])
def predict():
    float_features =[float(x) for x in request.form.values()]
    region_southeast=0
    region_southwest=0
    float_features.extend((region_southeast,region_southwest))
    print(float_features)
    features = np.array(float_features).reshape((1,8))
    print(features)
    prediction = model.predict(features)
    output=round(prediction[0],2)
    if output<0:
        return render_template('index.html',prediction_text="Sorry you cannot get medical insurance price")
    else:
        return render_template('index.html',prediction_text="You can get medical insurance at {} dollars".format(output))

if __name__=="__main__":
    app.run(debug=True)