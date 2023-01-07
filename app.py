import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np

app=Flask(__name__)
#load pickle file
model=pickle.load(open('diabetes_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(np.array(list(data.values())).reshape(1,-1))
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    print(data)
    print(np.array(data).reshape(1,-1))
    output=model.predict(np.array(data).reshape(1,-1))[0]
    print(output)
    if output==0:
            return render_template('index.html',prediction_text="Great!! you don't have diabetes")
    else:
            return render_template('index.html',prediction_text="Sorry!! you have diabetes")

if __name__=="__main__":
    app.run(debug=True)