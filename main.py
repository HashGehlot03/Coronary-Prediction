from flask import Flask, render_template,request,flash
import pandas as pd
import numpy as np
import joblib
model = joblib.load('model.joblib')
app = Flask(__name__)
df = pd.read_csv('coronary_prediction.csv')
cols = [col for col in df.columns if col != 'TenYearCHD']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]

    test_data = pd.DataFrame([int_features],columns = cols)
    final_features = [np.array(int_features)]
    prediction = model.predict(test_data)
    output = prediction[0]
    if output == 0:
        return render_template('predict.html', prediction_text = "Results are Negative , you don't have coronary disease")
    else:
        return render_template('predict.html', prediction_text = 'Results are Positive , you have coronary disease')


if __name__ == '__main__':
    app.run(debug = True)
