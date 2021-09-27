from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)

model = pickle.load(open('DTmodel.pkl','rb'))
dress= pd.read_csv("cleandata.csv")
m1=[36.304348, 160.388116, 66.376812]
s1=[9.954757, 7.068046, 5.102195]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    age = float(request.form.get("age"))
    height = float(request.form.get("height"))
    weight = int(request.form.get("weight"))
    ar = [age, height, weight]
    #fitting the input data according to dataset framed on the basis of z score
    Tzs = [0, 0, 0]
    for i in range(3):
        Tzs[i] = ((ar[i] - m1[i]) / s1[i])
    a1 = Tzs[0]
    h1 = Tzs[1]
    w1 = Tzs[2]
    b = h1 / w1
    ws = w1 * w1
    tin = (a1, h1, w1, b, ws)
    tarr = np.asarray(tin)
    reshaped_in = tarr.reshape(1, 5)
    prediction = model.predict(reshaped_in)
    sizen = ['XXS', 'XS', 'S', 'L', 'XL', 'XXL', 'XXXL']
    return sizen[prediction[0]-1]

if __name__=="__main__":
    app.run(debug=True)
