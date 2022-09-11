import numpy as np
import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

model1 = pickle.load(open('hpf_model.pkl','rb'))
model2 = pickle.load(open('bmi.pkl','rb'))
model3 = pickle.load(open('bcp.pkl','rb'))
model4 = pickle.load(open('rcq.pkl','rb'))


app = Flask(__name__)

@app.route("/")
def base():
    return render_template("Base.html")

@app.route('/hfp')
def hfp():
    return render_template('hfp.html')

@app.route('/result1', methods = ['POST'])
def get_data_from_html1():
    age = request.form['age']
    anaemia = request.form['anaemia']
    creatinine_phosphokinase = request.form['creatinine_phosphokinase']
    diabetes = request.form['diabetes']
    ejection_fraction = request.form['ejection_fraction']
    high_blood_pressure = request.form['high_blood_pressure']
    platelets = request.form['platelets']
    serum_creatinine = request.form['serum_creatinine']
    serum_sodium = request.form['serum_sodium']
    sex = request.form['sex']
    smoking = request.form['smoking']
    time = request.form['time']
    arr = np.array([[age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,
    high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]])
    sc_arr = StandardScaler()
    arr = sc_arr.fit_transform(arr)
    pred = model1.predict(arr)
    if pred == 0:
        pred = 'Death Will Not be Possible'
    else:
        pred = 'Death will be possible'
    return render_template('result1.html', data = pred)


@app.route('/bmi')
def bmi():
    return render_template('bmi.html')

@app.route('/result2', methods = ['POST'])
def get_data_from_html2():
    gender = request.form['gender']
    height = request.form['height']
    weight = request.form['weight']
    arr = np.array([[gender,height,weight]],dtype=float)
    """sc_arr = StandardScaler()
    arr = sc_arr.fit_transform(arr)"""
    pred = model2.predict(arr)
    if pred == 0:
        pred = 'Extremely Weak'
    elif pred == 1:
        pred = 'Weak'
    elif pred == 2:
        pred = 'Normal'
    elif pred == 3:
        pred = 'Overweight'
    elif pred == 4:
        pred = 'Obesity'
    else:
        pred = 'Extreme Obesity'
    return render_template('result1.html', data = pred)

@app.route('/bcp')
def bcp():
    return render_template('bcp.html')

@app.route("/result3", methods = ['POST'])
def get_data_from_html3():
    radius_mean = request.form['radius_mean']
    perimeter_mean = request.form['perimeter_mean']
    area_mean = request.form['area_mean']
    compactness_mean = request.form['compactness_mean']
    concavity_mean = request.form['concavity_mean']
    concave_points_mean = request.form['concave_points_mean']
    radius_se = request.form['radius_se']
    perimeter_se = request.form['perimeter_se']
    area_se = request.form['area_se']
    radius_worst = request.form['radius_worst']
    perimeter_worst = request.form['perimeter_worst']
    area_worst = request.form['area_worst']
    compactness_worst = request.form['compactness_worst']
    concavity_worst = request.form['concavity_worst']
    concave_points_worst = request.form['concave_points_worst']
    arr = np.array([[radius_mean,perimeter_mean,area_mean,compactness_mean,concavity_mean,
    concave_points_mean,radius_se,perimeter_se,area_se,radius_worst,perimeter_worst,
    area_worst,compactness_worst,concavity_worst,concave_points_worst]],dtype=float)
    pred = model3.predict(arr)
    if pred == 0:
        pred = 'Breast Cancer is not there'
    else:
        pred = 'Breast Cancer is there'
    return render_template('result1.html', data = pred)

@app.route('/rwq')
def rwq():
    return render_template('rwq.html')

@app.route('/result4', methods = ['POST'])
def get_data_from_html4():
    fixed_acidity = request.form['fixed_acidity']
    volatile_acidity = request.form['volatile_acidity']
    citric_acid = request.form['citric_acid']
    residual_sugar = request.form['residual_sugar']
    chlorides = request.form['chlorides']
    free_sulfur_dioxide = request.form['free_sulfur_dioxide']
    total_sulfur_dioxide = request.form['total_sulfur_dioxide']
    density = request.form['density']
    pH = request.form['pH']
    sulphates = request.form['sulphates']
    alcohol = request.form['alcohol']
    arr = np.array([[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,
    chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]])
    sc_arr = StandardScaler()
    arr = sc_arr.fit_transform(arr)
    pred = model4.predict(arr)
    if pred == 0:
        pred = 'The Quality of Wine is Bad.'
    else:
        pred = 'The Quality of Wine is Good.'
    return render_template('result1.html', data = pred)

if __name__ == "__main__":
    app.run(debug=True)