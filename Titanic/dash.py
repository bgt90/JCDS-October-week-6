from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd

app = Flask(__name__)

#halaman home
@app.route('/')
def home():
    return render_template('home.html')

# #halaman input prediksi
@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    return render_template('predict.html')

# #halaman hasil prediksi
@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        input = request.form

        df_predict = pd.DataFrame({
            'sex':[input['sex']],
            'age':[input['age']],
            'parch':[input['parch']],
            'fare':[input['fare']],
            'class':[input['class']],
            'embark_town':[input['et']],
            'alone':[input['alone']]
        })

        # df_predict = pd.DataFrame({
        #     'alcohol':[10],
        #     'density':[1],
        #     'fixed acidity level':['low'],
        #     'chlorides level':['low']
        # })
        # prediksi = 0.5

        prediksi = model.predict_proba(df_predict)[0][1]

        if prediksi > 0.5:
            status = "Alive"
        else:
            status = "Not alive"

        return render_template('result.html',
            data=input, pred=status)

if __name__ == '__main__':
    # model = joblib.load('model_joblib')

    filename = 'titanic2.sav'
    model = pickle.load(open(filename,'rb'))

    app.run(debug=True)