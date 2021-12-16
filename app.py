# Big Thanks to Krish Naik

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default root
@app.route('/')

# render index.html
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/iris')
def iris():
    return render_template('iris.html')

# /predict to the result
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    text = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']   

    return render_template('iris.html', character = zip(text,int_features), 
    prediction_text=f'This is {output}')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls throught request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output,data)



# mother app
if __name__ == "__main__":
    app.run(debug=True)