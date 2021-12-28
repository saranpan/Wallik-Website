# Big Thanks to Krish Naik

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle



app = Flask(__name__)
iris_model = pickle.load(open('model/iris_dt.pkl', 'rb'))

"""
value = [2, 3, 1, 1]
int_features = [x for x in value]

print(int_features)

final_features = [np.array(int_features)];print(final_features)
prediction = iris_model.predict(final_features);print(prediction)

output = prediction[0]
text = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']   
""" 


#default root
@app.route('/')

# render index.html
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/1_iris')
def iris():
    return render_template('1_iris.html')


# /predict to the result
@app.route('/1_iris',methods=['POST'])
def predict():
    '''For rendering results on HTML GUI'''
    # Receive the input from the POST request.

    # generator to list
    inputs = [x for x in request.form.values()]

    # Transforming the input list to numpy array for ML model
    features = [np.array(inputs)]

    output = iris_model.predict(features)[0]

    feature_name = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']   
    prediction_text = f'This is {output}'

    return render_template('1_iris.html', character = zip(feature_name,inputs), prediction_text=prediction_text)





@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls throught request
    '''
    data = request.get_json(force=True)
    prediction = iris_model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output,data)














# mother app
if __name__ == "__main__":
    app.run(debug=True)