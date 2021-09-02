import numpy as np

from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("student_mark_predictor.pkl")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    
    input_features = [int(x) for x in request.form.values()]
    # print(input_features)
    features_value = np.array(input_features)
    
    #validate input hours
    if input_features[0] <0 or input_features[0] >24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 if you live on the Earth')
        

    output = model.predict([features_value])[0][0].round(2)



    return render_template('index.html', prediction_text='You will get [{}%] marks, when you do study [{}] hours per day '.format(output, int(features_value[0])))


if __name__ == "__main__":
    app.run(debug=True)
    