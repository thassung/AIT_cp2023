from flask import Flask, jsonify, request, render_template
import pandas as pd
import joblib
import pickle

app = Flask(__name__, template_folder = 'templates')

model = joblib.load('RandomForest.pkl')

def atlas(country):
    map = pd.read_csv('./data/atlas.csv')
    map.set_index('Country',inplace=True)
    continent = map.loc[country]['Continent']
    deathrate = map.loc[country]['Death Rate']
    return continent, deathrate

@app.route("/", methods = ['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    age = data.get('age')
    country = data.get('country')
    exercise = data.get('exercise')
    sleep = data.get('sleep')
    bp = data.get('bp')
    triglycerides = data.get('triglycerides')
    cholesterol = data.get('cholesterol')
    diabetes = data.get('diabetes')
    heartprob = data.get('heartprob')
    obesity = data.get('obesity')
    alcohol = data.get('alcohol')
    country = 'Thailand' if country =='others' else country
    continent, deathrate = atlas(country)

    input = pd.DataFrame(data={
        'Diabetes' : [diabetes],
        'Cholesterol' : [cholesterol],
        'Systolic BP' : [bp],
        'Continent' : [continent],
        'Sleep Hours Per Day' : [sleep],
        'Exercise Hours Per Week' : [exercise],
        'Triglycerides' : [triglycerides],
        'Previous Heart Problems' : [heartprob],
        'Obesity' : [obesity],
        'Age' : [age],
        'Death Rate' : [deathrate],
        'Alcohol Consumption' : [alcohol]
    })
    
    input = input.replace('', None)

    imputer = pickle.load(open('./preprocessor/imputer.pkl', 'rb'))
    transformer = pickle.load(open('./preprocessor/transformer.pkl', 'rb'))

    fmed = ['Cholesterol', 'Systolic BP', 'Sleep Hours Per Day', 'Exercise Hours Per Week', 
            'Triglycerides', 'Age', 'Death Rate']
    fmode = ['Diabetes', 'Continent', 'Previous Heart Problems', 'Obesity', 
             'Alcohol Consumption']

    input = imputer.transform(input)
    input = pd.DataFrame(input, columns=fmed+fmode)
    input = transformer.transform(input)

    prediction = model.predict(input)[0]
    message = ''
    if prediction == 0:
        message = 'You are not at risk of Heart Attack'
        
    if prediction == 1:
        message = 'You are at risk of Heart Attack'
    ###result
    result = {
        'message': message
    }
    print(message)
    return jsonify(result), 200
        
if __name__ == "__main__":
    app.run(host='127.0.0.1',port='8080',debug=False)