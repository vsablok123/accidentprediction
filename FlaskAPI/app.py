
# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import datetime

# Load the Random Forest CLassifier model
filename = 'C:/Users/sablo/original/CarAccidentPrediction/MyProject/FlaskAPI/accident-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
accident_dataset = pd.read_csv("C:/Users/sablo/original/CarAccidentPrediction/MyProject/model_Data.csv")
dummy_input_data = [27.872295, -82.745537, 815, 3, 7, 2, 8, False, True, False, False,
       False, 62.1, 4.6, 30.09, 10.0, 93.0, 0, 0, 0, 0, 0, 0, 0]

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print(request.values)
    print(request.form)
    print(request.headers)
    if request.method == 'POST':
        
        req_data = request.form
        #print(req_data)
        modelinput = create_model_input(req_data)
        
        #data = np.array(dummy_input_data)
        my_prediction = classifier.predict(modelinput.values.reshape(1,-1))[0]
        #print(req_data)
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)


def create_model_input(request):
    inputdata = pd.Series(np.zeros(24), index = ['Start_Lat', 'Start_Lng', 'Cluster', 'Weekday', 'Hour', 'Date', 'Month',
       'Sunrise_Sunset', 'Traffic_Signal', 'Station', 'Junction', 'Crossing',
       'Temperature(F)', 'Wind_Speed(mph)', 'Pressure(in)', 'Visibility(mi)',
       'Humidity(%)', 'cloudy', 'fog', 'overcast', 'rain', 'snow',
       'thunderstorm', 'wind'])
    inputdata['Cluster'] = find_cluster(accident_dataset,float(request['latitude']),float(request['longitude']))
    inputdata['Start_Lat'] = float(request['latitude'])
    inputdata['Start_Lng'] = float(request['longitude'])
    inputdata['Temperature(F)'] = float(request['temp'])
    inputdata['Pressure(in)'] = float(request['pres'])
    inputdata['Wind_Speed(mph)'] = float(request['windspeed'])
    inputdata['Humidity(%)'] = float(request['humi'])
    inputdata['Visibility(mi)'] = float(request['visibility'])
    inputdata[request['roadfeature']] = 1
    if (request['weatherkeyword'] != 0):
        inputdata[request['weatherkeyword']] = 1
    
    datetime_object = datetime.datetime.strptime(request['date'], '%Y-%m-%dT%H:%M')
    inputdata['Hour'] = datetime_object.hour
    inputdata['Date'] = datetime_object.day
    inputdata['Month'] = datetime_object.month
    inputdata['Weekday'] = datetime_object.date().weekday()
    inputdata['Sunrise_Sunset'] = 1 if datetime_object.hour < 18 and datetime_object.hour > 6 else 0
    
    return inputdata
    
def find_cluster(accident_dataset, lat, long):
    # load all cluster accident waypoints to check against proximity
    cluster = 1
    # approximate radius of earth in km
    R = 6373.0
    for index, row in accident_dataset.iterrows():
        lat = np.radians(lat)
        lng = np.radians(long)
        latref = row['Start_Lat']
        lngref = row['Start_Lng']
        dlat = lat - latref
        dlng = lng - lngref
        a = np.sin(dlat / 2) ** 2 + np.cos(latref) * np.cos(lat) * np.sin(dlng/ 2) ** 2
        dist = R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
        
        if (dist < 0.050):
            cluster = row['Cluster']
            break


    return cluster



