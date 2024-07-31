from flask import Flask, request, jsonify, render_template
import joblib  # for loading the model
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('logreg_model.pkl')


@app.route('/')
def index():
    return render_template('index.html')

def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Initialize mappings with default values
    orbit_features = ['Orbit_ES-L1', 'Orbit_GEO', 'Orbit_GTO', 'Orbit_HEO', 'Orbit_ISS', 'Orbit_LEO', 'Orbit_MEO',
                      'Orbit_PO', 'Orbit_SO', 'Orbit_SSO', 'Orbit_VLEO']
    launchsite_features = ['LaunchSite_CCAFS SLC 40', 'LaunchSite_KSC LC 39A', 'LaunchSite_VAFB SLC 4E']
    landingpad_features = ['LandingPad_5e9e3032383ecb267a34e7c7', 'LandingPad_5e9e3032383ecb554034e7c9',
                           'LandingPad_5e9e3032383ecb6bb234e7ca', 'LandingPad_5e9e3032383ecb761634e7cb',
                           'LandingPad_5e9e3033383ecbb9e534e7cc']
    serial_features = ['Serial_B0003', 'Serial_B0005', 'Serial_B0007', 'Serial_B1003', 'Serial_B1004', 'Serial_B1005',
                       'Serial_B1006', 'Serial_B1007', 'Serial_B1008', 'Serial_B1010', 'Serial_B1011', 'Serial_B1012',
                       'Serial_B1013', 'Serial_B1015', 'Serial_B1016', 'Serial_B1017', 'Serial_B1018', 'Serial_B1019',
                       'Serial_B1020', 'Serial_B1021', 'Serial_B1022', 'Serial_B1023', 'Serial_B1025', 'Serial_B1026',
                       'Serial_B1028', 'Serial_B1029', 'Serial_B1030', 'Serial_B1031', 'Serial_B1032', 'Serial_B1034',
                       'Serial_B1035', 'Serial_B1036', 'Serial_B1037', 'Serial_B1038', 'Serial_B1039', 'Serial_B1040',
                       'Serial_B1041', 'Serial_B1042', 'Serial_B1043', 'Serial_B1044', 'Serial_B1045', 'Serial_B1046',
                       'Serial_B1047', 'Serial_B1048', 'Serial_B1049', 'Serial_B1050', 'Serial_B1051', 'Serial_B1054',
                       'Serial_B1056', 'Serial_B1058', 'Serial_B1059', 'Serial_B1060', 'Serial_B1062']
    gridfins_features = ['GridFins_False', 'GridFins_True']
    reused_features = ['Reused_False', 'Reused_True']
    legs_features = ['Legs_False', 'Legs_True']

    # Create feature values dictionary with default 0
    feature_values = {
        **dict.fromkeys(orbit_features, 0),
        **dict.fromkeys(launchsite_features, 0),
        **dict.fromkeys(landingpad_features, 0),
        **dict.fromkeys(serial_features, 0),
        **dict.fromkeys(gridfins_features, 0),
        **dict.fromkeys(reused_features, 0),
        **dict.fromkeys(legs_features, 0),
        'FlightNumber': 0,
        'PayloadMass': 0,
        'Flights': 0,
        'Block': 0,
        'ReusedCount': 0
    }

    # Update feature values from form data
    for key in feature_values.keys():
        if key in data:
            if key in ['FlightNumber', 'PayloadMass', 'Flights', 'Block', 'ReusedCount']:
                feature_values[key] = int(data.get(key, 0)) if key != 'PayloadMass' else float(data.get(key, 0))
            else:
                feature_values[key] = 1 if data.get(key) == 'True' else 0

    # Convert feature_values to list in the order required by the model
    feature_values_list = [
        feature_values['FlightNumber'], feature_values['PayloadMass'], feature_values['Flights'],
        feature_values['Block'], feature_values['ReusedCount'],
        feature_values['Orbit_ES-L1'], feature_values['Orbit_GEO'], feature_values['Orbit_GTO'],
        feature_values['Orbit_HEO'],
        feature_values['Orbit_ISS'], feature_values['Orbit_LEO'], feature_values['Orbit_MEO'],
        feature_values['Orbit_PO'],
        feature_values['Orbit_SO'], feature_values['Orbit_SSO'], feature_values['Orbit_VLEO'],
        feature_values['LaunchSite_CCAFS SLC 40'], feature_values['LaunchSite_KSC LC 39A'],
        feature_values['LaunchSite_VAFB SLC 4E'],
        feature_values['LandingPad_5e9e3032383ecb267a34e7c7'], feature_values['LandingPad_5e9e3032383ecb554034e7c9'],
        feature_values['LandingPad_5e9e3032383ecb6bb234e7ca'], feature_values['LandingPad_5e9e3032383ecb761634e7cb'],
        feature_values['LandingPad_5e9e3033383ecbb9e534e7cc'],
        feature_values['Serial_B0003'], feature_values['Serial_B0005'], feature_values['Serial_B0007'],
        feature_values['Serial_B1003'],
        feature_values['Serial_B1004'], feature_values['Serial_B1005'], feature_values['Serial_B1006'],
        feature_values['Serial_B1007'],
        feature_values['Serial_B1008'], feature_values['Serial_B1010'], feature_values['Serial_B1011'],
        feature_values['Serial_B1012'],
        feature_values['Serial_B1013'], feature_values['Serial_B1015'], feature_values['Serial_B1016'],
        feature_values['Serial_B1017'],
        feature_values['Serial_B1018'], feature_values['Serial_B1019'], feature_values['Serial_B1020'],
        feature_values['Serial_B1021'],
        feature_values['Serial_B1022'], feature_values['Serial_B1023'], feature_values['Serial_B1025'],
        feature_values['Serial_B1026'],
        feature_values['Serial_B1028'], feature_values['Serial_B1029'], feature_values['Serial_B1030'],
        feature_values['Serial_B1031'],
        feature_values['Serial_B1032'], feature_values['Serial_B1034'], feature_values['Serial_B1035'],
        feature_values['Serial_B1036'],
        feature_values['Serial_B1037'], feature_values['Serial_B1038'], feature_values['Serial_B1039'],
        feature_values['Serial_B1040'],
        feature_values['Serial_B1041'], feature_values['Serial_B1042'], feature_values['Serial_B1043'],
        feature_values['Serial_B1044'],
        feature_values['Serial_B1045'], feature_values['Serial_B1046'], feature_values['Serial_B1047'],
        feature_values['Serial_B1048'],
        feature_values['Serial_B1049'], feature_values['Serial_B1050'], feature_values['Serial_B1051'],
        feature_values['Serial_B1054'],
        feature_values['Serial_B1056'], feature_values['Serial_B1058'], feature_values['Serial_B1059'],
        feature_values['Serial_B1060'],
        feature_values['Serial_B1062'],
        feature_values['GridFins_False'], feature_values['GridFins_True'],
        feature_values['Reused_False'], feature_values['Reused_True'],
        feature_values['Legs_False'], feature_values['Legs_True']
    ]

    feature_values_list = np.array(feature_values_list).reshape(1, -1)

    # Check feature values length
    if feature_values_list.shape[1] != 83:
        return jsonify({
                           'error': f'Feature count mismatch. Expected 83 features, but got {feature_values_list.shape[1]} features.'})

    # Perform prediction

    prediction = model.predict(feature_values_list)[0]
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)

