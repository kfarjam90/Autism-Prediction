from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and encoders
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Feature list based on your notebook
features = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 
            'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
            'age', 'gender', 'ethnicity', 'jaundice', 'austim', 
            'contry_of_res', 'used_app_before', 'result', 'relation']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        form_data = request.form.to_dict()
        
        # Create a DataFrame with the same structure as training data
        input_data = {feature: [form_data.get(feature, '')] for feature in features}
        df = pd.DataFrame(input_data)
        
        # Convert numerical fields
        numerical_features = ['age', 'result']
        for feature in numerical_features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
        
        # Encode categorical features
        for column in df.select_dtypes(include=['object']).columns:
            if column in encoders:
                df[column] = encoders[column].transform(df[column])
        
        # Make prediction
        prediction = best_model.predict(df)
        probability = best_model.predict_proba(df)[0][1]
        
        result = "High probability of autism" if prediction[0] == 1 else "Low probability of autism"
        
        return render_template('result.html', 
                             prediction=result, 
                             probability=f"{probability*100:.2f}%",
                             input_data=form_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)