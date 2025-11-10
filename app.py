import gradio as gr
import numpy as np
import pandas as pd
import catboost as cb

print("--- Loading Final CatBoost Model ---")

# --- Step 1: Load the saved model ---
model_filename = "final_catboost_model.cbm"
try:
    model = cb.CatBoostRegressor()
    model.load_model(model_filename)
    print(f"Model '{model_filename}' loaded successfully!")
except Exception as e:
    print("="*50)
    print(f"ERROR: Model file '{model_filename}' not found.")
    print("Please run the 'train_and_save.py' script first.")
    print(f"Details: {e}")
    print("="*50)
    exit()


# --- Step 2: Define the Prediction Function ---
# This function MUST replicate the feature engineering from training
def predict_calories(Gender, Age, Duration, Heart_Rate, Body_Temp, Height, Weight):
    
    # 1. Create a dictionary with the raw data
    data = {
        'Gender': 0 if Gender == 'male' else 1,
        'Age': Age,
        'Duration': Duration,
        'Heart_Rate': Heart_Rate,
        'Body_Temp': Body_Temp,
        'Height': Height,
        'Weight': Weight
    }

    # 2. Apply the *exact* same feature engineering
    data['BMI'] = data['Weight'] / (data['Height'] / 100) ** 2
    
    data['Duration'] = np.log1p(data['Duration']) # Apply log-transform
    
    # Apply pd.cut to get the 'Age_Group' label
    age_group_bins = [0, 20, 40, 60, 80, 100]
    age_group_labels = ['<20', '20-40', '40-60', '60-80', '80+']
    data['Age_Group'] = pd.cut([data['Age']], 
                               bins=age_group_bins, 
                               labels=age_group_labels, 
                               right=True)[0]

    # 3. Create the final DataFrame for prediction
    # The column order must match the training 'X' DataFrame
    input_df = pd.DataFrame([data], columns=[
        'Gender', 'Age', 'Height', 'Weight', 'Duration', 
        'Heart_Rate', 'Body_Temp', 'BMI', 'Age_Group'
    ])

    # 4. Predict
    prediction = model.predict(input_df)[0]

    return f"Predicted Calories Burned: {prediction:.2f} kcal"

# --- Step 3: Create the Gradio Interface ---
interface = gr.Interface(
    fn=predict_calories,
    inputs=[
        gr.Radio(label="Gender", choices=["male", "female"]),
        gr.Number(label="Age (years)"),
        gr.Number(label="Duration (minutes)"),
        gr.Number(label="Heart Rate (bpm)"),
        gr.Number(label="Body Temperature (°C)"),
        gr.Number(label="Height (cm)"),
        gr.Number(label="Weight (kg)")
    ],
    outputs="text",
    title="Calorie Burn Predictor",
    description="Enter details to predict calories burned. Powered by the optimized CatBoost model."
)

print("\nLaunching Gradio App...")
interface.launch()