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

    data['BMI'] = data['Weight'] / (data['Height'] / 100) ** 2
    
    data['Duration'] = np.log1p(data['Duration']) 
    
    # Apply pd.cut to get the 'Age_Group' label
    age_group_bins = [0, 20, 40, 60, 80, 100]
    age_group_labels = ['<20', '20-40', '40-60', '60-80', '80+']
    data['Age_Group'] = pd.cut([data['Age']], 
                               bins=age_group_bins, 
                               labels=age_group_labels, 
                               right=True)[0]

    # 3. Create the final DataFrame for prediction
    input_df = pd.DataFrame([data], columns=[
        'Gender', 'Age', 'Height', 'Weight', 'Duration', 
        'Heart_Rate', 'Body_Temp', 'BMI', 'Age_Group'
    ])

    # 4. Predict
    prediction = model.predict(input_df)[0]
    return round(prediction, 2)

# --- Gradio Interface ---
tech_theme = gr.themes.Default(
    primary_hue="cyan",      
    secondary_hue="blue",
    neutral_hue="slate",   
    text_size="lg",        
    font=(gr.themes.GoogleFont("Orbitron"), "ui-sans-serif", "system-ui", "sans-serif"), # Futuristic font
    font_mono=(gr.themes.GoogleFont("Share Tech Mono"), "ui-monospace", "SFMono-Regular", "monospace"),
) 

# --- ANIMATED CSS gradient for the background ---
gradient_css = """
/* We must define the animation keyframes first */
@keyframes neonGradient {
    0%   { background-position: 0% 0%; }
    50%  { background-position: 100% 100%; }
    100% { background-position: 0% 0%; }
}

.gradio-container {
    /* 1. Create a large gradient with your techy colors */
    background-image: linear-gradient(
        -45deg, 
        #0f172a, /* Dark slate */
        #000000, /* Black */
        #06b6d4, /* Bright Neon Cyan */
        #0f172a, /* Dark slate again */
        #020024  /* Darkest blue */
    ) !important;
    
    /* 2. Make the background much larger than the container */
    background-size: 300% 300% !important;
    
    /* 3. Apply the animation */
    animation: neonGradient 15s ease-in-out infinite alternate !important;

    /* Fallback and sizing */
    background-color: #0f172a !important;
    background-repeat: no-repeat;
    background-attachment: fixed !C;
    min-height: 100vh;
}
"""

with gr.Blocks(theme=tech_theme, title="Calorie Predictor", css=gradient_css) as interface:
    gr.Markdown(
        """
        # 🔥 CALORIE BURN PREDICTOR
        ENTER YOUR PERSONAL AND EXERCISE DETAILS TO PREDICT THE TOTAL CALORIES BURNED.
        THIS APP IS POWERED BY A HIGH-ACCURACY CATBOOST MODEL (R² = 0.9999).
        """
    )
    
    # Create a 2-column layout for inputs
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            gr.Markdown("### 🧑 PERSONAL DETAILS")
            gender_input = gr.Radio(label="GENDER", choices=["male", "female"], value="male")
            age_input = gr.Slider(label="AGE (YEARS)", minimum=10, maximum=100, step=1, value=25)
            height_input = gr.Slider(label="HEIGHT (CM)", minimum=100, maximum=250, step=1, value=170)
            weight_input = gr.Slider(label="WEIGHT (KG)", minimum=30, maximum=200, step=0.5, value=70)
        
        with gr.Column(scale=1):
            gr.Markdown("### 🏋️ EXERCISE DETAILS")
            duration_input = gr.Slider(label="DURATION (MINUTES)", minimum=1, maximum=180, step=1, value=30)
            heart_rate_input = gr.Slider(label="HEART RATE (BPM)", minimum=50, maximum=220, step=1, value=120)
            body_temp_input = gr.Slider(label="BODY TEMP (°C)", minimum=35.0, maximum=42.0, step=0.1, value=37.5)

    # Create a row for the button
    with gr.Row():
        predict_btn = gr.Button("PREDICT CALORIES", variant="primary", scale=1)
        
    # Create a row for the output
    with gr.Row():
        calories_output = gr.Number(label="🚀 PREDICTED CALORIES (KCAL)", interactive=False, scale=2)

    # Link the button to the function
    predict_btn.click(
        fn=predict_calories,
        inputs=[
            gender_input,
            age_input,
            duration_input,
            heart_rate_input,
            body_temp_input,
            height_input,
            weight_input
        ],
        outputs=[calories_output]
    )

print("\nLaunching Gradio App with new UI...")
interface.launch()