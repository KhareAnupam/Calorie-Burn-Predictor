import pandas as pd
import numpy as np
import catboost as cb
from sklearn.model_selection import train_test_split
import time

print("--- Training and Saving Final CatBoost Model ---")

# --- Step 1: Load Data ---
print("Loading data...")
try:
    exercise_data = pd.read_csv('exercise.csv')
    calories_data = pd.read_csv('calories.csv')
except FileNotFoundError:
    print("Error: 'exercise.csv' or 'calories.csv' not found.")
    exit()

combined_data = pd.concat([exercise_data.drop(columns=['User_ID']), calories_data['Calories']], axis=1)
combined_data.ffill(inplace=True) 

# --- Step 2: Feature Engineering  ---
print("Applying feature engineering...")
combined_data['BMI'] = combined_data['Weight'] / (combined_data['Height'] / 100) ** 2
combined_data['Age_Group'] = pd.cut(combined_data['Age'],
                                    bins=[0, 20, 40, 60, 80, 100],
                                    labels=['<20', '20-40', '40-60', '60-80', '80+'])
combined_data['Duration'] = np.log1p(combined_data['Duration'])
combined_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

# Define features and target
X = combined_data.drop(columns=['Calories'])
y = combined_data['Calories']

# --- Training the Final Model ---
print("Training final model on ALL data...")

# Identify categorical features
cat_features = ['Age_Group']
if 'Gender' in X.columns:
    cat_features.append('Gender')

# --- Step 3: Train Final CatBoost Model ---
final_model = cb.CatBoostRegressor(
    iterations=5000, # Use the full iterations
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=4,
    loss_function='MAE',
    eval_metric='MAE',
    random_seed=42,
    verbose=500
)

start_time = time.time()
# Fit the model on the ENTIRE dataset
final_model.fit(X, y, cat_features=cat_features)
end_time = time.time()

print(f"Final model training completed in {end_time - start_time:.2f} seconds.")

# --- Step 4: Save the Model ---
model_filename = "final_catboost_model.cbm"
final_model.save_model(model_filename)

print("\n" + "="*50)
print(f"Final model saved as: {model_filename}")
print("Ready for Gradio app!")
print("="*50)