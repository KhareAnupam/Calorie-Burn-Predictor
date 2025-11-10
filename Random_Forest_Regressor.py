import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # All 3 METRICS
import time

print("--- Starting SK-Learn Random Forest Script ---")

# --- Step 1: Load and Prepare Data
print("Loading and preparing data...")

try:
    exercise_data = pd.read_csv('exercise.csv')
    calories_data = pd.read_csv('calories.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure 'exercise.csv' and 'calories.csv' are in the folder.")
    exit()

combined_data = pd.concat([exercise_data.drop(columns=['User_ID']), calories_data['Calories']], axis=1)
combined_data.ffill(inplace=True)

# --- Step 2: Advanced Feature Engineering ---
print("Executing advanced feature engineering...")

combined_data['BMI'] = combined_data['Weight'] / (combined_data['Height'] / 100) ** 2
combined_data['Age_Group'] = pd.cut(combined_data['Age'], bins=[0, 20, 40, 60, 80, 100], labels=['<20', '20-40', '40-60', '60-80', '80+'])
combined_data['Duration'] = np.log1p(combined_data['Duration'])

combined_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)
combined_data = pd.get_dummies(combined_data, columns=['Age_Group'], drop_first=True)

X_original = combined_data.drop(columns=['Calories'])
y = combined_data['Calories']

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_original)

print(f"Original feature count: {X_original.shape[1]}")
print(f"Features after PolynomialFeatures: {X_poly.shape[1]}")

# --- Step 3: Data Splitting and Scaling ---
print("Splitting and scaling data...")
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 4: Train the Random Forest Model ---
print("\n--- Training the Random Forest Regressor ---")
# n_estimators=500 to be comparable to the other models
# n_jobs=-1 tells the model to use all available CPU cores for training
rf_model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1, # This will speed up training significantly
    verbose=0  # Set to 1 for training progress
)

start_time = time.time()
rf_model.fit(X_train_scaled, y_train)
end_time = time.time()

print(f"Random Forest training completed in {end_time - start_time:.2f} seconds.")

# --- Step 5: Evaluate the Model ---
print("\n--- Evaluating the Random Forest Model ---")

final_predictions = rf_model.predict(X_test_scaled)

# Calculate all three metrics
final_mae = mean_absolute_error(y_test, final_predictions)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_r2 = r2_score(y_test, final_predictions)

print("\n" + "="*50)
print(" Random Forest (SK-Learn) Evaluation Results")
print("="*50)
print(f" Mean Absolute Error (MAE):     {final_mae:.4f} kcal")
print(f" Root Mean Squared Error (RMSE): {final_rmse:.4f} kcal")
print(f" R-squared (R²):                {final_r2:.4f}")
print("="*50)