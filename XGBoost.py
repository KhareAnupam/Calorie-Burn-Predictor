import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error
import time

print("--- Starting Powerful XGBoost Script ---")

# --- Step 1: Load and Prepare Data 
print("Loading and preparing data...")

try:
    exercise_data = pd.read_csv('exercise.csv')
    calories_data = pd.read_csv('calories.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure 'exercise.csv' and 'calories.csv' are in the folder.")
    exit()

# Combine datasets and handle potential missing values
combined_data = pd.concat([exercise_data.drop(columns=['User_ID']), calories_data['Calories']], axis=1)
combined_data.ffill(inplace=True) # Using ffill as in the original script

# --- Step 2: Advanced Feature Engineering (The "Secret Weapon") ---
print("Executing advanced feature engineering...")

# Create BMI, Age_Group, and log-transform Duration
combined_data['BMI'] = combined_data['Weight'] / (combined_data['Height'] / 100) ** 2
combined_data['Age_Group'] = pd.cut(combined_data['Age'], bins=[0, 20, 40, 60, 80, 100], labels=['<20', '20-40', '40-60', '60-80', '80+'])
combined_data['Duration'] = np.log1p(combined_data['Duration'])

# Encoding categorical variables
combined_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)
combined_data = pd.get_dummies(combined_data, columns=['Age_Group'], drop_first=True)

# Prepare data before creating polynomial features to get the original feature count
X_original = combined_data.drop(columns=['Calories'])
y = combined_data['Calories']

# The KEY STEP: Create Polynomial and Interaction Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_original)

print(f"Original feature count: {X_original.shape[1]}")
print(f"Features after PolynomialFeatures: {X_poly.shape[1]}")

# --- Step 3: Data Splitting and Scaling ---
print("Splitting and scaling data...")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Apply MinMaxScaler 
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 4: Hyperparameter Tuning with GridSearchCV ---
print("\n--- Searching for the best XGBoost parameters ---")
# This step systematically finds the best model settings.

# Define the parameter grid to search
param_grid = {
    'n_estimators': [500, 1000],          # Number of trees
    'learning_rate': [0.05, 0.1],         # Step size shrinkage
    'max_depth': [3, 5, 7],               # Maximum depth of a tree
    'colsample_bytree': [0.7, 1.0],       # Subsample ratio of columns when constructing each tree
}

# Initialize the XGBoost Regressor  
xgb_reg = xgb.XGBRegressor(random_state=42)

# Initialize GridSearchCV    3-fold cross-validation  (Hyperparameter Tuning)
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid,
                           cv=3, n_jobs=-1, scoring='neg_mean_absolute_error', verbose=2)

start_time = time.time()
grid_search.fit(X_train_scaled, y_train)
end_time = time.time()

print(f"Grid search completed in {end_time - start_time:.2f} seconds.")
print("Best parameters found: ", grid_search.best_params_)

# --- Step 5: Train Final Model and Evaluate ---
print("\n--- Training final XGBoost model with best parameters ---")

# Get the best estimator from the grid search
best_xgb_model = grid_search.best_estimator_

# Make predictions on the test data
final_predictions = best_xgb_model.predict(X_test_scaled)

# Calculate and print the final Mean Absolute Error (MAE)
final_mae = mean_absolute_error(y_test, final_predictions)

print("\n" + "="*50)
print(f" Powerful XGBoost - Final Mean Absolute Error (MAE): {final_mae:.4f} kcal")
print("="*50)