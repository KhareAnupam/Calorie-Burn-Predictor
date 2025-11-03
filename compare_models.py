import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import catboost as cb

# --- Step 1: Load and Prepare Data (Same as your other script) ---

# Load datasets
try:
    exercise_data = pd.read_csv('exercise.csv')
    calories_data = pd.read_csv('calories.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure 'exercise.csv' and 'calories.csv' are in the same folder as the script.")
    exit()

# Combine datasets
data = pd.concat([exercise_data, calories_data['Calories']], axis=1)

# Drop the User_ID column as it's not a feature
data.drop(columns=['User_ID'], inplace=True)

# Encoding categorical variables
data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

# Prepare features (X) and target (y)
X = data.drop(columns=['Calories'])
y = data['Calories']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply StandardScaler (very important for fair comparison)
# Note: Tree-based models are less sensitive to scaling than neural networks,
# but it's good practice to keep the preprocessing pipeline consistent.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- Step 2: Train and Evaluate XGBoost Model ---

print("Training XGBoost Model...")

# Initialize the XGBoost Regressor
# n_estimators is the number of trees in the forest.
# learning_rate controls how much each tree corrects the previous one.
xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the test data
xgb_predictions = xgb_model.predict(X_test)

# Calculate and print the Mean Absolute Error (MAE)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
print(f"XGBoost Model - Final Mean Absolute Error (MAE): {xgb_mae:.4f} kcal")
print("-" * 40)


# --- Step 3: Train and Evaluate CatBoost Model ---

print("Training CatBoost Model...")

# Initialize the CatBoost Regressor
# iterations is similar to n_estimators.
# verbose=0 keeps the output clean by not printing training progress.
cat_model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.05, random_state=42, verbose=0)

# Train the model
cat_model.fit(X_train, y_train)

# Make predictions on the test data
cat_predictions = cat_model.predict(X_test)

# Calculate and print the Mean Absolute Error (MAE)
cat_mae = mean_absolute_error(y_test, cat_predictions)
print(f"CatBoost Model - Final Mean Absolute Error (MAE): {cat_mae:.4f} kcal")
print("-" * 40)


# --- Step 4: Final Comparison ---
print("\n--- Model Comparison Summary ---")
print(f"XGBoost MAE:  {xgb_mae:.4f} kcal")
print(f"CatBoost MAE: {cat_mae:.4f} kcal")

best_model = "CatBoost" if cat_mae < xgb_mae else "XGBoost"
print(f"\n🏆 The best performing tree-based model is: {best_model}")






#  python -m venv venv
#  .\venv\Scripts\activate

#   installing necessary models
#  pip install torch pandas scikit-learn numpy