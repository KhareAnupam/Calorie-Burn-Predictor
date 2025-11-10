import pandas as pd
import numpy as np
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

print("--- Optimized CatBoost Script with Full Metrics ---")

# --- Step 1: Load Data ---
try:
    exercise_data = pd.read_csv('exercise.csv')
    calories_data = pd.read_csv('calories.csv')
except FileNotFoundError:
    print("Error: 'exercise.csv' or 'calories.csv' not found.")
    print("Please make sure the files are in the correct directory.")
    exit()


combined_data = pd.concat([exercise_data.drop(columns=['User_ID']), calories_data['Calories']], axis=1)
combined_data.ffill(inplace=True) 

# --- Step 2: Feature Engineering ---
combined_data['BMI'] = combined_data['Weight'] / (combined_data['Height'] / 100) ** 2
combined_data['Age_Group'] = pd.cut(combined_data['Age'],
                                    bins=[0, 20, 40, 60, 80, 100],
                                    labels=['<20', '20-40', '40-60', '60-80', '80+'])
combined_data['Duration'] = np.log1p(combined_data['Duration'])
combined_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

# Define features and target
X = combined_data.drop(columns=['Calories'])
y = combined_data['Calories']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features
cat_features = ['Age_Group']
if 'Gender' in X.columns:
    cat_features.append('Gender')

# --- Step 3: Train CatBoost with tuning ---
cat_model = cb.CatBoostRegressor(
    iterations=5000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=4,
    loss_function='MAE',  
    eval_metric='MAE',   
    random_seed=42,
    early_stopping_rounds=200,
    verbose=500
)

print("\nStarting CatBoost training...")
start_time = time.time()
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=cat_features)
end_time = time.time()

print(f"CatBoost training completed in {end_time - start_time:.2f} seconds.")

# --- Step 4: Evaluate ---
final_predictions = cat_model.predict(X_test)

# Calculating all three metrics
final_mae = mean_absolute_error(y_test, final_predictions)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # RMSE is the square root of MSE
final_r2 = r2_score(y_test, final_predictions)

print("\n" + "="*50)
print(" CatBoost Model Evaluation Results")
print("="*50)
print(f" Mean Absolute Error (MAE):     {final_mae:.4f} kcal")
print(f" Root Mean Squared Error (RMSE): {final_rmse:.4f} kcal")
print(f" R-squared (R²):                {final_r2:.4f}")
print("="*50)