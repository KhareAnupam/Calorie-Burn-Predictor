import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import time

print("--- Starting PyTorch Script ---")

# --- Step 1: Load Data and Add BMI Feature ---
print("Loading data and adding BMI as a feature...")

try:
    exercise_data = pd.read_csv('exercise.csv')
    calories_data = pd.read_csv('calories.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure 'exercise.csv' and 'calories.csv' are in the folder.")
    exit()

data = pd.concat([exercise_data.drop(columns=['User_ID']), calories_data['Calories']], axis=1)
data.ffill(inplace=True)

# THE KEY CHANGE: Engineer the BMI feature
data['BMI'] = data['Weight'] / (data['Height'] / 100) ** 2

# Encoding the 'Gender' variable
data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

# Prepare features (X) and target (y)
# Using BMI and drop Height/Weight to avoid redundant information
X = data[['Gender', 'Age', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']]
y = data['Calories']

# --- Step 2: Data Splitting and Scaling ---
print("Splitting and scaling data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 3: Convert Data to PyTorch Tensors ---
X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# --- Step 4: Define a Slightly More Capable PyTorch Model ---
class IntermediateNet(nn.Module):
    def __init__(self, input_dim):
        super(IntermediateNet, self).__init__()
        # A slightly wider network to learn better
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Step 5: Train the Model for Longer ---
print("Training the PyTorch model...")

input_dim = X_train_torch.shape[1]
model = IntermediateNet(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Increased epochs to give the model more time to learn from the new feature
n_epochs = 500

start_time = time.time()
for epoch in range(n_epochs):
    model.train()
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# --- Step 6: Evaluate the Model ---
print("\n--- Evaluating the PyTorch Model ---")

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_torch).squeeze()
    mae = mean_absolute_error(y_test_torch.numpy(), test_predictions.numpy())

print("\n" + "="*50)
print(f"PyTorch - Final Mean Absolute Error (MAE): {mae:.4f} kcal")
print("="*50)