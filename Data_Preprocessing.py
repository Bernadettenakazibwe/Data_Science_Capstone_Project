import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection  import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



df = pd.read_csv('Feature_Engineered_Global_Temp.csv')

df.rename(columns={'Temp_Anomaly_x': 'Monthly_Temp_anomaly', 'Temp_Anomaly_y': 'Season_Temp_anomaly'}, inplace=True)
df.dropna(inplace=True)


#print(df.head())

# Encode Categorical variables(OneHotEncoder for Season and Month)

df =pd.get_dummies(df, columns=['Season'], drop_first =True) #One hot encoding Season

# Convert Month names to numbers
month_mapping = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}
df["Month"] = df["Month"].map(month_mapping)

#3. Feature Selection
# Get all season columns dynamically since "Season" is replaced with one-hot encoded columns
season_columns = [col for col in df.columns if col.startswith("Season_")]
selected_features = ["Year", "Month","Monthly_Temp_anomaly","5-Year Moving Avg"] + season_columns
target_variable = "J-D"  # Our target is annual temperature anomaly

# Train_Test split - 80% train, 20% test

x = df[selected_features]
y = df[target_variable]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state =42, shuffle= False)

#Feature Scaling
scaler =StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Convert back to DataFrame
X_train = pd.DataFrame(x_train_scaled, columns=x_train.columns)
X_test = pd.DataFrame(x_test_scaled, columns=x_test.columns)

#print(df.tail())
#print(x_train.head())

#Model Training

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(x_train, y_train)  # Train the model
    y_pred = model.predict(x_test)  # Predict on test data
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"MAE": mae, "MSE": mse, "R²": r2}
    
    print(f"📌 {name} Performance:")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - MSE: {mse:.4f}")
    print(f"   - R² Score: {r2:.4f}\n")

# Find the best model (lowest MAE, highest R²)
best_model_name = max(results, key=lambda x: results[x]["R²"])
best_model = models[best_model_name]
print(f"✅ Best Model: {best_model_name}")

# Get predictions from the best model
y_pred_best = best_model.predict(x_test)