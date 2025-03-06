import pandas as pd
import numpy as np
import joblib  # For saving the model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Feature_Engineered_Global_Temp.csv")
print(df.head())
print(df.info())

# Encode Categorical variables(OneHotEncoder for Season and Month)

df =pd.get_dummies(df, columns=['Season'], drop_first =True) #One hot encoding Season


# Convert Month names to numerical values
month_mapping = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                 "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
df["Month"] = df["Month"].map(month_mapping)

# Create lag features
df["Lag_1"] = df["J-D"].shift(1)
df["Lag_2"] = df["J-D"].shift(2)
df["Lag_3"] = df["J-D"].shift(3)

# Create rolling statistics
df["Rolling_3"] = df["J-D"].rolling(window=3).mean()
df["Rolling_6"] = df["J-D"].rolling(window=6).mean()

#Renaming monthly and season anomalies
df.rename(columns={'Temp_Anomaly_x': 'Monthly_Temp_anomaly', 'Temp_Anomaly_y': 'Seasonal_Temp_anomaly'}, inplace=True)
# Drop NaN values (from lag/rolling features)
df.dropna(inplace=True)
print(df.info())
# Encode Month as cyclic features
df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

#Save new preprocessed data
df.to_csv("Preprocessed.csv", index=False)

season_columns = [col for col in df.columns if col.startswith("Season_")]

target_variable = "Monthly_Temp_anomaly"
selected_features = ["Year", "J-D","Month_sin", "Month_cos", "5-Year Moving Avg", "Lag_1", "Lag_2", "Lag_3", "Rolling_3", "Rolling_6","Seasonal_Temp_anomaly"] + season_columns

# Train-test split
X = df[selected_features]
y = df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print(X_train.columns)


# Standard Scaling for Linear Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

# Train Linear Regression Model (with polynomial features)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

joblib.dump(poly, "poly_transform.pkl")

lr = LinearRegression()
lr.fit(X_train_poly, y_train)
y_pred_lr = lr.predict(X_test_poly)

# Train Random Forest Model (with Grid Search for tuning)
rf = RandomForestRegressor()
param_grid_rf = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='r2')
grid_rf.fit(X_train, y_train)
y_pred_rf = grid_rf.best_estimator_.predict(X_test)

# Train XGBoost Model (with Grid Search for tuning)
xgb = XGBRegressor(objective='reg:squarederror')
param_grid_xgb = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 6, 10]}
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='r2')
grid_xgb.fit(X_train, y_train)
y_pred_xgb = grid_xgb.best_estimator_.predict(X_test)

# Train Decision Tree Model (with Grid Search for tuning)
dt = DecisionTreeRegressor()
param_grid_dt = {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}
grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='r2')
grid_dt.fit(X_train, y_train)
y_pred_dt = grid_dt.best_estimator_.predict(X_test)


# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {model_name} Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    return {"model": model_name, "mae": mae, "mse": mse, "r2": r2}


# Evaluate models
results = []
results.append(evaluate_model(y_test, y_pred_lr, "Linear Regression"))
results.append(evaluate_model(y_test, y_pred_rf, "Random Forest"))
results.append(evaluate_model(y_test, y_pred_xgb, "XGBoost"))
results.append(evaluate_model(y_test, y_pred_dt, "Decision Tree"))

# Find the best model (highest R² Score)
best_model = max(results, key=lambda x: x["r2"])
print(f"\n🏆 Best Model: {best_model['model']} with R² Score: {best_model['r2']:.4f}")

# Save the best model
if best_model["model"] == "Linear Regression":
    joblib.dump(lr, "best_model.pkl")
elif best_model["model"] == "Random Forest":
    joblib.dump(grid_rf.best_estimator_, "best_model.pkl")
elif best_model["model"] == "XGBoost":
    joblib.dump(grid_xgb.best_estimator_, "best_model.pkl")
elif best_model["model"] == "Decision Tree":
    joblib.dump(grid_dt.best_estimator_, "best_model.pkl")

print("✅ Best Model Saved as best_model.pkl!")


# Plot predictions vs actual values
plt.figure(figsize=(12, 5))
sns.scatterplot(x=y_test, y=y_pred_lr, label="Linear Regression", alpha=0.7)
sns.scatterplot(x=y_test, y=y_pred_rf, label="Random Forest", alpha=0.7)
sns.scatterplot(x=y_test, y=y_pred_xgb, label="XGBoost", alpha=0.7)
sns.scatterplot(x=y_test, y=y_pred_dt, label="Decision Tree", alpha=0.7)
plt.plot(y_test, y_test, color='black', linestyle='dashed', label="Ideal Prediction")
plt.xlabel("Actual Temperature Anomalies")
plt.ylabel("Predicted Temperature Anomalies")
plt.title("Model Predictions vs. Actual Values")
plt.legend()
plt.savefig('EDA_graphs/scatterplot_pred_vs_actual')

# Plot residuals
plt.figure(figsize=(12, 5))
sns.histplot(y_test - y_pred_lr, label="Linear Regression Residuals", kde=True, color='yellow', bins=20, alpha=0.6)
sns.histplot(y_test - y_pred_rf, label="Random Forest Residuals", kde=True, color='green', bins=20, alpha=0.6)
sns.histplot(y_test - y_pred_xgb, label="XGBoost Residuals", kde=True, color='red', bins=20, alpha=0.6)
sns.histplot(y_test - y_pred_dt, label="Decision Tree Residuals", kde=True, color='purple', bins=20, alpha=0.6)
plt.axvline(x=0, color='black', linestyle='dashed')
plt.xlabel("Residual Error")
plt.title("Residual Distribution of Models")
plt.legend()
plt.savefig('EDA_graphs/residualplot_pred_vs_actual')

print("✅ Model Training & Evaluation Completed!")
