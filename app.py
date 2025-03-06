import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
import streamlit as st
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Load model, scaler, and transformation pipeline
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
poly_transform = joblib.load("poly_transform.pkl")

# Load dataset
df = pd.read_csv("Preprocessed.csv")

# Streamlit UI
st.title("🌍 Global Temperature Anomaly Prediction Dashboard")
st.sidebar.header("User Input")

# Select year for prediction
year = st.sidebar.slider("Select Year", min_value=int(df["Year"].min()), max_value=int(df["Year"].max()) + 10, value=2025)

# Feature engineering
df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

# Extract latest data for lagging features
latest_data = df.iloc[-1]

# Prepare input features
input_data = pd.DataFrame({
    "Year": [year],
    "J-D": [latest_data["J-D"]],
    "Month_sin": [np.sin(2 * np.pi * 6 / 12)],  # Defaulting to June
    "Month_cos": [np.cos(2 * np.pi * 6 / 12)],
    "5-Year Moving Avg": [latest_data["5-Year Moving Avg"]],
    "Lag_1": [latest_data["Lag_1"]],
    "Lag_2": [latest_data["Lag_2"]],
    "Lag_3": [latest_data["Lag_3"]],
    "Rolling_3": [latest_data["Rolling_3"]],
    "Rolling_6": [latest_data["Rolling_6"]],
    "Seasonal_Temp_anomaly": [latest_data["Seasonal_Temp_anomaly"]]
})

# Add season encoding dynamically
season_columns = [col for col in df.columns if col.startswith("Season_")]
input_data[season_columns] = latest_data[season_columns]


# Scale and transform input features
input_scaled = scaler.transform(input_data)
input_poly = input_scaled if isinstance(model, RandomForestRegressor) else poly_transform.transform(input_scaled)

# Predict temperature anomaly
prediction = model.predict(input_poly)[0]
confidence_interval = 0.1  # Assumed based on residuals analysis

st.subheader("📌 Prediction Result")
st.write(f"Predicted Temperature Anomaly for {year}: **{prediction:.4f}°C**")
st.write(f"Confidence Interval: [{prediction - confidence_interval:.4f}°C, {prediction + confidence_interval:.4f}°C]")

# Function to convert sine & cosine values back to month names
def get_month_name(sin_val, cos_val):
    angle = math.atan2(sin_val, cos_val)
    month_index = round(12 * angle / (2 * np.pi)) % 12
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month_index]

# Convert sine & cosine back to month names
df["Month"] = df.apply(lambda row: get_month_name(row["Month_sin"], row["Month_cos"]), axis=1)

# Historical Monthly Trends
st.subheader(f"📈 Monthly Temperature Trends for {year}")

if year in df["Year"].values:
    df_selected_year = df[df["Year"] == year]
    df_selected_year["Month"] = pd.Categorical(df_selected_year["Month"], 
                                               categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], ordered=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=df_selected_year["Month"], y=df_selected_year["Monthly_Temp_anomaly"], marker="o", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature Anomaly (°C)")
    ax.set_title(f"Temperature Anomaly Trends for {year}")
    ax.grid(True)
else:
    st.write(f"⚠️ No historical data for {year}. Showing predicted trends instead.")

    # Generate predictions for all months
    expected_features = scaler.feature_names_in_
    default_values = {feature: 0 for feature in expected_features}

    monthly_predictions = []
    for month_index in range(1, 13):
        input_data = default_values.copy()
        input_data["Year"] = year
        input_data["Month_sin"] = np.sin(2 * np.pi * month_index / 12)
        input_data["Month_cos"] = np.cos(2 * np.pi * month_index / 12)

        input_df = pd.DataFrame([input_data])[expected_features]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        monthly_predictions.append(prediction)

    future_monthly_df = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "Predicted_Anomaly": monthly_predictions
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=future_monthly_df["Month"], y=future_monthly_df["Predicted_Anomaly"], marker="o", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Predicted Temperature Anomaly (°C)")
    ax.set_title(f"Predicted Monthly Temperature Anomalies for {year}")
    ax.grid(True)

st.pyplot(fig)

# Future Predictions (Next 10 Years)
st.subheader("🔮 Future Predictions")
future_years = np.arange(df["Year"].max() + 1, df["Year"].max() + 11)


future_predictions = []
for y in future_years:
    input_data["Year"] = y

    if y >= df["Year"].max():   # Future years
        #input_data = default_values.copy()
        input_df = pd.DataFrame([input_data]) 
        input_scaled = scaler.transform(input_df)  # Ensure proper format
        input_transformed = poly_transform.transform(input_scaled) if isinstance(model, LinearRegression) else input_scaled
        future_predictions.append(model.predict(input_transformed)[0])
    else:  # For past years
        input_data = np.array([[selected_year]])  # Convert to a 2D NumPy array
        future_predictions = df[df["Year"] == y]["Monthly_Temp_anomaly"].values[0]

    

    #input_scaled = scaler.transform(input_data)
    #input_transformed = poly_transform.transform(input_scaled) if isinstance(model, LinearRegression) else input_scaled
    #future_predictions.append(model.predict(input_transformed)[0])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(future_years, future_predictions, marker='o', linestyle='dashed', label="Predicted Anomalies")
ax.fill_between(future_years, np.array(future_predictions) - confidence_interval, 
                np.array(future_predictions) + confidence_interval, alpha=0.2)
ax.set_xlabel("Year")
ax.set_ylabel("Predicted Temperature Anomaly (°C)")
ax.legend()
st.pyplot(fig)

st.write("🔍 This model predicts future temperature anomalies based on historical patterns. Uncertainty is represented with a shaded confidence interval.")

# Save Predictions to CSV
output_file = "future_predictions.csv"
monthly_predictions_data = [[year, month, pred] for month, pred in zip(future_monthly_df["Month"], future_monthly_df["Predicted_Anomaly"])]
df_predictions = pd.DataFrame(monthly_predictions_data, columns=["Year", "Month", "Predicted_Anomaly"])

if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)
    df_final = pd.concat([df_existing, df_predictions], ignore_index=True)
else:
    df_final = df_predictions

df_final.to_csv(output_file, index=False)
st.success(f"✅ Predictions for {year} saved to {output_file}")
