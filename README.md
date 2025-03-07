# Global Temperature Anomalies Prediction Project

## Introduction: 
  In this project, we are using the power of machine learning to create a model that can help users to predict the future of the Global Temperature anomalies.

## About the project:


### Data Source : 

 [Global_Temp.csv](Data_Science_Capstone_Project/Global_Temp.csv)
### Tools  and Technologies used:

• Python (for data processing and machine learning) 
 o Pandas and NumPy (data manipulation) 
 o Scikit-learn (machine learning) 
 o Matplotlib, Seaborn (visualization) 
 o Statsmodels (statistical modeling) 
 • Streamlit (for deployment as an interactive web application)
 • Tableau/Power BI (for advanced visualizations and dashboards) 
 •VS code(IDE for coding)

###Data Overview:
Year   Jan  Feb   Mar   Apr  May   Jun   Jul   Aug   Sep      D-N       DJF     MAM     JJA     SON     5-Year Moving Avg  Month  Temp_Anomaly_x  Season  Temp_Anomaly_y
0  2015  0.87  0.9  0.96  0.75  0.8  0.81  0.73  0.79  0.84  0.044444  0.166667  0.0625  0.0  0.263158                0.9    Jan            0.87  Winter            0.85
1  2015  0.87  0.9  0.96  0.75  0.8  0.81  0.73  0.79  0.84  0.044444  0.166667  0.0625  0.0  0.263158                0.9    Jan            0.87  Spring            0.84
2  2015  0.87  0.9  0.96  0.75  0.8  0.81  0.73  0.79  0.84  0.044444  0.166667  0.0625  0.0  0.263158                0.9    Jan            0.87  Summer            0.78
3  2015  0.87  0.9  0.96  0.75  0.8  0.81  0.73  0.79  0.84  0.044444  0.166667  0.0625  0.0  0.263158                0.9    Jan            0.87    Fall            0.99
4  2015  0.87  0.9  0.96  0.75  0.8  0.81  0.73  0.79  0.84  0.044444  0.166667  0.0625  0.0  0.263158                0.9    Feb            0.90  Winter            0.85

 
 

### Data Cleaning: 
 - Replace '***' with NaN (missing values)
 - Converted non-numeric columns  to numeric forexample the "D-N", "DJF" columns to ensure all values in the numeric columns are numeric.
 - Removed missing values using 'dropna' funtion. There were missing values in columns e.g D-N and DJF
 - Detected outliers using boxplots in some columns  like month cloumns for September, November, March, October and SON column.
 - Removed or reduced outliers 
 - Created a month column , Temp_anomaly_x , Season and Temp_anomaly_y.  Temp_anomaly_x  corresponds to monthly temperature anomalies, and  Temp_anomaly_y corresponds to the seasonal temperature anomalies.
 - Renamed the season basing on DJF: "Winter", MAM: "Spring", JJA: "Summer", "SON": "Fall".
 - Saved the datafrom the first phase of cleaning in  [Feature_Engineered_Global_Temp.csv](Data_Science_Capstone_Project/Feature_Engineered_Global_Temp.csv)

### Exploaratory Analysis 1:
EDA with python: 
Key Insights:
  
  ## 1. Distribution of Temperature anomalies:
 
  ![Distribution_of_Global_Temp_Anomalies](Data_Science_Capstone_Project/Distribution_of_Global_Temp_Anomalies.png)

  1. This graph shows the distributin of temperature anomalies. There are multiple peaks meaning that some temperature anomalies occur more frequently than others.

  2. The KDE curve (smooth green line) estimates the probability distribution of anomalies.
  3. the KDE curve shows multiple peaks, it could indicate distinct periods of warming and cooling trends.
  4. The presence of high bars at 0 and 1°C suggests that anomalies are frequently at these values.


  <EDA_graphs/Tableu/2025-03-06 (1).png -->

  The above graph from Tableu visualisation shows also the distribution of Temperature anomalies over years with Temperature anomalies between 0 and 0.15 °C showing the highest freqeuncies.

  The Temperature anomalies between  0.70 and 1.05 °C show the lowest frequency.

 ## Trend in the Yearly Temperature Anomalies. 
 <EDA_graphs/Global_Temperature_Anomaly_Trends_to_Present).png --> 

 1. The above graph  shows trends in the Yearly Temperature anomalies(J-D) for the past 10 years with the temperature anomaly fluctuating  but increases significantly from  2022.
 2. Some years show a sharp rise, while others have a temporary decline before rising again.

 3. The recent sharp increase indicates accelerated warming in the past few decades.
 4. The increasing trend suggests global warming, where temperatures are rising over time.

 ## Correlation between the temperature anomalies for months, seasons and annual anomalies.
 <EDA_graphs/Heatmap_of_Temp_anomalies.png -->

 1. The heatmap shows that temperature anomalies are generally consistent across months and seasons, meaning that if one month is unusually warm, nearby months are also likely to be warm. Forexample there is high correlation between January and the next five months(until May), also there is a zero or less correlation for the next months with January. This is the same trend with other months.
 2. Stronger correlations between seasonal and annual averages suggest that longer-term warming trends are evident. This shown by the annual temperature anomaliy (j-d) having high correlation across all months and seasons.

 3. The negative correlations  indicate seasonal shifts forexample shift from summer to winter or winter/fall to summer. This is also indicated in months.


# Model Development and Evaluation

## Data Preprocessing
 To build an accurate model for predicting global temperature anomalies, I  performed several data preprocessing steps to enhance the quality and usability of the dataset:
   Data source file: <Feature_Engineered_Global_Temp.csv -->
  1. **Handling Categorical Variables:**
   - One-hot encoding was applied to the `Season` column, converting it into numerical format.
   - The `Month` column was converted from textual representation (e.g., "Jan") to numerical values (1-12).
   
2. **Feature Engineering:**
   - **Lag Features:** Created lagged versions of the target variable (`J-D`) to incorporate historical dependencies (Lag_1, Lag_2, Lag_3).
   - **Rolling Statistics:** Calculated moving averages over a 3-month and 6-month window to capture trends.
   - **Cyclic Encoding:** Represented the `Month` feature as sine and cosine values to preserve its cyclical nature.
   - **Renamed anomalies:** Monthly(Temp_Anomaly_x) and seasonal(Temp_Anomaly_y) temperature anomalies were standardized as `Monthly_Temp_anomaly` and `Seasonal_Temp_anomaly` respectively.

3. **Handling Missing Data:**
   - Missing values generated by lag and rolling features were removed to ensure a clean dataset.

4. **Feature Selection:**
   - The following features were selected for training:
     - `Year`, `J-D`, `Month_sin`, `Month_cos`, `5-Year Moving Avg`, `Lag_1`, `Lag_2`, `Lag_3`, `Rolling_3`, `Rolling_6`, `Seasonal_Temp_anomaly`, and encoded season features.

## Model Training
After preprocessing, I trained four machine learning models to predict `Monthly_Temp_anomaly`:

1. **Linear Regression** (with polynomial features)
   - Used **PolynomialFeatures (degree=2)** to capture nonlinear relationships.
   - Features were **standardized** using `StandardScaler`.

2. **Random Forest Regressor**
   - Performed **hyperparameter tuning** using `GridSearchCV` with parameters:
     - `n_estimators`: [50, 100, 200]
     - `max_depth`: [None, 10, 20]

3. **XGBoost Regressor**
   - Tuned using `GridSearchCV` with parameters:
     - `n_estimators`: [50, 100, 200]
     - `learning_rate`: [0.01, 0.1, 0.2]
     - `max_depth`: [3, 6, 10]

4. **Decision Tree Regressor**
   - Tuned with parameters:
     - `max_depth`: [None, 5, 10, 20]
     - `min_samples_split`: [2, 5, 10]

The dataset was split into **80% training and 20% testing**, with `shuffle=False` to maintain temporal order.

## Model Evaluation
Each model was evaluated using the following metrics:
- **Mean Absolute Error (MAE)** – Measures the average absolute difference between predictions and actual values.
- **Mean Squared Error (MSE)** – Penalizes larger errors more heavily.
- **R² Score** – Indicates how well the model explains variance in the data.

### Results:
#### **Linear Regression**
- MAE: **4.8514**
- MSE: **27.6351**
- R² Score: **-1184.8706** (poor performance, likely overfitting due to polynomial features)

#### **Random Forest**
- MAE: **0.2824**
- MSE: **0.0996**
- R² Score: **-3.2727**

#### **XGBoost**
- MAE: **0.2900**
- MSE: **0.1031**
- R² Score: **-3.4250**

#### **Decision Tree**
- MAE: **0.3093**
- MSE: **0.1157**
- R² Score: **-3.9651**

### **Best Model Selection**
The best-performing model based on R² score was **Random Forest**, with an R² score of **-3.2727**. Despite the negative score indicating suboptimal performance, it outperformed other models.

### **Visualization of Results**
1. **Predictions vs. Actual Values:**
   - A scatter plot was generated to compare predictions of each model against actual temperature anomalies.
   - The dashed black line represents an ideal prediction scenario where `Predicted = Actual`.

   <EDA_graphs/scatterplot_pred_vs_actual.png -->

   - Linear Regression (blue points) shows large deviations from the ideal prediction line (black dashed line), meaning it struggles to capture the data patterns.
   - Random Forest (orange), XGBoost (green), and Decision Tree (red) have points more tightly clustered around the ideal line, suggesting better predictions.
   - Linear Regression predictions have high variance and outliers, while the other models maintain more stability.
   **In conclusion:**  The Random Forest, XGBoost, and Decision Tree models outperform Linear Regression in predicting temperature anomalies. The Linear Regression model exhibits poor generalization, leading to many incorrect predictions.
   
2. **Residual Distribution:**
   - Residual plots were created for all models to analyze prediction errors.
   - The best model should have residuals centered around zero with minimal variance.

   <EDA_graphs/residualplot_pred_vs_actual.png -->

   - Linear Regression has significant errors, whereas Random Forest, XGBoost, and Decision Tree models have more concentrated and normally distributed residuals, indicating better predictive performance.

## Conclusion
- The **Random Forest model** performed best but still exhibited negative R² values, suggesting further improvement is needed.
✅ The best model was saved as `best_model.pkl` for deployment.



# App Deployment

## Overview

The deployed application provides an interactive platform for users to input a year and receive either the past global temperature anomalies or the predicted future global temperature anomalies. The app also visualizes predicted vs. actual temperature anomalies over time, allowing users to explore climate trends effectively.

## Deployment Process

The application was developed using **Streamlit**, a Python framework that enables easy deployment of machine learning models with interactive user interfaces. The deployment process involved the following steps:

1. **Building the Streamlit App**:

   - The app was implemented using Python and Streamlit.
   - It loads the trained **Random Forest model**, which outperformed other models in the evaluation phase.
   - Users can select a year, and the app predicts temperature anomalies based on historical data.

2. **Setting Up the Environment**:

   - Created a virtual environment to manage dependencies:
     ```bash
     python -m venv env
     source env/bin/activate  # For macOS/Linux
     env\Scripts\activate  # For Windows
     ```
   - Installed required libraries:
     ```bash
     pip install streamlit joblib pandas numpy scikit-learn xgboost
     ```

3. **Running the App Locally**:

   - The app was tested locally using:
     ```bash
     streamlit run app.py
     ```
   -

## User Interaction

- The user interface includes a **year selection slider**.
- The app displays the predicted temperature anomaly for the selected year.
- A **graph** compares predicted and actual temperature anomalies over time.
- Users can analyze climate trends using interactive visualization tools.
<EDA_graphs/dashboard.png>

## Conclusion

The successful deployment of the machine learning model enables real-time predictions of global temperature anomalies. The app is user-friendly, interactive, and helps in understanding climate change patterns effectively. Future improvements may include additional models, enhanced UI components, and real-time data integration.

