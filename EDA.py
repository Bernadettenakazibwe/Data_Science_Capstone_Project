import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset with correct headers
file_path = "Global_Temp.csv"  # Use the correct file path
df = pd.read_csv(file_path, skiprows=1)  # Skip first row for proper column headers

# Replace '***' with NaN (missing values)
df.replace("***", np.nan, inplace=True)

# Convert problematic columns to numeric
columns_to_convert = ["D-N", "DJF"]
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert while handling errors

# Check for missing values
missing_values = df.isnull().sum()

# Display cleaned dataset info
#print(df.info())
#print("\nMissing Values in Each Column:")
#print(missing_values) #Two missing values one in each of D-N and DJF
#print(df.head())

df.dropna(inplace=True)

columns= ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','J-D','D-N','DJF','MAM','JJA','SON']
#print(f'Statistical Analysis: \n {df[columns].describe()}')


# Exploratory Data Analysis

# Temperature trends over years by line plot

plt.figure(figsize=(12,6))
plt.plot(df['Year'], df['J-D'], marker='*', linestyle='-', color='g', label = 'Annual Mean (J-D)')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly  (°C)')
plt.legend()
plt.grid()
#plt.savefig("temperature_anomaly_plot.png", dpi=300)

#Box plot to Detect outliers
for col in columns:

    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df[col], color="green")  # Box plot for each column
    plt.title(f"Box Plot of {col}", fontsize=14)
    plt.ylabel(f'Temperature anomaly for {col}', fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add grid for readability
    plt.show()
    #plt.savefig(f"Boxplot{col}.png", dpi=300)


   

#Function to remove outliers
def remove_outliers(df, columns):
    Q1 = df[columns].quantile(0.25) #Lower quantile 
    #Q2 = df[columns].quartile(0.50) We only need Q1 and Q3
    Q3 = df[columns].quantile(0.75) # Upper quantile
    IQR = Q3 -Q1 # Interquantile Range

    lower_bound = Q1 -1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    #Filter Dataset to remove outliers

    df_no_outliers = df[((df[columns]< lower_bound) | (df[columns]>upper_bound)).any(axis=1) ]
    return df_no_outliers


df_cleaned = remove_outliers(df,columns)

#print(df_cleaned.head())


# Re- get the boxplots to see if the outliers are removed

#Box plot to Detect outliers
for col in columns:

    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df_cleaned[col], color="green")  # Box plot for each column
    plt.title(f"Box Plot of {col}", fontsize=14)
    plt.ylabel(f'Temperature anomaly for {col}', fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add grid for readability
    plt.show()
    plt.savefig((f'Box_plots_no_outliers/Boxplot{col}.png'), dpi=300)


#Feature Engineering - creating new features like seasons, 5-year moving average

# Convert 'Year' to integer
df_cleaned["Year"] = df_cleaned["Year"].astype(int)

# 🔹 1. Create a 5-Year Moving Average for Annual Temperature Anomaly (J-D)
df_cleaned["5-Year Moving Avg"] = df_cleaned["J-D"].rolling(window=5, min_periods=1).mean()

# 🔹 2. Extract Month Feature (Create a new column for each month)
df_long = df_cleaned.melt(id_vars=["Year"], value_vars=df_cleaned.columns[1:-5], var_name="Month", value_name="Temp_Anomaly")

# 🔹 3. Encode Seasons as Categorical Variables
season_mapping = {
    "DJF": "Winter",
    "MAM": "Spring",
    "JJA": "Summer",
    "SON": "Fall"
}

df_season = df_cleaned[["Year", "DJF", "MAM", "JJA", "SON"]].melt(id_vars=["Year"], var_name="Season", value_name="Temp_Anomaly")
df_season["Season"] = df_season["Season"].map(season_mapping)

# 🔹 4. Normalize Temperature Anomaly Data
scaler = MinMaxScaler()
df_cleaned[["J-D", "D-N", "DJF", "MAM", "JJA", "SON"]] = scaler.fit_transform(df_cleaned[["J-D", "D-N", "DJF", "MAM", "JJA", "SON"]])

# Save the processed dataset
df.to_csv("Feature_Engineered_Global_Temp.csv", index=False)

print("✅ Feature Engineering Complete! Data saved as 'Feature_Engineered_Global_Temp.csv'") 

# You need to save all the features in one file to use during the Exploratory data analysis