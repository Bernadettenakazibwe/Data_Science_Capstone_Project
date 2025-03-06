# Columns in dataframe are:
#'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
# 'Nov', 'Dec', 'J-D', 'D-N', 'DJF', 'MAM', 'JJA', 'SON',
# '5-Year Moving Avg', 'Month', 'Temp_Anomaly_x', 'Season',
#  'Temp_Anomaly_y'

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the feature-engineered dataset
df = pd.read_csv("Feature_Engineered_Global_Temp.csv")

# Convert 'Year' to datetime format for time series plots
df["Year"] = pd.to_datetime(df["Year"], format="%Y")

# Set 'Year' as the index for easier time series analysis
df.set_index("Year", inplace=True)

# Preview the dataset
print(df.head())
print(df.columns)


plt.figure(figsize=(20, 10))
plt.plot(df,  df["J-D"], label="Annual Temperature Anomaly (J-D)", color="red")

plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.title("Global Temperature Anomaly Trends (1880 - Present)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("EDA_graphs/Global_Temperature_Anomaly_Trends_to_Present).png", dpi=300)




#Monthly Temperature Anomaly trend

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index.year, y="Temp_Anomaly_x", hue="Month", palette="coolwarm")

plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.title("Monthly Temperature Anomalies Over Time")
plt.legend(title="Month", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("EDA_graphs/Monthly_Temperature_Anomalies_Over_Time).png", dpi=300)


# Seasonal Anomaly Over the Years
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index.year, y="Temp_Anomaly_y", hue="Season", palette="Set2")

plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.title("Seasonal Temperature Anomalies Over Time")
plt.legend(title="Season", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig('EDA_graphs/Seasonal_Temperature_Anomalies_Over_Time')


# Frequency distribution of Temperature Anomaly over years.
plt.figure(figsize=(10, 5))
sns.histplot(df["J-D"], bins=30, kde=True, color="green")

plt.xlabel("Temperature Anomaly (°C)")
plt.ylabel("Frequency")
plt.title("Distribution of Global Temperature Anomalies")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig('EDA_graphs/Distribution_of_Global_Temp_Anomalies')

#Correlation heatmap
numeric_columns= ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
'Nov', 'Dec', 'J-D', 'D-N', 'DJF', 'MAM', 'JJA', 'SON',
'5-Year Moving Avg', 'Temp_Anomaly_x',
'Temp_Anomaly_y']

plt.figure(figsize=(12, 9))
sns.heatmap(df[numeric_columns].corr(), annot=False, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap of Temperature Anomalies")
plt.savefig('EDA_graphs/Heatmap_of_Temp_anomalies')
