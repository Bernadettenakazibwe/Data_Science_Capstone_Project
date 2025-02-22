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