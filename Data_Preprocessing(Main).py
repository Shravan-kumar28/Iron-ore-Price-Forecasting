# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 09:33:00 2025

@author: shrav
"""


# Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sqlalchemy import create_engine, text
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


# Load dataset from CSV file
data = pd.read_csv(r"E:/360DigiTmg/ML Project/Python/Data_set_Iron/Data_set_Iron.csv")

# Database connection setup
user = 'root'
password = 'password'
db = 'Iron_ore'

# Creating SQL engine
engine = create_engine(f"mysql+pymysql://{user}:{password}@localhost/{db}")

# Storing dataset in MySQL table
data.to_sql('iron_ore_prices', con=engine, if_exists='replace', index=False, chunksize=1000)

# Fetching data from database
sql = 'select * from iron_ore_prices'
df = pd.read_sql_query(text(sql), con=engine.connect())

# Displaying first and last records
df.head()
df.tail()

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Display dataset information
df.info()

# Display dataset shape
df.shape

# Convert 'Date' column to datetime format
df["Date"] = pd.to_datetime(data["Date"], format='%m/%d/%Y')

# Removing percentage sign and converting 'Change %' column to float
df['Change %'] = df['Change %'].str.replace("%", "").astype(float)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Dropping irrelevant column 'Vol.'
df.drop(columns=["Vol."], inplace=True)

# Selecting numeric columns for analysis
Numeric_data = df.select_dtypes(include=["number"])

# First Moment Business Decision: Central Tendency (Mean, Median, Mode)
mean_values = Numeric_data.mean()
median_values = Numeric_data.median()
mode_values = Numeric_data.mode().iloc[0]  # Mode can have multiple values, so selecting first occurrence

CentralTendency_df = pd.DataFrame({
    "Mean": mean_values,
    "Median": median_values,
    "Mode": mode_values
})

#Displaying the results
print(CentralTendency_df)

# Second Moment Business Decision: Dispersion (Variance, Std Dev, Range)
variance_values = Numeric_data.var()
std_dev_values = Numeric_data.std()
range_values = Numeric_data.max() - Numeric_data.min()

Dispersion_df = pd.DataFrame({
    "Variance": variance_values,
    "Standard Deviation": std_dev_values,
    "Range": range_values
})

#Displaying the results
print(Dispersion_df)

# Third Moment Business Decision: Skewness
skewness_values = Numeric_data.skew()
print("Skewness of each numeric column:")
print(skewness_values) #Displaying the results

# Fourth Moment Business Decision: Kurtosis
kurtosis_values = Numeric_data.kurt()
print("Kurtosis of each numerical column:")
print(kurtosis_values) #Displaying the results





""" Business Insights:-
1.Mode(126.01) is higher than both mean and median,that means 126.01 was a frequently occurring price.
2.Change %,Most days had little or no price change, confirming a stable market.
3.Mean is slightly positive, the overall trend in prices might be slightly upward over time.
4.price(Variance):- High price volatility increase financial risk.
5.right-skewed distribution,it means there are some extreme high prices, but most values are concentrated on the lower end.
6.frequent small declines in price,Buyers should purchase during dips rather than peaks.
7.kurtosis of 1.06 suggests that iron ore prices were relatively stable over time.
"""

# Calculate rolling variance with a 30-day window
rolling_variance = df["Price"].rolling(window=365).var()

# Plotting rolling variance
plt.figure(figsize=(10,6))
plt.plot(df['Date'], rolling_variance, label="365-Days Rolling Variance", color="purple")
plt.title("Rolling Variance of Iron Ore Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Variance")
plt.legend()
plt.xticks(rotation=45)
plt.show()


# Rolling Mean (Trend Analysis) for the target variable 'Price'
rolling_mean_series = df["Price"].rolling(window=365).mean()

plt.figure(figsize=(10,6))
plt.plot(df["Date"], df["Price"], label="Iron Ore Price")
plt.plot(df["Date"], rolling_mean_series, label="365-days Rolling Mean", color="red")
plt.title("365-Days Rolling Mean of Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Univariate Analysis
# Histogram
plt.figure(figsize=(10,6))
df.hist(figsize=(10,6), bins=30, edgecolor="black")
plt.title("Histograms of Numeric Columns", fontsize=14)
plt.show()

# Box-Plot (Detect Outliers)
df.plot(kind='box', subplots=True, sharey=False, figsize=(18,10))
plt.title("Outliers Detection")
plt.subplots_adjust(wspace=0.75)
plt.show()

# Q-Q Plot (Check Normality of Price Distribution)
plt.figure(figsize=(8,6))
stats.probplot(df["Price"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Price")
plt.show()

# Bivariate Analysis (Correlation Analysis)
correlation_matrix = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Multivariate Analysis (Pair Plot)
sns.pairplot(df, diag_kind="kde")
plt.title("Pair Plot", fontsize=14)
plt.show()

#Mean,median, and mode for Price, Open, High, and Low are almost identical.
#High correlation between these features
#Multicollinearity can distort model predictions and increase variance.
#Extra columns increase memory usage and processing time.
# Dropping highly correlated redundant columns
df.drop(columns=['Open', 'High', 'Low'], inplace=True)


"""
Dickey-Fuller Test (ADF Test)
Null Hypothesis (H₀): The series is stationary.
Alternative Hypothesis (H₁): The series is non-stationary.
If p-value < 0.05, reject H₀ → The series is stationary.
"""

adf_test = adfuller(df["Price"])
print(f"ADF Statistic: {adf_test[0]}")
print(f"P-Value: {adf_test[1]}")

if adf_test[1] < 0.05:
    print("The time series is STATIONARY.")
else:
    print("The time series is NON-STATIONARY.")
    
    
# Sorting data by Date and setting it as index
df = df.sort_values("Date")  

# Perform Seasonal Decomposition of Time Series
# 'additive' model assumes the data is a sum of trend, seasonality, and residual components
# 'period=365' assumes a yearly seasonal pattern (useful for daily data spanning multiple years)
decomposition = seasonal_decompose(df["Price"], model="additive", period=365) 

# Set figure size for better visualization
plt.figure(figsize=(12, 8))

# Plot Original Data (Raw Time Series)
plt.subplot(4, 1, 1)  # Create the first subplot (out of 4 rows)
plt.plot(df["Price"], label="Original Data")  # Plot the original price data
plt.legend()  # Add legend to the plot

# Plot the Trend Component
plt.subplot(4, 1, 2)  # Second subplot
plt.plot(decomposition.trend, label="Trend", color="red")  # Trend component in red
plt.legend()

# Plot the Seasonality Component
plt.subplot(4, 1, 3)  # Third subplot
plt.plot(decomposition.seasonal, label="Seasonality", color="green")  # Seasonal pattern in green
plt.legend()

# Plot the Residual (Noise) Component
plt.subplot(4, 1, 4)  # Fourth subplot
plt.plot(decomposition.resid, label="Residual (Noise)", color="gray")  # Residual component in gray
plt.legend()
# Adjust layout to prevent overlapping of plots
plt.tight_layout()
# Display the complete decomposition plots
plt.show()

#High Autocorrelation values past values strongly influence future values.
# Computing autocorrelation for different lags
lags = [1, 7, 30, 90, 180, 365]  # Daily, weekly, monthly, quarterly, semi-annual, annual
autocorr_values = [df["Price"].autocorr(lag=lag) for lag in lags]

# Created a DataFrame to display results
autocorr_df = pd.DataFrame({"Lag": lags, "Autocorrelation": autocorr_values})

print(autocorr_df)

# Applying log transformation to Price
df['Price']=np.log(df['Price'])

# Q-Q Plot (Price After log Transformation)
plt.figure(figsize=(8,6))
stats.probplot(df["Price"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Price")
plt.show()

df=df.set_index("Date")
