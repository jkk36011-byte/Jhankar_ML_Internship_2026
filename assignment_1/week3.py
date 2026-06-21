# -*- coding: utf-8 -*-
"""Agricultural Yield Dataset"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""PART-A: Understanding the Dataset"""

# Question 1: Dataset Overview
df = pd.read_csv('/content/agricultural_yield_dataset.csv')

print("First 10 Records:")
print(df.head(10))

print("\nRows and Columns:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

# Question 2: Data Types and Missing Values
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Question 3: Descriptive Statistics
print("\nSummary Statistics:")
print(df.describe())

print("\nMean Values:")
print(df.mean(numeric_only=True))

print("\nStandard Deviation:")
print(df.std(numeric_only=True))

"""PART-B: Exploratory Data Analysis"""

# Question 4: Distribution Analysis

plt.figure(figsize=(6,4))
plt.hist(df['rainfall_mm'], bins=20)
plt.title("Rainfall Distribution")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df['temperature_c'], bins=20)
plt.title("Temperature Distribution")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df['fertilizer_kg'], bins=20)
plt.title("Fertilizer Distribution")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df['yield_ton_per_hectare'], bins=20)
plt.title("Yield Distribution")
plt.show()

# Question 5: Crop Type Analysis

print("\nCrop Type Frequency:")
print(df['crop_type'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='crop_type', data=df)
plt.title("Crop Type Count")
plt.show()

# Most frequent crop
print("Most Frequent Crop:")
print(df['crop_type'].value_counts().idxmax())

# Question 6: Soil Type Analysis

print("\nSoil Type Frequency:")
print(df['soil_type'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='soil_type', data=df)
plt.title("Soil Type Count")
plt.show()

print("Most Common Soil Type:")
print(df['soil_type'].value_counts().idxmax())

# Question 7: Yield Distribution

plt.figure(figsize=(6,4))
plt.hist(df['yield_ton_per_hectare'], bins=20)
plt.title("Yield Distribution")
plt.show()

# Question 8: Scatter Plot Analysis

plt.figure(figsize=(6,4))
plt.scatter(df['rainfall_mm'], df['yield_ton_per_hectare'])
plt.xlabel("Rainfall")
plt.ylabel("Yield")
plt.title("Rainfall vs Yield")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df['fertilizer_kg'], df['yield_ton_per_hectare'])
plt.xlabel("Fertilizer")
plt.ylabel("Yield")
plt.title("Fertilizer vs Yield")
plt.show()

# Question 9: Correlation Analysis

numerical_df = df.select_dtypes(include=np.number)

corr_matrix = numerical_df.corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

print("\nCorrelation with Yield:")
print(corr_matrix['yield_ton_per_hectare'].sort_values(ascending=False))

# Question 10: Group Based Analysis

crop_avg_yield = df.groupby('crop_type')['yield_ton_per_hectare'].mean()
print("\nAverage Yield by Crop Type:")
print(crop_avg_yield)

soil_avg_yield = df.groupby('soil_type')['yield_ton_per_hectare'].mean()
print("\nAverage Yield by Soil Type:")
print(soil_avg_yield)

print("\nHighest Yield Crop:")
print(crop_avg_yield.idxmax())

print("\nHighest Yield Soil:")
print(soil_avg_yield.idxmax())

"""PART-C: Data Preparation"""

# Question 11: Feature Encoding

print("\nCategorical Columns:")
print(df.select_dtypes(include='object').columns)

df_encoded = pd.get_dummies(df, drop_first=True)

print("\nFirst 5 Rows After Encoding:")
print(df_encoded.head())

# Question 12: Feature Selection

X = df_encoded.drop('yield_ton_per_hectare', axis=1)
y = df_encoded['yield_ton_per_hectare']

print("\nFeatures (X):")
print(X.head())

print("\nTarget Variable (y):")
print(y.head())

"""PART-D: Machine Learning"""

# Question 13: Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# Question 14: Linear Regression Model

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)

print("\nIntercept:")
print(lr.intercept_)

print("\nCoefficients:")
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
})

print(coef_df)

print("\nHighest Positive Coefficient:")
print(coef_df.loc[coef_df['Coefficient'].idxmax()])
