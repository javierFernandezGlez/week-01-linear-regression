# Step 1: Import necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Step 2: Load the California Housing dataset
data = fetch_california_housing(as_frame=True)

# Step 3: Convert to pandas DataFrame
X = data.data  # Features
y = data.target  # Target (median house value in $100,000s)
df = X.copy()
df['MedHouseVal'] = y

# Step 4: Explore the data
print("\n--- First 5 rows of the dataset ---")
print(df.head())

print("\n--- Dataset shape (rows, columns) ---")
print(df.shape)

print("\n--- Feature names ---")
print(list(X.columns))

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Basic statistics ---")
print(df.describe())

print("\n--- Target variable (MedHouseVal) ---")
print(f"Min: {y.min()} | Max: {y.max()} | Mean: {y.mean()} | Std: {y.std()}")

# Optional: Show missing values
print("\n--- Missing values per column ---")
print(df.isnull().sum())

print("\n✅ Data loaded and explored!") 