import pandas as pd # for data manipulation and analysis
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # built on top of matplotlib for plotting to look nicer
from sklearn.datasets import fetch_california_housing # for loading the dataset
import numpy as np # for numerical operations


print("Loading California Housing dataset...")
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['MedHouseVal'] = housing.target

print(f"Dataset shape: {data.shape}")
print(f"Features: {list(data.columns)}")
print(f"Target: {data.columns[-1]}")
print("\nFirst few rows:")
print(data.head())

plt.figure(figsize=(10, 6))

plt.hist(data['MedHouseVal'], bins=50, alpha=0.5, color='skyblue', edgecolor='red')

plt.title('Distribution of House Prices in California', fontsize=14, fontweight='bold')

plt.xlabel('House Price (in $100,000s)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('house_price_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


print("\n" + "="*50)
print("2. CORRELATION HEATMAP")
print("="*50)

correlation_matrix = data.corr()

print(correlation_matrix)

plt.figure(figsize=(12, 10))

sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f', square=True, cbar_kws={'shrink': 0.8})

plt.title('Correlation Matrix of Housing Features', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()

plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')

plt.show()

house_price_corr = correlation_matrix['MedHouseVal'].sort_values(ascending=False)

for feature, corr_value in house_price_corr.items():
    print(f"{feature}: {corr_value:.3f}")


print("\n" + "="*50)
print("3. SCATTER PLOT: Median Income vs House Value")
print("="*50)

plt.figure(figsize=(8, 6))

plt.scatter(data['MedInc'], data['MedHouseVal'], alpha=0.5, color='red')

plt.title('House Value vs Median Income', fontsize=14, fontweight='bold')

plt.xlabel('Median Income (in $10,000s)', fontsize=12)

plt.ylabel('House Value (in $100,000s)', fontsize=12)

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scatter_medinc_vs_houseval.png', dpi=300, bbox_inches='tight')

plt.show()

print("\nNotice how higher median income neighborhoods tend to have higher house values. This is a strong positive relationship, which is why MedInc is so important for predicting house prices.")


print("\n" + "="*50)
print("4. SPLITTING DATA FOR LINEAR REGRESSION")
print("="*50)


from sklearn.model_selection import train_test_split

X = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Test target shape: {y_test.shape}")


print("\n" + "="*50)
print("5. FEATURE SCALING (Standardization)")
print("="*50)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Mean of scaled training features (should be close to 0): {X_train_scaled.mean(axis=0)}")
print(f"Std of scaled training features (should be close to 1): {X_train_scaled.std(axis=0)}")


print("\n" + "="*50)
print("6. VISUALIZING SCALING EFFECT (MedInc & HouseAge)")
print("="*50)

features = ['MedInc', 'HouseAge']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(X_train[features]['MedInc'], bins=30, alpha=0.7, label='MedInc')
plt.hist(X_train[features]['HouseAge'], bins=30, alpha=0.7, label='HouseAge')
plt.title('Before Scaling')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(X_train_scaled[:, 0], bins=30, alpha=0.7, label='MedInc (scaled)')
plt.hist(X_train_scaled[:, 1], bins=30, alpha=0.7, label='HouseAge (scaled)')
plt.title('After Scaling')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig('scaling_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nNotice how both features are now centered around 0 and have similar spread after scaling. This helps the model treat them equally!")