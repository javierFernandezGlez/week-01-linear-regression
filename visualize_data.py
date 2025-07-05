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