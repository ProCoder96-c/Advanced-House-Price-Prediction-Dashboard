"""
House Price Prediction using Machine Learning
A complete ML project demonstrating regression analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
print(data.columns)

# Set random seed
np.random.seed(42)

print("="*60)
print("HOUSE PRICE PREDICTION SYSTEM")
print("="*60)

# ==========================================
# 1. DATA GENERATION
# ==========================================
print("\n1. Generating synthetic housing dataset...")

n_samples = 500

area = np.random.uniform(500, 5000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.randint(0, 50, n_samples)
location_score = np.random.uniform(1, 10, n_samples)

price = (area * 150 +
         bedrooms * 15000 +
         bathrooms * 10000 +
         location_score * 20000 -
         age * 2000 +
         np.random.normal(0, 50000, n_samples))

data = pd.DataFrame({
    'Area_sqft': area,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Age_years': age,
    'Location_Score': location_score,
    'Price': price
})

print(f"Generated {len(data)} samples")
print("\nDataset Preview:")
print(data.head())

# ==========================================
# 2. PREPROCESSING
# ==========================================
print("\n2. Preprocessing data...")

X = data.drop('Price', axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MODEL TRAINING
# ==========================================
print("\n3. Training models...")

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# ==========================================
# 4. EVALUATION
# ==========================================
print("\n4. Evaluating models...")

def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")

    return r2

lr_r2 = evaluate(y_test, lr_pred, "Linear Regression")
rf_r2 = evaluate(y_test, rf_pred, "Random Forest")

# ==========================================
# 5. VISUALIZATION
# ==========================================
print("\n5. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Heatmap
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=axes[0, 0])
axes[0, 0].set_title("Correlation Matrix")

# Linear Regression Plot
axes[0, 1].scatter(y_test, lr_pred)
axes[0, 1].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--')
axes[0, 1].set_title("Linear Regression")

# Random Forest Plot
axes[1, 0].scatter(y_test, rf_pred)
axes[1, 0].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--')
axes[1, 0].set_title("Random Forest")

# Feature Importance
importance = pd.Series(rf_model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind='barh', ax=axes[1, 1])
axes[1, 1].set_title("Feature Importance")

plt.tight_layout()

# ✅ FIXED PATH (no error)
plt.savefig("model_analysis.png")
print("Saved: model_analysis.png")

# ==========================================
# 6. SAMPLE PREDICTION
# ==========================================
print("\n6. Sample Prediction")

sample = pd.DataFrame({
    'Area_sqft': [2500],
    'Bedrooms': [3],
    'Bathrooms': [2],
    'Age_years': [10],
    'Location_Score': [8]
})

sample_scaled = scaler.transform(sample)

print("LR Prediction:", lr_model.predict(sample_scaled)[0])
print("RF Prediction:", rf_model.predict(sample_scaled)[0])

# ==========================================
# 7. SAVE DATASET
# ==========================================

# ✅ FIXED PATH (no error)
data.to_csv("housing_dataset.csv", index=False)
print("\nSaved: housing_dataset.csv")

print("\n✅ PROJECT COMPLETED SUCCESSFULLY")