import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==========================
# LOAD DATA (TRY BOTH FILES)
# ==========================
try:
    data = pd.read_csv("housing.csv")
    print("✅ Using housing.csv")
except:
    data = pd.read_csv("housing_dataset(1).csv")
    print("✅ Using housing_dataset(1).csv")

print("Columns:", data.columns)

# ==========================
# TARGET COLUMN
# ==========================
target = "Price"

# ==========================
# CLEANING
# ==========================
data = data.dropna()

# ==========================
# REMOVE USELESS COLUMN
# ==========================
if "Address" in data.columns:
    data = data.drop("Address", axis=1)

# ==========================
# ENCODE TEXT (if any)
# ==========================
for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# ==========================
# FEATURE ENGINEERING
# ==========================
if "Area_sqft" in data.columns:
    data["price_per_sqft"] = data["Price"] / data["Area_sqft"]

if "Bathrooms" in data.columns and "Bedrooms" in data.columns:
    data["bath_per_bed"] = data["Bathrooms"] / data["Bedrooms"]

# ==========================
# FEATURES & TARGET
# ==========================
X = data.drop("Price", axis=1)
y = data["Price"]

# ==========================
# NORMALIZATION
# ==========================
y = np.log1p(y)

# ==========================
# SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# SCALING
# ==========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# MODELS
# ==========================
xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.05)

xgb.fit(X_train, y_train)
lgbm.fit(X_train, y_train)

# ==========================
# EVALUATION
# ==========================
def evaluate(model, name):
    pred = model.predict(X_test)
    pred = np.expm1(pred)
    actual = np.expm1(y_test)

    print(f"\n{name}")
    print("R2:", r2_score(actual, pred))
    print("MAE:", mean_absolute_error(actual, pred))

evaluate(xgb, "XGBoost")
evaluate(lgbm, "LightGBM")

# ==========================
# SAVE
# ==========================
joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(lgbm, "lgbm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ DONE: Models saved")