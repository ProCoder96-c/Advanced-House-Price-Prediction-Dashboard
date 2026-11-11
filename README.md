# HouseAI: Advanced House Price Prediction Dashboard

A Streamlit dashboard that predicts house prices with gradient-boosted models (LightGBM and XGBoost), lets you upload your own housing CSV, and plots the data with Plotly and Folium.

## Results

Evaluated on a held-out 20% test split (1,000 of 5,000 rows, `random_state=42`). Predictions are converted back from log prices before scoring.

| Model | R² | MAE (price units) |
|---|---|---|
| XGBoost | 0.900 | 88,410 |
| LightGBM (used by the dashboard) | 0.896 | 89,643 |

Re-run `python train_model.py` to reproduce these numbers.

## Data

`housing.csv` has 5,000 rows with five numeric features and a `Price` target:

- Avg. Area Income
- Avg. Area House Age
- Avg. Area Number of Rooms
- Avg. Area Number of Bedrooms
- Area Population

The free-text `Address` column is dropped before training.

## Pipeline

1. Drop missing rows and the `Address` column.
2. Label-encode any text columns.
3. Log-transform the target with `log1p` to reduce skew.
4. Split 80/20 and scale features with `StandardScaler`.
5. Train XGBoost and LightGBM (300 trees each, learning rate 0.05).
6. Save `xgb_model.pkl`, `lgbm_model.pkl` and `scaler.pkl`.

## Dashboard features

- Upload any housing CSV from the sidebar, or use the bundled dataset.
- Sliders and dropdowns for property details, with a live price estimate.
- Folium map in the sidebar.
- Model performance panel.
- Plotly price distribution and feature-versus-price charts.

## Run it

```bash
git clone https://github.com/ProCoder96-c/Advanced-House-Price-Prediction-Dashboard.git
cd Advanced-House-Price-Prediction-Dashboard
pip install -r requirements.txt
python train_model.py        # optional: retrains and rewrites the .pkl files
streamlit run app.py
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `train_model.py` | Trains and evaluates XGBoost and LightGBM, saves the models |
| `housing.csv` | Main dataset (5,000 rows) |
| `house_price_prediction.py` | Earlier baseline script: generates a synthetic 500-row dataset (`housing_dataset.csv`), compares linear regression and random forest, and saves `model_analysis.png` |
| `model_analysis.png` | Plots from the baseline script: correlation matrix, model fit, feature importance |

## Tech stack

Python, Streamlit, LightGBM, XGBoost, scikit-learn, pandas, NumPy, Plotly, Folium, joblib.
