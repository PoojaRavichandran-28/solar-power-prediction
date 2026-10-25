"""
Solar Power Generation Prediction - training script

Dataset: Kaggle "Solar Power Generation Data" (Plant 1)
Put these two files in the same folder as this script:
    Plant_1_Generation_Data.csv
    Plant_1_Weather_Sensor_Data.csv

Run:  python train.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------- 1. Load
gen = pd.read_csv("Plant_1_Generation_Data.csv")
weather = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")

# The two files use different date formats
gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], dayfirst=True)
weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"])

# Total plant output per timestamp (sum over all inverters)
gen = gen.groupby("DATE_TIME", as_index=False)["DC_POWER"].sum()

df = gen.merge(
    weather[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]],
    on="DATE_TIME",
    how="inner",
).sort_values("DATE_TIME")

# ---------------------------------------------------------------- 2. Clean + features
df = df.dropna()
df["hour"] = df["DATE_TIME"].dt.hour + df["DATE_TIME"].dt.minute / 60
df["month"] = df["DATE_TIME"].dt.month

FEATURES = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "hour", "month"]
TARGET = "DC_POWER"

# ---------------------------------------------------------------- 3. Time-based split (no shuffling)
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# ---------------------------------------------------------------- 4. Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results, preds = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    preds[name] = p
    results[name] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, p)),
        "MAE": mean_absolute_error(y_test, p),
        "R2": r2_score(y_test, p),
    }

results_df = pd.DataFrame(results).T.round(3)
print("\nModel comparison:\n", results_df)

best_name = results_df["R2"].idxmax()
best_model = models[best_name]
print(f"\nBest model: {best_name}")

joblib.dump({"model": best_model, "features": FEATURES}, "solar_model.joblib")

# ---------------------------------------------------------------- 5. Plots
# Actual vs predicted (last 3 days of test set)
n = 96 * 3  # 15-min data -> 96 points per day
plt.figure(figsize=(11, 4))
plt.plot(test["DATE_TIME"].iloc[-n:], y_test.iloc[-n:], label="Actual")
plt.plot(test["DATE_TIME"].iloc[-n:], preds[best_name][-n:], label=f"Predicted ({best_name})")
plt.ylabel("DC power (kW)")
plt.title("Actual vs predicted solar power")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)

# Feature importance (tree models only)
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(6, 3.5))
    imp.plot(kind="barh")
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)

# Daily pattern
plt.figure(figsize=(7, 3.5))
df.groupby(df["DATE_TIME"].dt.hour)[TARGET].mean().plot()
plt.xlabel("Hour of day")
plt.ylabel("Mean DC power (kW)")
plt.title("Average daily generation profile")
plt.tight_layout()
plt.savefig("daily_profile.png", dpi=150)

print("\nSaved: solar_model.joblib, actual_vs_predicted.png, feature_importance.png, daily_profile.png")
