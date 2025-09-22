# main.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import joblib

# 1. Load dataset
df = pd.read_csv("data/house_price_cleaned.csv")  # make sure this path is correct

# 2. Separate features and target
X = df.drop("SalePrice", axis=1)
y = np.log1p(df["SalePrice"])  # log-transform target

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature engineering: identify categorical and numeric columns
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 5. Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# 6. Define XGBoost model
model = Pipeline([
    ("preprocessing", preprocessor),
    ("xgb", XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42
    ))
])

# 7. Train model
model.fit(X_train, y_train)

# 8. Predict and evaluate
y_pred_log = model.predict(X_test)
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae = mean_absolute_error(y_test_real, y_pred_real)
r2 = r2_score(y_test_real, y_pred_real)

print("\n Model Evaluation:")
print(f"R² Score      : {r2:.4f}")
print(f"RMSE (dollars): ${rmse:,.2f}")
print(f"MAE (dollars) : ${mae:,.2f}")

# 9. Save model (optional)
joblib.dump(model, "results/xgb_model_pipeline.pkl")
print("\n🎉 Model saved to 'results/xgb_model_pipeline.pkl'")
