import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("StudentPerformanceFactors.csv")
print("Dataset loaded shape", df.shape)
df.head()  #first 5 rows
print("Columns",df.columns)
print("\nMissing values",df.isnull().sum())
print("\nData types",df.dtypes)

# Features & Target
FEATURE_COL="Hours_Studied"
TARGET_COL="Exam_Score"
x=df[[FEATURE_COL]]
y=df[TARGET_COL]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# Linear Regression
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Linear Regression ---")
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Polynomial Regression (degree=2)
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)
y_pred_poly = poly_model.predict(x_test_poly)

mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\n--- Polynomial Regression (Degree=2) ---")
print("MSE:", mse_poly)
print("RMSE:", rmse_poly)
print("R2 Score:", r2_poly)

# -------------------------------
# Visualization
plt.figure(figsize=(8,5))

# Actual points
plt.scatter(x_test, y_test, label="Actual", color="blue")

# Linear Regression predictions
plt.scatter(x_test, y_pred_poly, color="red", label="Linear Predicted")

# Polynomial Regression curve
sorted_zip = sorted(zip(x_test[FEATURE_COL], y_pred_poly))
x_sorted, y_poly_sorted = zip(*sorted_zip)
plt.plot(x_sorted, y_poly_sorted, color="green", label="Polynomial Predicted")

plt.xlabel(FEATURE_COL)
plt.ylabel(TARGET_COL)
plt.legend()
plt.title("Linear vs Polynomial Regression")
plt.show()