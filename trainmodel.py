import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Linear Regression 

# df = pd.read_csv("laptops_simple.csv")

# print(df.head())
# print(df.info())
# print(df.shape)
# print(df.isnull().sum())

# X = df.drop("price", axis=1)
# y = df["price"]

# X_one_encode = pd.get_dummies(
#     X,
#     columns = ["brand","category","processor","graphics"],
#     drop_first = True,
#     dtype = int
# )

# X_train, X_test, y_train, y_test = train_test_split(
#     X_one_encode,y,
#     test_size=0.2,
#     random_state=42
# )

# model = LinearRegression()
# model.fit(X_train,y_train)

# y_pred = model.predict(X_test)

# print(X_test)
# print(y_pred)

# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print("MAE:", round(mae, 2))
# print("MSE:", round(mse, 2))
# print("RMSE:", round(rmse, 2))
# print("R2 Score:", round(r2, 4))

# results = pd.DataFrame({
#     "Actual Price": y_test.values,
#     "Predicted Price": np.round(y_pred, 2)
# })
# print(results.head(15))

#Decision Tree 

# df = pd.read_csv("laptops_simple.csv")

# print(df.head())
# print(df.info())
# print(df.shape)
# print(df.isnull().sum())

# X = df.drop("price", axis=1)
# y = df["price"]

# X_one_encode = pd.get_dummies(
#     X,
#     columns = ["brand","category","processor","graphics"],
#     drop_first = True,
#     dtype = int
# )

# print(X.head())
# print(X.shape[1])

# X_train, X_test, y_train, y_test = train_test_split(
#     X_one_encode,y,
#     test_size=0.2,
#     random_state=42
# )

# model = DecisionTreeRegressor(random_state=42)
# model.fit(X_train,y_train)

# y_pred = model.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print("MAE:", round(mae, 2))
# print("MSE:", round(mse, 2))
# print("RMSE:", round(rmse, 2))
# print("R2 Score:", round(r2, 4))

# results = pd.DataFrame({
#     "Actual Price": y_test.values,
#     "Predicted Price": np.round(y_pred, 2)
# })
# print(results.head(15))


# Random Forest 


df = pd.read_csv("laptops_simple.csv")

X = df.drop("price", axis=1)
y = df["price"]

X_one_encode = pd.get_dummies(
    X,
    columns = ["brand","category","processor","graphics"],
    drop_first = True,
    dtype = int
)


X_train, X_test, y_train, y_test = train_test_split(
    X_one_encode,y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": np.round(y_pred, 2)
})
print(results.head(15))
























































































