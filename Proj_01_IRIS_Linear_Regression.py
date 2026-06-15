import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

irisset = datasets.load_iris()

X = irisset.data[:50, 0:1]
y = irisset.data[:50, 1]

reg = LinearRegression().fit(X, y)

yPredict = reg.predict(X)

mse = mean_squared_error(y, yPredict)
r2 = r2_score(y, yPredict)

print("Train MSE =", mse)
print("Train R2 score =", r2)

plt.figure()

plt.scatter(y, yPredict, color='blue', alpha=0.6)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    color='red',
    linestyle='--'
)

plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Sepal Width")
plt.ylabel("Predicted Sepal Width")

plt.grid()
plt.show()

