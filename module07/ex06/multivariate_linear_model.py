import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR


def plot_univariate(col1: str, col2: str, alpha: float, max_iter: int, colors: tuple) -> None:
    data = pd.read_csv("../resources/spacecraft_data.csv")
    X = data[col1].to_numpy().reshape(-1, 1)
    Y = data[col2].to_numpy().reshape(-1, 1)
    lr = MyLR(np.array([[1], [1]]), alpha=alpha, max_iter=max_iter)
    print(lr.fit_(X, Y))
    print(lr.loss_(lr.predict_(X), Y))
    plt.scatter(X, Y, color=colors[0], label=col2)
    plt.scatter(X, lr.predict_(X), s=6,
                color=colors[1],
                label=f"Predicted {col2}")
    plt.xlabel(f"$x_1$: {col1}")
    plt.ylabel(f"$y$: {col2}")
    plt.legend()
    plt.show()


print('Univariate')
plot_univariate('Age',
                'Sell_price',
                0.005,
                200000,
                ('darkblue', 'cornflowerblue'))
plot_univariate('Thrust_power',
                'Sell_price',
                0.00005,
                100000,
                ('green', 'lime'))
plot_univariate('Terameters',
                'Sell_price',
                0.00005,
                200000,
                ('purple', 'violet'))


# ---


def plot_multivariate(data: np.ndarray, col1: str, col2: str, Y: np.ndarray, Y_hat: np.ndarray, colors: tuple) -> None:
    plt.scatter(data[col1], Y, color=colors[0], label=col2)
    plt.scatter(data[col1], Y_hat, s=6, color=colors[1],
                label=f"Predicted {col2}")
    plt.xlabel(f"$x_1$: {col1}")
    plt.ylabel(f"$y$: {col2}")
    plt.legend()
    plt.show()


print('Multivariate')
data = pd.read_csv("../resources/spacecraft_data.csv")
X = data[['Age', 'Thrust_power', 'Terameters']].to_numpy().reshape(-1, 3)
Y = data['Sell_price'].to_numpy().reshape(-1, 1)
lr = MyLR(np.array([[1], [1], [1], [1]]), alpha=0.00001, max_iter=600000)
print(lr.fit_(X, Y))
print(lr.loss_(lr.predict_(X), Y))
i = 0
for according_to, colors in zip(['Age', 'Thrust_power', 'Terameters'], [('darkblue', 'cornflowerblue'), ('green', 'lime'), ('purple', 'violet')]):
    print(according_to, colors)
    plot_multivariate(data,
                      according_to,
                      'Sell_price',
                      Y,
                      lr.predict_(X),
                      colors)
    i += 1
