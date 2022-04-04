import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from my_linear_regression import MyLinearRegression as MyLR


def best_hypothesis():
    # Read dataset
    data = pd.read_csv("../resources/are_blue_pills_magics.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1, 1)
    Yscore = np.array(data['Score']).reshape(-1, 1)
    # Analyze
    lr = MyLR([[89.0], [-8]], alpha=0.01, max_iter=5000)
    theta = lr.fit_(Xpill, Yscore)
    xMin, xMax = min(Xpill), max(Xpill)
    plt.scatter(Xpill,
                Yscore,
                color='cornflowerblue',
                label="$S_{true(pills)}$")
    plt.plot([xMin, xMax],
             [theta[0] + (theta[1] * xMin),
              theta[0] + (theta[1] * xMax)],
             color='limegreen',
             linestyle='dashed')
    plt.scatter(Xpill,
                lr.predict_(Xpill),
                color='limegreen',
                label="$S_{predict(pills)}$")
    plt.xlabel("Quantity of blue pill (in micrograms)")
    plt.ylabel("Space driving score")
    plt.legend(loc='upper right')
    plt.show()


def loss_function():
    # Read dataset
    data = pd.read_csv("../resources/are_blue_pills_magics.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1, 1)
    Yscore = np.array(data['Score']).reshape(-1, 1)
    # Generate gray colors
    n = 6
    colors = pl.cm.Greys(np.linspace(0, 1, n + 1))
    # TODO Plot over WHAT ?
    # TODO Select `n` thetas (as theta0)
    # TODO plot them for theta1 from `x1` (e.g -20) to `x2` (e.g 20)
    # Analyze X times
    lr = MyLR([[89.0], [-8]], alpha=0.01, max_iter=1)
    for i in range(n):
        original_theta = lr.thetas[1]
        theta = lr.fit_(Xpill, Yscore)
        theta[1] = original_theta
        lr.thetas = theta
        plt.plot(Xpill,
                 lr.loss_elem_(lr.predict_(Xpill), Yscore),
                 label="J(${{\Theta_0}}$ = $c_{}$, c${{\Theta_1}}$)".format(i),
                 color=colors[i + 1])  # Skip first color
    plt.xlabel("${\Theta_1}$")
    plt.ylabel("Cost function J(${\Theta_0}$, ${\Theta_1}$)")
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    best_hypothesis()
    loss_function()
