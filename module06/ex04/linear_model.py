import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR


def best_hypothesis():
    data = pd.read_csv("../resources/are_blue_pills_magics.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1, 1)
    Yscore = np.array(data['Score']).reshape(-1, 1)
    lr = MyLR([1, 0], alpha=0.01, max_iter=5000)
    theta = lr.fit_(Xpill, Yscore)
    xMin, xMax = min(Xpill), max(Xpill)
    plt.scatter(Xpill, Yscore, color='cornflowerblue')
    plt.plot([xMin, xMax], [theta[0] + (theta[1] * xMin),
                            theta[0] + (theta[1] * xMax)], color='limegreen', linestyle='dashed')
    plt.scatter(Xpill, lr.predict_(Xpill), color='limegreen')
    plt.show()


best_hypothesis()


def cost_function():
    data = pd.read_csv("../resources/are_blue_pills_magics.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1, 1)
    Yscore = np.array(data['Score']).reshape(-1, 1)
    lr = MyLR([1, 0], alpha=0.01, max_iter=5000)


cost_function()
