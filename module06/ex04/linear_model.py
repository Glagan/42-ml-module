from re import M
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from my_linear_regression import MyLinearRegression as MyLR


def best_hypothesis(Xpill: np.ndarray, Yscore: np.ndarray) -> MyLR:
    # Analyze
    model = MyLR([[89.0], [-8]], alpha=0.000001, max_iter=100)
    theta = model.fit_(Xpill, Yscore)
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
                model.predict_(Xpill),
                color='limegreen',
                marker='x',
                label="$S_{predict(pills)}$")
    plt.xlabel("Quantity of blue pill (in micrograms)")
    plt.ylabel("Space driving score")
    plt.legend(loc='upper right')
    plt.show()
    return model


def show_losses(Xpill: np.ndarray, Yscore: np.ndarray) -> None:
    # Generate some random thetas close from the fitted one
    # [[ 74.], [ 80.], [ 86.], [ 92.], [ 98.], [104.]]
    theta0_list = np.linspace(74, 104, 6).reshape(-1, 1)
    n = len(theta0_list)
    # Use a continuous value to
    continuous_theta1 = np.arange(-18, -2, 0.01).reshape(-1, 1)
    # Generate gray colors
    colors = pl.cm.Greys(np.linspace(0, 1, n + 1))
    for index, theta0 in enumerate(theta0_list):
        loss = []
        model = MyLR([theta0, [0.]])
        for theta1 in continuous_theta1:
            model.thetas[1] = theta1
            loss.append(model.loss_(Yscore, model.predict_(Xpill)))
        plt.plot(continuous_theta1,
                 loss,
                 label="J(${{\Theta_0}}$=$c_{}$, ${{\Theta_1}}$)".format(index),
                 color=colors[index + 1])  # Skip first color
    plt.grid()
    plt.xlabel("${\Theta_1}$")
    plt.ylabel("Cost function J(${\Theta_0}$, ${\Theta_1}$)")
    plt.legend(loc='lower right')
    # Set zoom
    plt.xlim([-14.5, -3.5])
    plt.ylim([10, 150])
    plt.show()


if __name__ == "__main__":
    # Read dataset
    data = pd.read_csv("../resources/are_blue_pills_magics.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1, 1)
    Yscore = np.array(data['Score']).reshape(-1, 1)
    # Train best model and compare to the true results
    model = best_hypothesis(Xpill, Yscore)
    # Show the loss function for different thetas combinations
    show_losses(Xpill, Yscore)
    # Show the MSE from the previously trained best model
    print(f"MSE: {model.mse_(model.predict_(Xpill), Yscore)}")
