from argparse import ArgumentParser
from itertools import combinations
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from data_spliter import data_spliter
from my_logistic_regression import MyLogisticRegression as MyLR

if __name__ == '__main__':
    # Open datasets
    try:
        x_df = pd.read_csv('../resources/solar_system_census.csv')
    except BaseException as error:
        print(f"Invalid dataset: {error}")
        exit(1)
    try:
        y_df = pd.read_csv('../resources/solar_system_census_planets.csv')
    except BaseException as error:
        print(f"Invalid dataset: {error}")
        exit(1)
    features = ['weight', 'height', 'bone_density']
    x = x_df[features].to_numpy().reshape((-1, 3))
    y = y_df['Origin'].to_numpy().reshape((-1, 1))

    # Split dataset
    xTrain, xTest, yTrain, yTest = data_spliter(x, y, 0.8)

    # Train models
    models = []
    for i in range(4):
        model_yTrain = np.where(yTrain == i, 1, 0)
        model = MyLR([[1.], [1.], [1.], [1.]], alpha=0.0005, max_iter=200000)
        print(f"Trained model for {i}:\n{model.fit_(xTrain, model_yTrain)}")
        models.append(model)

        # Show current model results
        predictions = np.round(model.predict_(xTrain))
        correct_predictions = np.sum(predictions == model_yTrain)
        print(
            f"[Train] Correct predicted values: {correct_predictions} / {model_yTrain.shape[0]} ({(correct_predictions / model_yTrain.shape[0]) * 100:.2f}%)")
        model_yTest = yTest.copy()
        model_yTest[model_yTest == i] = 1
        model_yTest[model_yTest != 1] = 0
        predictions = np.round(model.predict_(xTest))
        correct_predictions = np.sum(predictions == model_yTest)
        print(f"[Test]  Correct predicted values: {correct_predictions} / {model_yTest.shape[0]}  ({(correct_predictions / model_yTest.shape[0]) * 100:.2f}%)")

    # Predict the whole dataset for each models 4 * [len, 1]
    # -- and select the highest value and assign the planet to it
    multiclass_predictions = np.array([model.predict_(x) for model in models]).T.reshape(-1, 4)
    predictions = np.argmax(multiclass_predictions,  axis=1).reshape(-1, 1)
    correct_predictions = np.sum(predictions == y)
    print(f"[Overall] Correct predicted values: {correct_predictions} / {y.shape[0]} ({(correct_predictions / y.shape[0]) * 100:.2f}%)")

    # Show one scatter plot for each of X, Y = features and color = prediction
    features_pair = combinations(features, 2)
    fig, dim_axs = plt.subplots(ncols=3)
    cmap = plt.cm.get_cmap('jet', 5)
    planets = ["The flying cities of Venus", "United Nations of Earth", "Mars Republic", "The Asteroids' Belt colonies"]
    axs = dim_axs.flatten()
    for i, (feature1, feature2) in enumerate(features_pair):
        # Scatter each planets with it's own color
        for p in range(4):
            axs[i].scatter(x_df[feature1].loc[(predictions == p)], x_df[feature2].loc[(predictions == p)], color=cmap(p), label=f"{planets[p]}")
        axs[i].set_xlabel(f"{feature1}")
        axs[i].set_ylabel(f"{feature2}")
    plt.legend()
    plt.show()
