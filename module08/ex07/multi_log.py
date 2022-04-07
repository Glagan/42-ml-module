from argparse import ArgumentParser
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
        print(
            f"[Test]  Correct predicted values: {correct_predictions} / {model_yTest.shape[0]}  ({(correct_predictions / model_yTest.shape[0]) * 100:.2f}%)")

    # Predict the whole dataset for each models 4 * [len, 1]
    # -- and select the highest value and assign the planet to it
    multiclass_predictions = np.array([model.predict_(x) for model in models])
    predictions = np.argmax(multiclass_predictions, axis=0)
    correct_predictions = np.sum(predictions == y)
    print(f"[Overall] Correct predicted values: {correct_predictions} / {y.shape[0]} ({(correct_predictions / y.shape[0]) * 100:.2f}%)")

    # Show one scatter plot for each of X = features and Y = prediction
    # TODO fix plot
    fig, dim_axs = plt.subplots(ncols=3)
    axs = dim_axs.flatten()
    for i, feature in enumerate(features):
        axs[i].scatter(x_df[feature], y, color="purple")
        axs[i].scatter(x_df[feature], predictions, color="pink", alpha=0.5)
        axs[i].set_xlabel(f"{feature}")
        axs[i].set_ylabel(f"Prediction (1 = zipcode)")
    plt.show()
