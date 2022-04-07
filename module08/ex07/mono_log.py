from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from data_spliter import data_spliter
from my_logistic_regression import MyLogisticRegression as MyLR

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('zipcode', help='ZIP code of the planet: 0, 1, 2 or 3', type=int)
    args = parser.parse_args()

    zipcode = args.zipcode

    # Check arguments
    if not isinstance(zipcode, int) or zipcode < 0 or zipcode > 3:
        parser.print_help()
        exit(1)

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
    y = np.where(y == zipcode, 1, 0)

    # Split dataset
    xTrain, xTest, yTrain, yTest = data_spliter(x, y, 0.8)

    # Train model
    model = MyLR([[1.], [1.], [1.], [1.]], alpha=0.0005, max_iter=200000)
    print(f"Trained:\n{model.fit_(xTrain, yTrain)}")

    # Show model results
    predictions = np.round(model.predict_(xTrain))
    correct_predictions = np.sum(predictions == yTrain)
    print(f"[Train] Correct predicted values: {correct_predictions} / {yTrain.shape[0]} ({(correct_predictions / yTrain.shape[0]) * 100:.2f}%)")
    predictions = np.round(model.predict_(xTest))
    correct_predictions = np.sum(predictions == yTest)
    print(f"[Test]  Correct predicted values: {correct_predictions} / {yTest.shape[0]}  ({(correct_predictions / yTest.shape[0]) * 100:.2f}%)")

    # Show one scatter plot for each of X = features and Y = prediction
    # TODO fix plot
    predictions = np.round(model.predict_(x))
    fig, dim_axs = plt.subplots(ncols=3)
    axs = dim_axs.flatten()
    for i, feature in enumerate(features):
        axs[i].scatter(x_df[feature], y, color="purple")
        axs[i].scatter(x_df[feature], predictions, color="pink", alpha=0.5)
        axs[i].set_xlabel(f"{feature}")
        axs[i].set_ylabel(f"Prediction (1 = zipcode)")
    plt.show()
