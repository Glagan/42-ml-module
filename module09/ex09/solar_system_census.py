import base64

# TODO Train the first model (best one)
# TODO Bar plot (score) of the models with their lambda value (What is score ?)
# TODO Show F1 score of all of the models (same graph ? 2 columns per lambda values ?)
# TODO Show 3D scatterplot (x, y, z) features and 2 colors -> predictions + real values
# TODO 2 3D scatterplots ? Since each planets also need their own colors ?

from itertools import combinations
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from my_logistic_regression import MyLogisticRegression
from polynomial_model_extended import add_polynomial_features
from scores import *


use_polynomial = True


def initial_theta():
    if use_polynomial:
        return np.array([[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]])
    return np.array([[1.], [1.], [1.], [1.]])


if __name__ == '__main__':
    # Open datasets
    try:
        x_df = pd.read_csv('../resources/solar_system_census.csv')
        # Normalize
        if use_polynomial:
            x_df = (x_df - x_df.mean()) / x_df.std()
    except BaseException as error:
        print(f"Invalid dataset: {error}")
        exit(1)
    try:
        y_df = pd.read_csv('../resources/solar_system_census_planets.csv')
    except BaseException as error:
        print(f"Invalid dataset: {error}")
        exit(1)
    # Merge both datasets to split once
    features = ['weight', 'height', 'bone_density']
    dataset = pd.concat((x_df[features], y_df['Origin']), axis=1)

    # Split dataset in 2: train and test
    train,  test = np.split(dataset.sample(frac=1), [int(.8 * len(dataset))])
    xTrain = train[features].to_numpy().reshape((-1, 3))
    yTrain = train['Origin'].to_numpy().reshape((-1, 1))
    xTest = test[features].to_numpy().reshape((-1, 3))
    yTest = test['Origin'].to_numpy().reshape((-1, 1))

    # Load saved models
    try:
        models_df = pd.read_csv('models.csv')
    except BaseException as err:
        print(f"Failed to open saved models: {err}")
    print(models_df)
    if models_df.shape != (7, 8):
        print("Invalid dataset")

    # Extract models from the DataFrame
    models = []
    for index, row in models_df.iterrows():
        print(base64.b64decode(row['0']))
        models.append([
            MyLogisticRegression(np.frombuffer(base64.b64decode(row['0'])), alpha=row['alpha'], lambda_=row['lambda_'], max_iter=row['max_iter']),
            MyLogisticRegression(np.frombuffer(base64.b64decode(row['1'])), alpha=row['alpha'], lambda_=row['lambda_'], max_iter=row['max_iter']),
            MyLogisticRegression(np.frombuffer(base64.b64decode(row['2'])), alpha=row['alpha'], lambda_=row['lambda_'], max_iter=row['max_iter']),
            MyLogisticRegression(np.frombuffer(base64.b64decode(row['3'])), alpha=row['alpha'], lambda_=row['lambda_'], max_iter=row['max_iter']),
        ])
    print(models)
