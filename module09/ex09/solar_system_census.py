from itertools import combinations
from pickle import Unpickler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_logistic_regression import MyLogisticRegression
from polynomial_model_extended import add_polynomial_features
from scores import *
from data_spliter import data_spliter


# Open datasets
try:
    x_df = pd.read_csv('../resources/solar_system_census.csv')
    # Normalize
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
df = pd.concat((x_df[features], y_df['Origin']), axis=1)

# Split dataset in 2: train and test
x = df[features].to_numpy().reshape(-1, 3)
y = df['Origin'].to_numpy().reshape(-1, 1)
xTrain, xTest, yTrain, yTest = data_spliter(x, y, 0.8)
xPoly = add_polynomial_features(x, 3)
xTrain_poly = add_polynomial_features(xTrain, 3)
xTest_poly = add_polynomial_features(xTest, 3)

# Load saved models
try:
    file = open('models.pickle', mode='rb')
    saved_training = Unpickler(file).load()
except BaseException as err:
    print(f"Failed to open saved models: {err}")
    exit(1)

# Check validity
if not isinstance(saved_training, list) or len(saved_training) != 3:
    print("The saved models should be a valid list")

# Display plots
try:
    alpha, max_iter, lambda_ = saved_training[0]
    best_thetas = saved_training[1]
    trained_models = saved_training[2]

    # Train the best model
    best_model = []
    print("Training best model...")
    for i in range(4):
        yTrain_one = np.where(yTrain == i, 1, 0)
        model = MyLogisticRegression(best_thetas + 0, alpha=alpha, max_iter=max_iter, lambda_=lambda_)
        model.fit_(xTrain_poly, yTrain_one)
        best_model.append(model)

    f1_scores = []
    # Calculate f1 score for the best model
    multiclass_predictions = np.array([model.predict_(xTest_poly) for model in best_model]).T.reshape(-1, 4)
    yTest_hat = np.argmax(multiclass_predictions, axis=1).reshape(-1, 1)
    best_model_f1_score = f1_score_(yTest, yTest_hat)
    print(f"Best model f1 score: {best_model_f1_score}")
    f1_scores.append(best_model_f1_score)

    # Calculate f1 score for all other models
    for trained_model in trained_models:
        multiclass_predictions = np.array([model.predict_(xTest_poly) for model in trained_model]).T.reshape(-1, 4)
        yTest_hat = np.argmax(multiclass_predictions, axis=1).reshape(-1, 1)
        model_f1_score = f1_score_(yTest, yTest_hat)
        f1_scores.append(model_f1_score)

    # Bar plot (f1 score) of all models
    bars = (lambda_, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, f1_scores)
    plt.xticks(y_pos, bars)
    plt.show()

    # Predict the whole dataset for the plot
    multiclass_predictions = np.array([model.predict_(xPoly) for model in best_model]).T.reshape(-1, 4)
    y_hat = np.argmax(multiclass_predictions, axis=1).reshape(-1, 1)
    # Show one scatter plot for each of X, Y = features and color = prediction
    features_pair = combinations(features, 2)
    fig, dim_axs = plt.subplots(ncols=3)
    cmap = plt.cm.get_cmap('jet', 5)
    planets = ["The flying cities of Venus", "United Nations of Earth", "Mars Republic", "The Asteroids' Belt colonies"]
    axs = dim_axs.flatten()
    for i, (feature1, feature2) in enumerate(features_pair):
        # Scatter each planets with it's own color
        for p in range(4):
            axs[i].scatter(x_df[feature1].loc[(y == p)], x_df[feature2].loc[(y == p)], color=cmap(p), label=f"{planets[p]}")
        # Scatter predictions after -- to show them above the real values
        for p in range(4):
            axs[i].scatter(x_df[feature1].loc[(y_hat == p)], x_df[feature2].loc[(y_hat == p)], color=cmap(p), marker='x', label=f"Predicted {planets[p]}")
        axs[i].set_xlabel(f"{feature1}")
        axs[i].set_ylabel(f"{feature2}")
    plt.legend()
    plt.show()
except BaseException as error:
    print(f"Invalid saved models {error}")
    exit(1)
