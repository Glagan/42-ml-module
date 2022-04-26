import pickle
import numpy as np
import pandas as pd
from my_logistic_regression import MyLogisticRegression
from polynomial_model_extended import add_polynomial_features
from scores import *


def initial_theta():
    return np.array([[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]])


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

# Split dataset in 3: train, cross-validation and test
split_size = [int(.8 * len(df)), int(.9 * len(df))]
train, validate, test = np.split(df.sample(frac=1), split_size)
xTrain = train[features].to_numpy().reshape((-1, 3))
yTrain = train['Origin'].to_numpy().reshape((-1, 1))
xValidation = validate[features].to_numpy().reshape((-1, 3))
yValidation = validate['Origin'].to_numpy().reshape((-1, 1))
xTest = test[features].to_numpy().reshape((-1, 3))
yTest = test['Origin'].to_numpy().reshape((-1, 1))

# Lambdas to use
lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
alpha = 0.0005
max_iter = 25000

# Train each models with a polynomial of 3
xTrain_poly = add_polynomial_features(xTrain, 3)
xTest_poly = add_polynomial_features(xTest, 3)
xValidation_poly = add_polynomial_features(xValidation, 3)
models = []
for lambda_ in lambdas:
    current_models = []
    # Train each models with an One-vs-All method for each planets
    for i in range(4):
        yTrain_one = np.where(yTrain == i, 1, 0)
        model = MyLogisticRegression(initial_theta(), alpha=alpha, max_iter=max_iter, lambda_=lambda_)
        print(f"Training with lambda={lambda_} (p={i})")
        model.fit_(xTrain_poly, yTrain_one)
        # Show f1 score against cross validation set for each models
        # yValidation_hat = np.round(model.predict_(xValidation_poly))
        # yValidation_one = np.where(yValidation == i, 1, 0)
        # print(f"f1 score: {f1_score_(yValidation_one, yValidation_hat, pos_label=i)}")
        current_models.append(model)
    # Predict the values for the whole set with this lambda
    multiclass_predictions = np.array([model.predict_(xTest_poly) for model in current_models]).T.reshape(-1, 4)
    yTest_hat = np.argmax(multiclass_predictions, axis=1).reshape(-1, 1)
    # Show correctly predicted values
    correct_predictions = np.sum(yTest_hat == yTest)
    print(f"Correct predicted values: {correct_predictions} / {yTest.shape[0]} ({(correct_predictions / yTest.shape[0]) * 100:.2f}%)")
    print(f"Overall f1 score: {f1_score_(yTest, yTest_hat)}")
    models.append(current_models)
    print("")

# Save models
try:
    file = open("models.pickle", mode='wb')
    pickle.dump([
        [alpha, max_iter, 0.6],
        initial_theta(),
        models
    ], file=file)
except BaseException as error:
    print(f"Failed to save trained models: {error}")
