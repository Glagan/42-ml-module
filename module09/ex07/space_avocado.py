import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from polynomial_model_extended import add_polynomial_features
from ridge import MyRidge
from data_spliter import data_spliter

# Open dataset
try:
    df = pd.read_csv('../resources/space_avocado.csv')
except BaseException as error:
    print(f"Failed to read dataset: {error}")
    exit(1)
features = ['weight', 'prod_distance', 'time_delivery']
x = df[features].to_numpy().reshape(-1, 3)
y = df['target'].to_numpy().reshape(-1, 1)
lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Load trained models and the best model initial thetas
try:
    models = np.load('models.npz')
    if len(models) != 26:
        raise ValueError("missing value in saved models")
    alpha, max_iter, polynomials, lambda_ = [value.item() for value in models["arr_0"]]
    print(alpha, max_iter, polynomials, lambda_)
    best_model_theta = models["arr_1"]
    other_models = [models[f"arr_{index}"] for index in range(2, 26)]
except BaseException as error:
    print(f"Failed to read dataset: {error}")
    exit(1)

# Split dataset
xTrain, xTest, yTrain, yTest = data_spliter(x, y, 0.8)

# Normalize
x_pol = add_polynomial_features(xTrain, int(polynomials))
norm_mean, norm_std = x_pol.mean(axis=0), x_pol.std(axis=0)
x_pol = (x_pol - norm_mean) / norm_std
# Train model
model = MyRidge(best_model_theta, alpha=alpha, max_iter=int(max_iter), lambda_=lambda_)
print("Training best model...")
model.fit_(x_pol, yTrain)
# Calculate loss against test set
x_pred = add_polynomial_features(xTest, int(polynomials))
x_pred = (x_pred - norm_mean) / norm_std
loss = model.loss_(yTest, model.predict_(x_pred))
print(f"Loss: {loss}")


def init_plot(title: str):
    fig, dim_axs = plt.subplots(ncols=3)
    axs = dim_axs.flatten()
    for i, feature in enumerate(features):
        # First scatter: other planets -- select X for each features where prediction is 0
        axs[i].scatter(df[feature], y, color="purple", s=10, label="Dataset")
        axs[i].set_xlabel(f"{feature}")
        axs[i].set_ylabel(f"Target price")
    plt.title(title)
    plt.legend()
    return axs


def fill_plot(axs, model: MyRidge, polynomial: int, color: str) -> None:
    x_pol = add_polynomial_features(x, polynomial)
    x_pol = (x_pol - x_pol.mean(axis=0)) / x_pol.std(axis=0)
    predictions = model.predict_(x_pol)
    for i, feature in enumerate(features):
        # Second scatter for each models
        axs[i].scatter(df[feature], predictions, color=color, alpha=0.5, s=8, label=f"Prediction")
        axs[i].set_xlabel(f"{feature}")
        axs[i].set_ylabel(f"Target price")


# Show one scatter plot for each features against the price and color = prediction
axs = init_plot("Best model")
fill_plot(axs, model, int(polynomials), "pink")
plt.show()
# -- for each saved models
cmap = plt.cm.get_cmap('jet', 7)
for polynomial in range(0, 4):
    # Skip the best model
    if polynomial + 1 == polynomials:
        continue
    axs = init_plot(f"Polynomial {polynomial + 1}")
    for index, lambda_ in enumerate(lambdas):
        other_model = MyRidge(other_models[(polynomial * 6) + index])
        fill_plot(axs, model, int(polynomials), cmap(index + 1))
    plt.show()
