import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from polynomial_model import add_polynomial_features
from my_linear_regression import MyLinearRegression as MyLR
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

# Load trained models and the best model initial thetas
try:
    models = np.load('models.npz')
    if len(models) != 6:
        raise ValueError("missing value in saved models")
    alpha, max_iter, polynomials = [value.item() for value in models["arr_0"]]
    print(alpha, max_iter, polynomials)
    best_model_theta = models["arr_1"]
    other_models = [
        models["arr_2"],
        models["arr_3"],
        models["arr_4"],
        models["arr_5"]
    ]
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
model = MyLR(best_model_theta, alpha=alpha, max_iter=int(max_iter))
print("Training best model...")
model.fit_(x_pol, yTrain)
# Calculate loss against test set
x_pred = add_polynomial_features(xTest, int(polynomials))
x_pred = (x_pred - norm_mean) / norm_std
loss = model.loss_(yTest, model.predict_(x_pred))
print(f"Loss: {loss}")


def show_model(model: MyLR, polynomial: int, title: str) -> None:
    fig, dim_axs = plt.subplots(ncols=3)
    axs = dim_axs.flatten()
    x_pol = add_polynomial_features(x, polynomial)
    x_pol = (x_pol - x_pol.mean(axis=0)) / x_pol.std(axis=0)
    predictions = model.predict_(x_pol)
    for i, feature in enumerate(features):
        # First scatter: other planets -- select X for each features where prediction is 0
        axs[i].scatter(df[feature], y, color="purple", label="Dataset")
        # Second scatter: the trained planet -- select X for each features where prediction is 1
        axs[i].scatter(df[feature], predictions, color="pink", alpha=0.5, label=f"Prediction")
        axs[i].set_xlabel(f"{feature}")
        axs[i].set_ylabel(f"Target price")
    plt.title(title)
    plt.legend()
    plt.show()


# Show one scatter plot for each features against the price and color = prediction
show_model(model, int(polynomials), "Best model")
# -- for each saved models
for index, thetas in enumerate(other_models):
    # Skip the best model
    if index + 1 == polynomials:
        continue
    other_model = MyLR(thetas)
    show_model(other_model, index + 1, f"Polynomial {index + 1}")
