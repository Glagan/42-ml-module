import math
from turtle import fd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from polynomial_model import add_polynomial_features
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

# Split dataset
xTrain, xTest, yTrain, yTest = data_spliter(x, y, 0.8)

# Base Theta
thetas = [
    np.random.rand(4, 1),
    np.random.rand(7, 1),
    np.random.rand(10, 1),
    np.random.rand(13, 1),
]
max_polynomial = len(thetas)

# Train models
alpha = 0.005
max_iter = 50000
models = []
models_loss = []
losses = []
lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
for i in range(0, max_polynomial):
    for lambda_ in lambdas:
        print(f"[Polynomial {i + 1} / λ {lambda_}] Training...")
        model = MyRidge(thetas[i] + 0, alpha=alpha, max_iter=max_iter, lambda_=lambda_)
        # Normalize dataset after polynomials
        x = add_polynomial_features(xTrain, i + 1)
        norm_mean, norm_std = x.mean(axis=0), x.std(axis=0)
        x = (x - norm_mean) / norm_std
        _, loss = model.fit_(x, yTrain, check_loss=True)
        models_loss.append(loss)
        # Calculate loss against test set
        # Test set is normalized using the same values as the training set
        x_test = add_polynomial_features(xTest, i + 1)
        x_test = (x_test - norm_mean) / norm_std
        loss = model.loss_(yTest, model.predict_(x_test))
        print(f"[Polynomial {i + 1} / λ {lambda_}] Test loss: {loss}")
        losses.append(loss)
        models.append(model)

# Save trained models
# -- with the alpha and iterations
# -- and the initial thetas for the best model
np.savez("models.npz",
         # A lambda_ value of 0.6 is good without increasing loss too much
         np.array([alpha, max_iter, 4, 0.6]),
         # The model with 4 polynomial is the best
         thetas[3],
         *[model.thetas for model in models])

# Show loss over time for the trained models on the test set
cmap = plt.cm.get_cmap('jet', len(models) + 1)
j = 0
for index, model_loss in enumerate(models_loss):
    polynomial = math.floor(index / 6) + 1
    lambda_ = lambdas[j % len(lambdas)]
    plt.plot(list(range(1, max_iter + 1)), model_loss, c=cmap(index + 1), label=f"Polynomial {polynomial} / λ {lambda_}")
    j += 1
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
