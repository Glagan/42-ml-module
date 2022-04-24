import math
from turtle import fd
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

# Split dataset in 3: train, cross-validation and test
split_size = [int(.8 * len(df)), int(.9 * len(df))]
train, validate, test = np.split(df.sample(frac=1), split_size)
xTrain = train[features].to_numpy().reshape((-1, 3))
yTrain = train['target'].to_numpy().reshape((-1, 1))
xValidation = validate[features].to_numpy().reshape((-1, 3))
yValidation = validate['target'].to_numpy().reshape((-1, 1))
xTest = test[features].to_numpy().reshape((-1, 3))
yTest = test['target'].to_numpy().reshape((-1, 1))

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
        # Calculate loss against validation set
        # Validation set is normalized using the same values as the training set
        x_validate = add_polynomial_features(xValidation, i + 1)
        x_validate = (x_validate - norm_mean) / norm_std
        loss = model.loss_(yValidation, model.predict_(x_validate))
        print(f"[Polynomial {i + 1} / λ {lambda_}] Validation loss: {loss}")
        losses.append(loss)
        models.append(model)

# Calculate loss against test set for the best model
# Test set is normalized using the same values as the training set
best_model = models[(3 * 6) + 3]
x_test = add_polynomial_features(xTest, i + 1)
x_test = (x_test - norm_mean) / norm_std
loss = model.loss_(yTest, model.predict_(x_test))
print(f"Best model test loss: {loss}")

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
