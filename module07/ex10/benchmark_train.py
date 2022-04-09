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
x = df[['weight', 'prod_distance', 'time_delivery']].to_numpy().reshape(-1, 3)
y = df['target'].to_numpy().reshape(-1, 1)

# Split dataset
xTrain, xTest, yTrain, yTest = data_spliter(x, y, 0.8)

# Base Theta
thetas = [
    np.array([[1.], [1.], [2.], [3.]]).reshape(-1, 1),
    np.array([[1.], [1.], [2.], [3.], [4.]]).reshape(-1, 1),
    np.array([[1.], [1.], [2.], [3.], [4.], [5.]]).reshape(-1, 1),
    np.array([[1.], [1.], [2.], [3.], [4.], [5.], [6.]]).reshape(-1, 1),
]
max_polynomial = len(thetas)

# Train models
max_iter = 10000
models = []
models_loss = []
losses = []
for i in range(0, max_polynomial):
    print(f"[Polynomial {i + 1}] Training...")
    current_loss = []
    model = MyLR(thetas[i], alpha=0.000000000000001, max_iter=1)
    x = add_polynomial_features(xTrain, i + 1)
    for j in range(max_iter):
        model.fit_(x, yTrain)
        current_loss.append(model.loss_(yTrain, model.predict_(x)))
    x_test = add_polynomial_features(xTest, i + 1)
    loss = model.loss_(yTest, model.predict_(x_test))
    print(f"[Polynomial {i + 1}] Test loss: {loss}")
    losses.append(loss)
    models.append(model)
    models_loss.append(current_loss)

# Save trained models
np.savez("models.npz", *[model.theta for model in models])

# Show loss over time for the trained models on the test set
cmap = plt.cm.get_cmap('jet', max_polynomial + 1)
for index, model_loss in enumerate(models_loss):
    plt.plot(list(range(1, max_iter + 1)), model_loss, c=cmap(index + 1), label=f"Polynomial {index + 1}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Show test loss per models
# plt.plot(list(range(1, max_polynomial)), loss, c=cmap(index + 1), label=f"Polynomial {index + 1}")
