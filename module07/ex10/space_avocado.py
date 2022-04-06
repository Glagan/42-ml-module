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
    np.array([[1.], [1.], [1.], [1.]]).reshape(-1, 1),
    np.array([[1.], [1.], [1.], [1.], [1.]]).reshape(-1, 1),
    np.array([[1.], [1.], [1.], [1.], [1.], [1.]]).reshape(-1, 1),
    np.array([[1.], [1.], [1.], [1.], [1.], [1.], [1.]]).reshape(-1, 1),
]
max_polynomial = len(thetas)

# Train models
models = []
losses = []
for i in range(max_polynomial):
    model = MyLR(thetas[i], max_iter=100000)
    model.alpha = model.alpha / max(100000 * (i + 1), 1)
    x = add_polynomial_features(xTrain, i + 1)
    # print(f"[Polynomial {i + 1}]\n{model.fit_(x, yTrain)}")
    loss = model.loss_(yTrain, model.predict_(x))
    print(f"[Polynomial {i + 1}] Loss: {loss}")
    losses.append(loss)
    models.append(model)

# Show MSE score for the trained models
plt.plot(list(range(1, max_polynomial + 1)), losses)
plt.xlabel("Polynomial degree")
plt.ylabel("MSE Score")
plt.show()

# Show differences
# TODO
# ax = plt.axes()
# cmap = plt.cm.get_cmap('jet', max_polynomial + 1)
# polynomial = 1
# for model in models:
#     x = add_polynomial_features(xTest, polynomial)
#     predictions = model.predict_(x)
#     print(
#         'x', len(list(range(xTest.shape[0]))), 'y_hat', predictions.shape)
#     ax.scatter(list(range(xTest.shape[0])), predictions,
#                label=f"Predictions for p={polynomial}",
#                color=cmap(polynomial),
#                alpha=0.75)
#     polynomial = polynomial + 1
# plt.scatter(list(range(xTest.shape[0])), yTest, label="Dataset", c='blue')
# plt.legend()
# plt.show()
