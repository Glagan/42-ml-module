import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from polynomial_model import add_polynomial_features
from my_linear_regression import MyLinearRegression as MyLR

# Open dataset
try:
    df = pd.read_csv('../resources/are_blue_pills_magics.csv')
except BaseException as error:
    print(f"Failed to read dataset: {error}")
    exit(1)
xMicrograms = df['Micrograms'].to_numpy().reshape(-1, 1)
yScore = df['Score'].to_numpy().reshape(-1, 1)

# Base Theta
thetas = [
    np.array([[1.], [1.]]).reshape(-1, 1),
    np.array([[1.], [1.], [1.]]).reshape(-1, 1),
    np.array([[1.], [1.], [1.], [1.]]).reshape(-1, 1),
    np.array([[-20], [160], [-80], [10], [-1]]).reshape(-1, 1),
    np.array([[1140], [-1850], [1110], [-305], [40], [-2]]).reshape(-1, 1),
    np.array([[9110], [-18015], [13400], [-4935],
             [966], [-96.4], [3.86]]).reshape(-1, 1),
]

# Train models
models = []
losses = []
for i in range(6):
    model = MyLR(thetas[i], max_iter=200000)
    model.alpha = model.alpha / max(100000 * i, 1)
    x = add_polynomial_features(xMicrograms, i + 1)
    print(f"[Polynomial {i + 1}]\n{model.fit_(x, yScore)}")
    loss = model.loss_(yScore, model.predict_(x))
    print(f"[Polynomial {i + 1}] Loss: {loss}")
    losses.append(loss)
    models.append(model)

# Show MSE score for the trained models
plt.plot(list(range(1, 7)), losses)
plt.xlabel("Polynomial degree")
plt.ylabel("MSE Score")
plt.show()

# Show differences
continuous_x = np.arange(1, 6.5, 0.01).reshape(-1, 1)
cmap = plt.cm.get_cmap('jet', 7)
polynomial = 1
for model in models:
    x = add_polynomial_features(xMicrograms, polynomial)
    predictions = model.predict_(x)
    plt.scatter(xMicrograms, predictions,
                # label=f"Predictions for p={polynomial}",
                color=cmap(polynomial),
                alpha=0.75)
    x = add_polynomial_features(continuous_x, polynomial)
    plt.plot(continuous_x, model.predict_(x),
             label=f"Polynomial {polynomial} predictions",
             color=cmap(polynomial))
    polynomial = polynomial + 1
plt.scatter(df['Micrograms'], df['Score'], label="Dataset", c='blue')
plt.xlabel("Micrograms")
plt.ylabel("Score")
plt.legend()
plt.show()

"""
When looking only at the loss score, the model with the most polynomial is the best,
and the model with the second most polynomial is as good as the one with one polynomial.
When looking at the graph, we can see that the models with the most polynomials are
overfitting the dataset, and is correctly predicting the datasets expected results but
will likely not correctly predict another real row from another dataset.
"""
