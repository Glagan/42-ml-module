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

# TODO load models.npz or exit
# TODO Select the best initial model (The first one)
# TODO Train the best initial model
# TODO Plot the true price and predicted price ?? What are the axis ?

# Split dataset
xTrain, xTest, yTrain, yTest = data_spliter(x, y, 0.8)

# Train model
thetas = np.array([[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]]).reshape(-1, 1)
model = MyLR(thetas, alpha=0.000000000000000001, max_iter=1000000)
print("Training model...")
x_pol = add_polynomial_features(xTrain, 5)
model.fit_(x_pol, yTrain)
x_pred = add_polynomial_features(x, 5)
predictions = model.predict_(x_pred)
loss = model.loss_(y, predictions)
print(f"Loss: {loss}")

# Show one scatter plot for each features against the price and color = prediction
fig, dim_axs = plt.subplots(ncols=3)
cmap = plt.cm.get_cmap('jet', 3)
axs = dim_axs.flatten()
for i, feature in enumerate(features):
    # First scatter: other planets -- select X for each features where prediction is 0
    axs[i].scatter(df[feature], y, color="purple", label="Dataset")
    # Second scatter: the trained planet -- select X for each features where prediction is 1
    axs[i].scatter(df[feature], predictions, color="pink", alpha=0.5, label=f"Prediction")
    axs[i].set_xlabel(f"{feature}")
    axs[i].set_ylabel(f"Target price")
plt.legend()
plt.show()
