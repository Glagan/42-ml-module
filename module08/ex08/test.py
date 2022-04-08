import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from other_metrics import accuracy_score_, f1_score_, precision_score_, recall_score_

y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

# Example 1:
print("np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))")
print("np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))")
# Accuracy
# your implementation
print("self   ", accuracy_score_(y, y_hat))
# Output:
# 0.5
# sklearn implementation
print("sklearn", accuracy_score(y, y_hat))
# Output:
# 0.5

# Precision
# your implementation
print("self   ", precision_score_(y, y_hat))
# Output:
# 0.4
# sklearn implementation
print("sklearn", precision_score(y, y_hat))
# Output:
# 0.4

# Recall
# your implementation
print("self   ", recall_score_(y, y_hat))
# Output:
# 0.6666666666666666
# sklearn implementation
print("sklearn", recall_score(y, y_hat))
# Output:
# 0.6666666666666666

# F1-score
# your implementation
print("self   ", f1_score_(y, y_hat))
# Output:
# 0.5
# sklearn implementation
print("sklearn", f1_score(y, y_hat))
# Output:
# 0.

y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

# Example 1.1:
print("\nnp.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])")
print("np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])")
# Accuracy
# your implementation
print("self   ", accuracy_score_(y, y_hat))
# Output:
# 0.625
# sklearn implementation
print("sklearn", accuracy_score(y, y_hat))
# Output:
# 0.625
# Precision
# your implementation
print("self   ", precision_score_(y, y_hat, pos_label='dog'))
# Output:
# 0.6
# sklearn implementation
print("sklearn", precision_score(y, y_hat, pos_label='dog'))
# Output:
# 0.6
# Recall
# your implementation
print("self   ", recall_score_(y, y_hat, pos_label='dog'))
# Output:
# 0.75
# sklearn implementation
print("sklearn", recall_score(y, y_hat, pos_label='dog'))
# Output:
# 0.75
# F1-score
# your implementation
print("self   ", f1_score_(y, y_hat, pos_label='dog'))
# Output:
# 0.6666666666666665
# sklearn implementation
print("sklearn", f1_score(y, y_hat, pos_label='dog'))
# Output:
# 0.6666666666666665

y_hat = np.array([1, 0, 2, 1, 2, 0, 1, 1, 2]).reshape((-1, 1))
y = np.array([1, 0, 2, 0, 1, 2, 0, 1, 2]).reshape((-1, 1))

# Example 2:
print("\nnp.array([1, 0, 2, 1, 2, 0, 1, 1, 2]).reshape((-1, 1))")
print("np.array([1, 0, 2, 0, 1, 2, 0, 1, 2]).reshape((-1, 1))")
# Accuracy
# your implementation
print("self   ", accuracy_score_(y, y_hat))
# sklearn implementation
print("sklearn", accuracy_score(y, y_hat))

# Precision
# your implementation
print("self   ", precision_score_(y, y_hat, pos_label=2))
# sklearn implementation
print("sklearn", precision_score(y, y_hat, labels=[2], average='weighted'))

# Recall
# your implementation
print("self   ", recall_score_(y, y_hat, pos_label=2))
# sklearn implementation
print("sklearn", recall_score(y, y_hat, labels=[2], average='weighted'))

# F1-score
# your implementation
print("self   ", f1_score_(y, y_hat, pos_label=2))
# sklearn implementation
print("sklearn", f1_score(y, y_hat, labels=[2], average='weighted'))

y_hat = np.array(['a', 'b', 'c', 'a', 'c', 'b', 'a', 'a', 'c']).reshape((-1, 1))
y = np.array(['a', 'b', 'c', 'b', 'a', 'c', 'b', 'a', 'c']).reshape((-1, 1))

# Example 3:
print("\nnp.array(['a', 'b', 'c', 'a', 'c', 'b', 'a', 'a', 'c']).reshape((-1, 1))")
print("np.array(['a', 'b', 'c', 'b', 'a', 'c', 'b', 'a', 'c']).reshape((-1, 1))")
# Accuracy
# your implementation
print("self   ", accuracy_score_(y, y_hat))
# sklearn implementation
print("sklearn", accuracy_score(y, y_hat))

# Precision
# your implementation
print("self   ", precision_score_(y, y_hat, pos_label='a'))
# sklearn implementation
print("sklearn", precision_score(y, y_hat, labels=['a'], average='weighted'))

# Recall
# your implementation
print("self   ", recall_score_(y, y_hat, pos_label='a'))
# sklearn implementation
print("sklearn", recall_score(y, y_hat, labels=['a'], average='weighted'))

# F1-score
# your implementation
print("self   ", f1_score_(y, y_hat, pos_label='a'))
# sklearn implementation
print("sklearn", f1_score(y, y_hat, labels=['a'], average='weighted'))

y_hat = np.array([0, 2, 1, 3]).reshape((-1, 1))
y = np.array([0, 1, 2, 3]).reshape((-1, 1))

# Example 4:
print("\nnp.array([0, 2, 1, 3]).reshape((-1, 1))")
print("np.array([0, 1, 2, 3]).reshape((-1, 1))")
# Accuracy
# your implementation
print("self   ", accuracy_score_(y, y_hat))
# sklearn implementation
print("sklearn", accuracy_score(y, y_hat))

# Precision
# your implementation
print("self   ", precision_score_(y, y_hat, pos_label=1))
# sklearn implementation
print("sklearn", precision_score(y, y_hat, labels=[1], average='weighted'))

# Recall
# your implementation
print("self   ", recall_score_(y, y_hat, pos_label=1))
# sklearn implementation
print("sklearn", recall_score(y, y_hat, labels=[1], average='weighted'))

# F1-score
# your implementation
print("self   ", f1_score_(y, y_hat, pos_label=1))
# sklearn implementation
print("sklearn", f1_score(y, y_hat, labels=[1], average='weighted'))
