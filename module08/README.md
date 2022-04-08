# Module 08

Questions answers:

> **Why do we use logistic hypothesis for a classification problem rather than a linear hypothesis ?**
> Logistic hypothesis is used since the value we expect is a prediction, not a "value", so the predicted value need to be in a given range to represent "yes" or "no" (is the prediction of a given class or another).

> **What is the decision boundary ?**
> It's the limits of the range for our predictions (minimum is 0 and maximum is 1), where if the prediction is closer to 0 it's classified as not of the expected class and on the other end if it's closer to 1 it's classified as of the expected class.

> **In the case we decide to use a linear hypothesis to tackle a classification problem, why the classification of some data points can be modified by considering more examples for example, extra data points with extrem ordinate) ?**
> Using a linear hypothesis for a classification problem will split the predictions in 2, and adding points in extrem ordinate will always yield incorrect predictions, since moving the split to either end for the decision boundary will make all predictions only of the given class or not of the given class. I think.

> **In a one versus all classification approach, how many logisitic regressor do we need to distinguish between N classes ?**
> We need as many logistic regressor as there is classes, since each regressor predict if given value is either of it's class or not of it's class.

> **Can you explain the difference between accuracy and precision? What is the type I and type II errors ?**
> The accuracy is the percentage of correctly predicted values, while the precision is how accurate these predictions are, meaning how much you can trust the model when it makes a prediction of the expected class.
> Type I and Type II errors are respectively the False Positive and the False Negative.
> False Positives is when a prediction that should not have been of the expected class is predicted, and False Negatives is the opposite, when a prediction that should have been of the expected class is not predicted.

> **What is the interest of the F1-score ?**
> The F1-score combine all other errors to measure both the False Positives and False Negatives.
