# module07

Questions answers:

> **What is the main (obvious) difference between univariate and multivariate linear regression ?**
> A multivariate regression has more variables.

> **Is there a minimum number of variables needed to perform a multivariate linear regression ?**
> A multivariate regression need to have at least 2 features, or else it is an univariate regression.

> **Is there a maximum number of variables needed to perform a multivariate linear regression ? In theory and in practice ?**
> There is no limit to the number of features of a multivariate regression, but a huge number of features will make a model slower to train since it will need more computation.

> **Is there a difference between univariate and multivariate linear regression in terms of performance evaluation ?**
> A multivariate regression will be slower since it has a greater Matrix size and need more operation compared to an univariate regression with the same amount of examples in a dataset.

> **What does it mean geometrically to perform a multivariate gradient descent with two variables ?**
> A multivariate gradient descent with 2 variables makes it a 3D problems, since each increase in the number of features increase the number of dimensions by one geometrically.

> **Can you explain what is overfitting ?**
> Overfitting is when a model predict a good response _too much_ for a given dataset (usually the training dataset), and fails to generalize when used on other or real world datasets.

> **Can you explain what is underfitting ?**
> Underfitting is the opposite of overfitting, it will juste have bad results and fails to predict accurately.

> **Why is it important to split the data set in a training and a test set ?**
> It's important to split the dataset to _prevent_ overfitting, using another metric to check the dataset while training it.

> **If a model overfits, what will happen when you compare its performance on the training set and the test set ?**
> The performance on the training set will be high but on the opposite it will be poor with the test set.

> **If a model underfits, what do you think will happen when you compare its performance on the training set and the test set ?**
> The performance will be poor with both sets, since the model is not trained enough to identify good results or is too simplistic.

## Resources

- Matplotlib colors
  - https://matplotlib.org/stable/gallery/color/named_colors.html
