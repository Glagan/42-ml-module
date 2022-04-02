# Module 05

Questions answers:

> **Why do we concatenate a column of ones to the left of the x vector when we use the linear algebra trick ?**  
> We add a column of 1 to the left of the x vector to have the correct matrix dimension to do a multiplication with the theta vector, and it's filled with 1 so it ignores theta0.

> **Why does the loss function square the distances between the data points and their predicted values ?**  
> The distance between data points is squared so it ignores negative and positive values order.

> **What does the loss function's output represent ?**  
> The cost function output represent the "distance" between the predicted result and the real result.

> **Toward which value do we want the loss function to tend? What would that mean ?**  
> We want the cost function output to go towards 0, which would mean that all of the predicitions match perfectly the real result.

> **Do you understand why are matrix multiplications are not commutative ?**  
> Matrix multiplication are not commutative because the dimensions of the matrices could not be valid when their position in the multiplication are reversed.
