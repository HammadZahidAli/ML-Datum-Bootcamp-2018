### Coding Assignment - 2 - Implementation of Logistic Regression using Gradient Descent

Hi Guys,

In this coding assignment you need to use the dataset provided to you in dataset folder and implement the logistic regression model using gradient descent technique.

#### Problem:
The data you are given is the famous iris dataset.

The dataset is about a “iris” plant. There are multiple types of “iris” plants. This dataset contains data about three types of classes of “iris” plant. Those 3 classes are:

* Iris Setosa
* Iris Versicolour
* Iris Virginica


**NOTE: Setosa class is linearly separable from the other two classes. Versicolour and Virginica are NOT linearly separable.**

Our objective is to separate Setosa type of iris flowers, from other type of flowers. Thus this problem becomes a binary classification problem as we can label Versicolour and Virginica type of iris flowers as “Not Setosa”.

As setosa is linearly separable from other two classes, which means the decision boundary between setosa and other classes will be a straight line which means we can apply the logistic regression method to separate setosa from other classes.

NOTE: Logistic Regression can be used to separate classes which are NOT linearly separated BUT we still need to cover that topic.

#### Dataset:
The data set contains 3 classes of 50 rows each,where each class refers to a type of iris plant.  
For detailed information on the dataset and attributes you can visit the following link:
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names


#### Specific Instructions:
You need to use following attributes as the input variables (X): 

* 1st column) sepal length in cm  
* 2nd column) sepal width in cm
* 3rd column) petal length in cm
* 4th column) petal width in cm

**However, if you feel that omitting any of these variables will get you better results, feel free to experiment. :)**

In terms of your response variable (Y):

* You need to change the response variable (class) for the Versicolour and Virginica category to “Not Setosa”.
* After that you must assign the following number encoding to your response variable:
1 - Setosa
0 - Not Setosa

#### Submission Guidelines: 
You need to use the given directory structure and put the code you develop in the code folder. **You need to upload the final version to your github repository and share the url of your repository with me on facebook group for feedback and evaluation.**

#### Related Articles and documents:
For gradient descent and logistic regression, you can refer to our sessions/notes as it contains all the necessary information for you to develop this implementation. Also you can refer to the code of gradient descent which we used for implementing linear regression using gradient descent. In terms of the changes required for logistic regression we have already discussed the generalized formula for gradient descent in logistic regression.

You can find detailed explanation of how MLE (Maximum likelihood estimation) is used to derive the cost function which we used for estimation of our beta parameters using gradient descent in logistic regression in the below given links to stackoverflow questions. This is a detailed derivation and requires intermediate knowledge of statistics. 

1. [Deriving Logistic Regression Cost Function using Maximum Likelihood Estimation Principle.](https://math.stackexchange.com/a/2076936)
2. [How to calculate gradient using cost function](https://math.stackexchange.com/a/2597378)




