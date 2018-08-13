
# coding: utf-8

# In[1]:


# Agenda of the exericse
# To get you introduced to Pandas and sklearn, two of the most heavily used libraries in python for machine learning
# and data analysis.

# Pandas library helps us in easy handling and manipulation of data
# You can refer to the below given link for basic tutorial on getting started with pandas 
# http://wavedatalab.github.io/datawithpython/ 
import pandas as pd

# Sklearn library helps us in easily applying already implemented machine learning algorithms to our dataset.
# Sklearn library has a huge number of modules and in this example we are going to use only naive_bayes for 
# NaiveBayes algorithm, neighbors for KNN algorithm and metrics for evaluation metrics.
# To learn about all other modules present in sklearn module you can refer to the below given link:
# http://scikit-learn.org/stable/modules/classes.html
from sklearn.naive_bayes import GaussianNB as NaiveBayes
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


# In[2]:


# To get data loaded into memmory we can use function "read_csv" from pandas. As our dataset is of format "csv", that's why
# we used read_csv function. There are other functions available for other formats as well. 
# Here's an extensive list of all input output functions pandas provide - 
# https://pandas.pydata.org/pandas-docs/stable/io.html

data = pd.read_csv("../Datasets/iris.csv",names = ["Sepal Length","Sepal Width","Petal Length","Petal Width","Category"])


# In[3]:


# As we have discussed, that in machine learning it is neccessary to split dataset into training and testing dataset
# so as to get a better insight on the performance of our models.

# To divide the data we are using groupby function of pandas library which group our data by category meaning
# we get three groups setosa, virginica and versicolor and out of each group we take the first 30 rows for training
# using head function and last 20 rows for testing using tail function. To learn more about grouping in pandas
# you can refer to the below given link.
# https://pandas.pydata.org/pandas-docs/version/0.23/groupby.html


training_data = data.groupby('Category').head(30)
testing_data = data.groupby('Category').tail(20)

print ("\nTraining Data")
print ("===============\n")
print (training_data)
print ("\nTesting Data")
print ("===============\n")
print (testing_data)


# In[4]:


# Initialization of Naive Bayes and K nearest neighbour models. 

bayes_classifier = NaiveBayes()
knn_classifier = KNN(n_neighbors = 5)


# In[5]:


# Training of our Naive bayes and K nearest neighbour models.

# Input variables of training data
train_X = training_data.iloc[:,:4]

# Output variable of training data
train_Y = training_data.iloc[:,4]

bayes_classifier.fit(train_X,train_Y)
knn_classifier.fit(train_X,train_Y)


# In[6]:


# Input variables of testing data
test_X = testing_data.iloc[:,:4]

# Output variable of testing data
test_Y = testing_data.iloc[:,4]

# Predictions using knn
pred_Y_KNN = knn_classifier.predict(test_X)

# Predictions using Bayes
pred_Y_Bayes = bayes_classifier.predict(test_X)


# In[45]:


print ("Accuracy:")
print ("K-Nearest Neighbour:",accuracy_score(pred_Y_KNN,test_Y))
print ("Naive Bayes:",accuracy_score(pred_Y_Bayes,test_Y),"\n")

print ("Precision:")
print ("K-Nearest Neighbour:",precision_score(pred_Y_KNN,test_Y,average="micro"))
print ("Naive Bayes:",precision_score(pred_Y_Bayes,test_Y,average="micro"),"\n")

print ("Recall:")
print ("K-Nearest Neighbour:",recall_score(pred_Y_KNN,test_Y,average="micro"))
print ("Naive Bayes:",recall_score(pred_Y_Bayes,test_Y,average="micro"))

