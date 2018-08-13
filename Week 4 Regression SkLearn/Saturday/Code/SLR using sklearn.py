
# coding: utf-8

# In[22]:


import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#############
## Data IO ##
#############

def get_data(filepath):
	# Opens the file handler for the dataset file. Using variable 'f' we can access and manipulate our file anywhere in our code
	# after the next code line.
	f = open(filepath,"r")

	# Predictors Collection (or your input variable) (which in this case is just the duration of eruption)
	X = []

	# Output Response (or your output variable) (which in this case is the duration after which next eruption will occur.)
	Y = []

	# Initializing a reader generator using reader method from csv module. A reader generator takes each line from the file
	# and converts it into list of columns.
	reader = csv.reader(f)

	# Using for loop, we are able to read one row at a time.
	for row in reader:
		if row[1]!="Duration":
			X.append(float(row[1]))
			Y.append(float(row[2]))

	# Close the file once we have succesffuly stored all data into our X and Y variables.
	f.close()

	return [X,Y]

X,Y = get_data("../Datasets/geyser.csv")

# Reshaping and formatting of data
X = np.array(X).reshape(len(X),1)
Y = np.array(Y)

# Initializing the simple linear Regression model
slr = LinearRegression(normalize = True)
# Fitting (or Training) the simple linear regression model
slr.fit(X,Y)

print("Coefficient(beta_one): ",slr.coef_)
print("(beta_zero): ",slr.intercept_)

# Total error of the fit
print (mean_absolute_error(Y,slr.predict(X)))
print (mean_squared_error(Y,slr.predict(X)))
print (r2_score(Y,slr.predict(X)))

print (slr.predict(3.4))

