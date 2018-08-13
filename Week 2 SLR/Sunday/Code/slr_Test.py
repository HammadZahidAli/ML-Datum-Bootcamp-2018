# -*- coding: utf-8 -*-


import csv
import numpy as np
import scipy as sp
import matplotlib
from matplotlib import pyplot as plt


########################
## Our Custom Methods ##
########################

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
		if row[0]!="var":
			X.append(float(row[0]))
			Y.append(float(row[1]))


	# Close the file once we have succesffuly stored all data into our X and Y variables.
	f.close()

	return [X,Y]

# Method to estimate the coefficients
# This is the method which is used to train (or teach) our simple linear regression model
# Detailed Derivation of beta_1 and beta_0 estimation is present at the url:
# https://are.berkeley.edu/courses/EEP118/current/derive_ols.pdf
# Y = B1*X + B0

def slr(X,Y):
	intermediate_beta = []
	beta_1_numerator_sum = 0.0
	beta_1_denominator_sum = 0.0

	for i in range(0,len(X)):
		beta_1_numerator_sum = beta_1_numerator_sum + ((X[i]-np.mean(X))*(Y[i]-np.mean(Y)))
		beta_1_denominator_sum = beta_1_denominator_sum + ((X[i] - np.mean(X))**2)

		beta_1 = beta_1_numerator_sum/beta_1_denominator_sum
		beta_0 = np.mean(Y) - ((beta_1)*np.mean(X))

		intermediate_beta.append([beta_0,beta_1])

	return [[beta_0,beta_1],intermediate_beta]


# Method to predict response variable Y (in this case interval before the next erruption) for new values of X (in this case
# duration of eruption) using the estimated coefficientsself.
# This method can predict Response variable (Y) for single as well as multiple values of X. If only a single numerical Value
# input variable (X) which in this case is Duration is passed. It will return the prediction for only that single numerical
# value. If a collection of different values for input variable (list) is passed, it will return a list of predictions
# for each input value.
# "if" statement on line number 72 takes care of understanding if the input value is singular or a list.
def predict(coef,X):
	beta_0 = coef[0]
	beta_1 = coef[1]

	fy = []
	if type(X) == list:
		for x in X:
			fy.append(beta_0 + (beta_1 * x))
		return fy

	# Our Regression Model defined using the coefficients from slr function
	Y = beta_0 + (beta_1 * X)

	return Y

# This method Visualize the training process and how will the training process in simple linear Regression
# would look like if you plot it on the scatter plot.
def training_process_visualization(X,Y):
	plt.ion()
	plt.figure(figsize=(6,5))
	for i in range(0,len(X)):
		plt.title("Training Process")
		plt.xlabel("Duration of Eruption (in minutes)")
		plt.ylabel("Time duration before the next eruption (in minutes)")
		plt.xlim(1.3,5.1)
		plt.ylim(40,95)
		plt.scatter(X[0:i], Y[0:i],s=2)
		x = sp.linspace(1.4,5, 2000)
		y = predict(intermediate_beta[i],x)
		print ("i = "+str(i)+" | x = "+str(round(X[i],4))+" | y = "+str(round(Y[i],4))+" | Updated_Beta0 = "+str(round(intermediate_beta[i][0],2))+" | Updated_Beta1 = "+str(round(intermediate_beta[i][1],2)))
		plt.plot(x,y)
		plt.pause(0.001)
		if i != (len(X) - 1):
			plt.clf()

# Visualize the data using Scatter plot of matplotlib library. A scatter plot is a plot between two continuous variables.
# and it helps us in determining the relationship between those two continuous variables.
# For more information on working of scatter plot function of matplotlib - you can visit the following url:
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
def show_scatter_plot(X,Y,coefficients = None):
	plt.scatter(X,Y,s = 2)
	plt.xlabel("Duration of Eruption (in minutes)")
	plt.ylabel("Time duration before the next eruption (in minutes)")
	if coefficients:
		x = sp.linspace(1,X[-1], 2000)
		plt.plot(x, predict(coefficients,x), linewidth=2)
	plt.show()

############################################################################################################################

X,Y = get_data("../Datasets/test.csv")
show_scatter_plot(X,Y)

################################################
## Model Training (or coefficient estimation) ##
################################################
# Using our slr function we estimate coefficients of our regression line. The slr function returns a list of coefficients

coefficients,intermediate_beta = slr(X,Y)
training_process_visualization(X,Y)
show_scatter_plot(X,Y,coefficients)

########################
## Making Predictions ##
########################

# Using our predict function and the coefficients given by our slr function we can now predict the time it will take
# for the next eruption.
last_eruption_duration = float(input("Duration of the last eruption (in minutes):"))
print ("Time it will take for the next eruption to occur (in minutes):",predict(coefficients,last_eruption_duration))

#######################
## Error Calculation ##
#######################

print ("\n\nAccuracy Metrics of the model\n-------------------------------------")

# Calculation of RSE
RSS = 0
for idx in range(0,len(X)):
    actual_y = Y[idx]
    predicted_y = predict(coefficients,X[idx])
    RSS = RSS + ((actual_y - predicted_y)**2)

RSE = np.sqrt((1/float(len(X)-2))*RSS)
print("RSS: ",RSS)
print ("Residual Standard Error:",RSE)
print ("% Residual Standard Error (over average Interval):", (RSE/np.mean(Y))*100)


# Calculation of R_Squared
TSS = 0
for idx in range(0,len(X)):
    actual_y = Y[idx]
    TSS = TSS + ((actual_y - np.mean(Y))**2)

R_Squared = ((TSS) - (RSS)) / (TSS)

print ("\nR-Squared Value:",R_Squared)
