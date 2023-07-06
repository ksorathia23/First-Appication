#!/usr/bin/env python
# coding: utf-8

# Python linear Modeling Assignment

# Import pandas, sklearn, matplotlib
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys

print(sys.argv[1])

print("Running linear modeling of data python script")
print()

# set notebook variable
filename = "regrex1.csv"

print ("loading filename {}".format(filename))
# use read_csv() to read regex1.csv file
dataset = pd.read_csv(filename)
dataset.describe()
dataset

# Plot Data
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("scatter.png")

# Fitting Linear Regression to the Dataset
model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])


# Adjusting R Squared
model.score(dataset[['x']], dataset[['y']])


# Visualizing the Linear Regression results
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("scatter.png")
