# load dataset

import pandas as pd
dataset=pd.read_csv("SalaryData.csv")
print("Dataset has been loaded..")

# creating feature and target variable
# Here x is feature and y is target

x=dataset[["YearsExperience"]]
y=dataset["Salary"]

# now to train our model

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
print("Model has been created..")

# Now model has been trained

# for saving model

import joblib
joblib.dump(model, "Salarypredicter-model.pk1")
print("Now our model has been saved..")




