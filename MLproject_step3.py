#EXPLORE AND CLEAN THE DATA

import pandas as pd  # make sure pandas is imported

df = pd.read_csv("diabetes.csv")  # load the data from the file


# Step: Check missing values and split into input (X) and target (y)

print(df.isnull().sum())  # check if any column has missing data

X = df.drop('Outcome', axis=1)  # X = features (what we use to predict)
y = df['Outcome']               # y = target (what we want to predict)


print(X)
print(y)