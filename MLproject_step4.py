import pandas as pd  # make sure pandas is imported

df = pd.read_csv("diabetes.csv")  # load the data from the file


# Step: Check missing values and split into input (X) and target (y)

print(df.isnull().sum())  # check if any column has missing data

X = df.drop('Outcome', axis=1)  # X = features (what we use to predict)
y = df['Outcome']               # y = target (what we want to predict)




#normalise or standardise the data

from sklearn.preprocessing import StandardScaler  # tool to scale data

scaler = StandardScaler()              # create the scaler object
X_scaled = scaler.fit_transform(X)     # fit and transform the input features (X)

print(X_scaled[:5])  # look at first 5 scaled rows

# Optional: print shape to confirm it worked
print("Shape before scaling:", X.shape)
print("Shape after scaling:", X_scaled.shape)

print(y.unique())  # shows all unique values in the Outcome column

