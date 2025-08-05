import pandas as pd  # make sure pandas is imported

df = pd.read_csv("diabetes.csv")  # load the data from the file

df.fillna(df.mean(), inplace=True)  #Fill missing values with the column’s average (mean)


# Step: Check missing values and split into input (X) and target (y)

#print(df.isnull().sum())  # check if any column has missing data

X = df.drop('Outcome', axis=1)  # X = features (what we use to predict)
y = df['Outcome']               # y = target (what we want to predict)


#normalise or standardise the data

from sklearn.preprocessing import StandardScaler  # tool to scale data

scaler = StandardScaler()              # create the scaler object
X_scaled = scaler.fit_transform(X)     # fit and transform the input features (X)

#print(X_scaled[:5])  # look at first 5 scaled rows

# Optional: print shape to confirm it worked
#print("Shape before scaling:", X.shape)
#print("Shape after scaling:", X_scaled.shape)

#print(y.unique())  # shows all unique values in the Outcome column


from sklearn.model_selection import train_test_split  # tool to split data

# Split into training and testing sets (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Check how many rows in each
#print("Training rows:", len(X_train))
#print("Testing rows:", len(X_test))

from sklearn.linear_model import LogisticRegression  # model for prediction

model = LogisticRegression()      # create the model
model.fit(X_train, y_train)       # train it on training data

#step final: Make Predictions on Testing Data
# Import tools to measure accuracy and performance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#  Step final.1 — Make predictions on test data
# This is like asking the model to take a test using questions (X_test) it has never seen
y_pred = model.predict(X_test)

#  Step final.2 — Calculate accuracy of the model
# Accuracy means: how many predictions were correct out of total
accuracy = accuracy_score(y_test, y_pred)
print(" Accuracy of the model:", accuracy)

#  Step final.3 — Show confusion matrix
# This helps you see: How many 1s and 0s were predicted correctly or incorrectly
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:\n", cm)
# Format: [[TN, FP], [FN, TP]]
# TN = True Negatives (predicted 0, actually 0)
# TP = True Positives (predicted 1, actually 1)
# FP = False Positives (predicted 1, actually 0)
# FN = False Negatives (predicted 0, actually 1)

#  Step final.4 — Show detailed classification report
# It shows precision, recall, F1-score for both classes (0 and 1)
print("\n🧾 Classification Report:\n")
print(classification_report(y_test, y_pred))
