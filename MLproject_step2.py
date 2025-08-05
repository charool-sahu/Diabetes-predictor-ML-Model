import pandas as pd  # import the pandas library and give it a short name 'pd'

df = pd.read_csv("diabetes.csv")  # load the file into Python
print(df.head())  # show first 5 rows of the data
