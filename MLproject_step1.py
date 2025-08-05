import pandas as pd       # to work with data tables
import numpy as np        # to do math operations
import matplotlib.pyplot as plt  # to make charts
import seaborn as sns     # to make prettier charts

from sklearn.model_selection import train_test_split  # to split data
from sklearn.preprocessing import StandardScaler       # to scale data
from sklearn.linear_model import LogisticRegression    # ML model 1
from sklearn.ensemble import RandomForestClassifier    # ML model 2
from sklearn.svm import SVC                            # ML model 3

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
