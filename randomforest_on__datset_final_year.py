# -*- coding: utf-8 -*-
"""RandomForest_on__datset_Final_year.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_I93qz2VtQ9IibvoYFwosO7M_rbaPxf2

# **Random Forsest on dataset**

---
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import LogisticRegression


def processModel(input_values):
    df = pd.read_csv('cancer_patient_data_sets.csv')


    df = df.drop(columns=['index', 'Patient Id' ])

    df["Level"].replace({'High': 2, 'Medium': 1, 'Low': 0}, inplace=True)


    df["Level"].value_counts()

    iv = df.copy()
    iv = iv.drop(columns=['Level'], axis = 1)


    x= df.iloc[:, :-1]
    y= df.iloc[:,  -1]


    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    from sklearn.ensemble import RandomForestClassifier
    clf= RandomForestClassifier (criterion='gini',
                             max_depth = 8,
                             min_samples_split=10,
                             random_state=5)

    from sklearn.ensemble import RandomForestClassifier
    clf= RandomForestClassifier (criterion='entropy',
                             max_depth = 8,
                             min_samples_split=10,
                             random_state=5)

    clf.fit (x_train,y_train)

    clf.feature_importances_


# entropy
    clf.feature_importances_
    importances = clf.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize = (8,5))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), x_train.columns[indices])
    plt.title("Importance Features")

    importances = clf.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize = (8,5))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), x_train.columns[indices])
    plt.title("Importance Features")

    y_pred = clf.predict(x_test)

    feature_names = df.columns.tolist()

    features = ['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards',
            'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking',
            'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
            'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
            'Frequent Cold', 'Dry Cough', 'Snoring']


    input_df = pd.DataFrame([input_values], columns=features)

    y_pred = clf.predict(input_df)
    return y_pred

# input_values = [0, 0, 7, 7, 7, 7, 5, 7, 4, 8, 7, 7, 3, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8]
input_values =[0,1,4,5,6,5,5,4,6,7,2,3,4,8,8,7,9,2,1,4,6,7,2]

print(processModel(input_values))

