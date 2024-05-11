matrix = confusion_matrix(y_test, y_pred)
# sns.heatmap(matrix, annot=True, fmt="d")
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# print(classification_report(y_test, y_pred))
# TP = matrix[1, 1]
# TN = matrix[0, 0]
# FP = matrix[0, 1]
# FN = matrix[1, 0]

# # ginni
# from sklearn.ensemble import RandomForestClassifier
# clf= RandomForestClassifier (criterion='gini',
#                              max_depth = 8,
#                              min_samples_split=10,
#                              random_state=5)
# clf.fit (x_train,y_train)
# clf.feature_importances_
# importances = clf.feature_importances_
# indices = np.argsort(importances)
# plt.figure(figsize = (8,5))
# plt.barh(range(len(indices)), importances[indices])
# plt.yticks(range(len(indices)), x_train.columns[indices])
# plt.title("Importance Features")
# y_pred = clf.predict(x_test)

# matrix = confusion_matrix(y_test, y_pred)
# sns.heatmap(matrix, annot=True, fmt="d")
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# print(classification_report(y_test, y_pred))
# TP = matrix[1, 1]
# TN = matrix[0, 0]
# FP = matrix[0, 1]
# FN = matrix[1, 0]

# """Ginni
# -------------------------------------------------
# -----------------------------------------
# """

# from sklearn.ensemble import RandomForestClassifier
# clf= RandomForestClassifier (criterion='gini',
#                              max_depth = 8,
#                              min_samples_split=10,
#                              random_state=5)
# clf.fit (x_train,y_train)
# clf.feature_importances_
# importances = clf.feature_importances_
# plt.yticks(range(len(indices)), x_train.columns[indices])
# plt.title("Importance Features")
# # y_pred = clf.predict(x_test)

# # matrix = confusion_matrix(y_test, y_pred)
# # sns.heatmap(matrix, annot=True, fmt="d")
# # plt.title('Confusion Matrix')
# # plt.xlabel('Predicted')
# # plt.ylabel('True')
# # print(classification_report(y_test, y_pred))
# # TP = matrix[1, 1]
# # TN = matrix[0, 0]
# # FP = matrix[0, 1]
# # FN = matrix[1, 0]

# """**Entropy**
# ------------------------------------------------------------------













# """

# from sklearn.ensemble import RandomForestClassifier
# clf= RandomForestClassifier (criterion='entropy',
#                              max_depth = 8,
#                              min_samples_split=10,
#                              random_state=5)
# clf.fit (x_train,y_train)
# clf.feature_importances_
# importances = clf.feature_importances_


# indices = np.argsort(importances)
# plt.figure(figsize = (8,5))
# plt.barh(range(len(indices)), importances[indices])
# plt.yticks(range(len(indices)), x_train.columns[indices])
# plt.title("Importance Features")
# y_pred = clf.predict(x_test)

# # matrix = confusion_matrix(y_test, y_pred)
# # sns.heatmap(matrix, annot=True, fmt="d")
# # plt.title('Confusion Matrix')
# # plt.xlabel('Predicted')
# # plt.ylabel('True')
# # print(classification_report(y_test, y_pred))
# # TP = matrix[1, 1]
# # TN = matrix[0, 0]
# # FP = matrix[0, 1]
# # FN = matrix[1, 0]

# importances

# """# **Log_loss**
# ---------------------------------------
# """

# from sklearn.ensemble import RandomForestClassifier
# clf= RandomForestClassifier (criterion='log_loss',
#                              max_depth = 8,
#                              min_samples_split=10,
#                              random_state=5)
# clf.fit (x_train,y_train)
# clf.feature_importances_
# importances = clf.feature_importances_
# indices = np.argsort(importances)
# plt.figure(figsize = (8,5))
# plt.barh(range(len(indices)), importances[indices])
# plt.yticks(range(len(indices)), x_train.columns[indices])
# plt.title("Importance Features")
# # y_pred = clf.predict(x_test)

# # matrix = confusion_matrix(y_test, y_pred)
# # sns.heatmap(matrix, annot=True, fmt="d")
# # plt.title('Confusion Matrix')
# # plt.xlabel('Predicted')
# # plt.ylabel('True')
# # print(classification_report(y_test, y_pred))
# # TP = matrix[1, 1]
# # TN = matrix[0, 0]
# # FP = matrix[0, 1]
# # FN = matrix[1, 0]

# from sklearn.ensemble import RandomForestClassifier
# from tabulate import tabulate
# import numpy as np
# import matplotlib.pyplot as plt

# # Assuming x_train and y_train are defined

# clf = RandomForestClassifier(criterion='log_loss', max_depth=8, min_samples_split=10, random_state=5)
# clf.fit(x_train, y_train)

# importances = clf.feature_importances_
# indices = np.argsort(importances)

# # Plot the feature importances
# plt.figure(figsize=(8, 5))
# plt.barh(range(len(indices)), importances[indices])
# plt.yticks(range(len(indices)), x_train.columns[indices])
# plt.title("Importance Features")
# plt.show()

# # Create a table of feature importances
# table_data = zip(x_train.columns[indices], importances[indices])
# table_headers = ["Feature", "Importance"]
# table = tabulate(table_data, headers=table_headers, tablefmt="pretty")

# print("Feature Importances Table:")
# print(table)

# from sklearn.ensemble import RandomForestClassifier
# import numpy as np
# import matplotlib.pyplot as plt
# from tabulate import tabulate

# # Assuming x_train and y_train are defined

# clf = RandomForestClassifier(criterion='log_loss', max_depth=8, min_samples_split=10, random_state=5)
# clf.fit(x_train, y_train)

# importances = clf.feature_importances_
# indices = np.argsort(importances)[::-1]  # Sort indices in descending order

# # Create a table of sorted feature importances
# table_data = list(zip(x_train.columns[indices], importances[indices]))
# table_headers = ["Feature", "Importance"]
# table = tabulate(table_data, headers=table_headers, tablefmt="grid")

# # Plot the sorted feature importances as a table
# fig, ax = plt.subplots(figsize=(8, 5))
# ax.axis('off')
# table = ax.table(cellText=table_data, colLabels=table_headers, loc='center', cellLoc='center', colLoc='center')
# table.auto_set_font_size(False)
# table.set_fontsize(10)

# # Save the table as an image
# plt.savefig("feature_importances_table_sorted.png", bbox_inches='tight')

# # Display the plot
# plt.show()

# from sklearn.ensemble import RandomForestClassifier
# import numpy as np
# import matplotlib.pyplot as plt
# import csv

# # Assuming x_train and y_train are defined

# clf = RandomForestClassifier(criterion='log_loss', max_depth=8, min_samples_split=10, random_state=5)
# clf.fit(x_train, y_train)

# importances = clf.feature_importances_
# indices = np.argsort(importances)

# # Plot the feature importances
# plt.figure(figsize=(8, 5))
# plt.barh(range(len(indices)), importances[indices])
# plt.yticks(range(len(indices)), x_train.columns[indices])
# plt.title("Importance Features")
# plt.show()

# # Create a table of feature importances
# table_data = zip(x_train.columns[indices], importances[indices])
# table_headers = ["Feature", "Importance"]

# # Save to CSV file
# csv_filename = "feature_importances.csv"
# with open(csv_filename, 'w', newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(table_headers)
#     writer.writerows(table_data)

# print(f"Feature Importances Table saved to {csv_filename}")

# """Using bayesian model

# ================================
# ================================
# """

# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# from sklearn import metrics

# # Assuming 'X' is your feature matrix and 'y' is your target variable
# x = iv
# y = df['Level']

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # Create a Gaussian Naive Bayes classifier
# nb_classifier = GaussianNB()

# # Train the classifier
# nb_classifier.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = nb_classifier.predict(X_test)

# # Evaluate the performance
# accuracy = metrics.accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")

# df.info()

# # arr1 = list()
# # arr1 = list()

# # arr1 = list()
# # arr1 = list()
# # arr1 = list()
# # arr1 = list()
# # arr1 = list()
# arrCount = [0,0,0,0,0,0,0,0,]


# for i in df['Level']:
#   values = int(i)
#   # print(values)
#   if values == 0:
#     alcohol = int(df['Alcohol use'][values])
#     arrCount[values]+=1

# print(arrCount)

# """/// probabaility



# """

# new_df = df[['Alcohol use', 'Level']]
# new_df

# alcoholOne = [0,0,0,0,0,0,0,0,]

# for index, row in new_df.iterrows():
#     if row['Alcohol use'] == 1 and row['Level'] == 0:
#         alcoholOne[0] += 1

#     elif row['Alcohol use'] == 2 and row['Level'] == 0:
#         alcoholOne[1] += 1

#     elif row['Alcohol use'] == 3 and row['Level'] == 0:
#         alcoholOne[2] += 1

#     elif row['Alcohol use'] == 4 and row['Level'] == 0:
#         alcoholOne[3] += 1

#     elif row['Alcohol use'] == 5 and row['Level'] == 0:
#         alcoholOne[4] += 1

#     elif row['Alcohol use'] == 6 and row['Level'] == 0:
#         alcoholOne[5] += 1

#     elif row['Alcohol use'] == 7 and row['Level'] == 0:
#         alcoholOne[6] += 1
#     elif row['Alcohol use'] == 8 and row['Level'] == 0:
#         alcoholOne[7] += 1

# print(sum(alcoholOne))
# print(alcoholOne)

# alcoholOne = [0,0,0,0,0,0,0,0,]

# for index, row in new_df.iterrows():
#     if row['Alcohol use'] == 1 and row['Level'] == 1:
#         alcoholOne[0] += 1

#     elif row['Alcohol use'] == 2 and row['Level'] == 1:
#         alcoholOne[1] += 1

#     elif row['Alcohol use'] == 3 and row['Level'] == 1:
#         alcoholOne[2] += 1

#     elif row['Alcohol use'] == 4 and row['Level'] == 1:
#         alcoholOne[3] += 1

#     elif row['Alcohol use'] == 5 and row['Level'] == 1:
#         alcoholOne[4] += 1

#     elif row['Alcohol use'] == 6 and row['Level'] == 1:
#         alcoholOne[5] += 1

#     elif row['Alcohol use'] == 7 and row['Level'] == 1:
#         alcoholOne[6] += 1
#     elif row['Alcohol use'] == 8 and row['Level'] == 1:
#         alcoholOne[7] += 1

# print(alcoholOne)

# alcoholOne = [0,0,0,0,0,0,0,0]

# for index, row in new_df.iterrows():
#     if row['Alcohol use'] == 1 and row['Level'] == 2:
#         alcoholOne[0] += 1

#     elif row['Alcohol use'] == 2 and row['Level'] == 2:
#         alcoholOne[1] += 1

#     elif row['Alcohol use'] == 3 and row['Level'] == 2:
#         alcoholOne[2] += 1

#     elif row['Alcohol use'] == 4 and row['Level'] == 2:
#         alcoholOne[3] += 1

#     elif row['Alcohol use'] == 5 and row['Level'] == 2:
#         alcoholOne[4] += 1

#     elif row['Alcohol use'] == 6 and row['Level'] == 2:
#         alcoholOne[5] += 1

#     elif row['Alcohol use'] == 7 and row['Level'] == 2:
#         alcoholOne[6] += 1
#     elif row['Alcohol use'] == 8 and row['Level'] == 2:
#         alcoholOne[7] += 1

# print(sum(alcoholOne))

# # Assuming 'Alcohol use' is the column name in your DataFrame
# alcohol_counts = new_df['Alcohol use'].value_counts().sort_index()

# print("Alcohol Counts:")
# print(alcohol_counts)

# # Assuming 'Alcohol use' and 'Level' are the column names in your DataFrame
# filtered_df = new_df[(new_df['Level'] == 0) & (new_df['Alcohol use'] >= 1) & (new_df['Alcohol use'] <= 8)]
# alcohol_counts = filtered_df['Alcohol use'].value_counts().sort_index()

# print("Alcohol Counts in the Range 1 to 8 when Level is 0:")
# print(alcohol_counts)

# """// Coughing of blood--------------------------
# // Probability calculation-------------------- **bold text**
# """

# new_df = df[['Coughing of Blood', 'Level']]
# new_df

# # coughing = set()

# # for i in df['Coughing of Blood']:
# #   coughing.add(i)

# # print(coughing)

# coughing = [0,0,0,0,0,0,0,0 , 0]

# for index, row in new_df.iterrows():
#     if row['Coughing of Blood'] == 1 and row['Level'] ==  2:
#         coughing[0] += 1

#     elif row['Coughing of Blood'] == 2 and row['Level'] ==  2:
#         coughing[1] += 1

#     elif row['Coughing of Blood'] == 3 and row['Level'] ==  2:
#         coughing[2] += 1

#     elif row['Coughing of Blood'] == 4 and row['Level'] ==  2:
#         coughing[3] += 1

#     elif row['Coughing of Blood'] == 5 and row['Level'] ==  1:
#         coughing[4] += 1

#     elif row['Coughing of Blood'] == 6 and row['Level'] ==  2:
#         coughing[5] += 1

#     elif row['Coughing of Blood'] == 7 and row['Level'] ==  2:
#         coughing[6] += 1
#     elif row['Coughing of Blood'] == 8 and row['Level'] ==  2:
#         coughing[7] += 1
#     elif row['Coughing of Blood'] == 9 and row['Level'] == 2:
#         coughing[8] += 1
# print(coughing)

# """// obesity======================================="""

# # Obesity = set()

# # for i in df['Obesity']:
# #   Obesity.add(i)

# # print(Obesity)

# new_df = df[['Obesity', 'Level']]
# # new_df

# obesity_counts = [0, 0, 0, 0, 0, 0, 0]

# for index, row in new_df.iterrows():
#     if row['Obesity'] == 1 and row['Level'] == 1:
#         obesity_counts[0] += 1

#     elif row['Obesity'] == 2 and row['Level'] == 1:
#         obesity_counts[1] += 1

#     elif row['Obesity'] == 3 and row['Level'] == 1:
#         obesity_counts[2] += 1

#     elif row['Obesity'] == 4 and row['Level'] == 1:
#         obesity_counts[3] += 1

#     elif row['Obesity'] == 5 and row['Level'] == 1:
#         obesity_counts[4] += 1

#     elif row['Obesity'] == 6 and row['Level'] ==1:
#         obesity_counts[5] += 1

#     elif row['Obesity'] == 7 and row['Level'] == 1:
#         obesity_counts[6] += 1

# print(obesity_counts)

# """// PAssive Smoker"""

# # Obesity = set()

# # for i in df['Passive Smoker']:
# #   Obesity.add(i)

# # print(Obesity)

# new_df = df[['Passive Smoker', 'Level']]



# passive_smoker_counts = [0, 0, 0, 0, 0, 0, 0, 0]

# for index, row in new_df.iterrows():
#     if row['Passive Smoker'] == 1 and row['Level'] == 2:
#         passive_smoker_counts[0] += 1

#     elif row['Passive Smoker'] == 2 and row['Level'] == 2:
#         passive_smoker_counts[1] += 1

#     elif row['Passive Smoker'] == 3 and row['Level'] == 2:
#         passive_smoker_counts[2] += 1

#     elif row['Passive Smoker'] == 4 and row['Level'] == 2:
#         passive_smoker_counts[3] += 1

#     elif row['Passive Smoker'] == 5 and row['Level'] == 2:
#         passive_smoker_counts[4] += 1

#     elif row['Passive Smoker'] == 6 and row['Level'] == 2:
#         passive_smoker_counts[5] += 1

#     elif row['Passive Smoker'] == 7 and row['Level'] == 2:
#         passive_smoker_counts[6] += 1

#     elif row['Passive Smoker'] == 8 and row['Level'] == 2:
#         passive_smoker_counts[7] += 1

# print(passive_smoker_counts)

# """/// Balanced Diet

# """

# # Obesity = set()

# # for i in df['Balanced Diet']:
# #   Obesity.add(i)

# # print(Obesity)

# new_df = df[['Balanced Diet', 'Level']]

# balanced_diet_counts = [0, 0, 0, 0, 0, 0, 0]

# for index, row in new_df.iterrows():
#     if row['Balanced Diet'] == 1 and row['Level'] == 2:
#         balanced_diet_counts[0] += 1

#     elif row['Balanced Diet'] == 2 and row['Level'] == 2:
#         balanced_diet_counts[1] += 1

#     elif row['Balanced Diet'] == 3 and row['Level'] == 2:
#         balanced_diet_counts[2] += 1

#     elif row['Balanced Diet'] == 4 and row['Level'] == 2:
#         balanced_diet_counts[3] += 1

#     elif row['Balanced Diet'] == 5 and row['Level'] == 2:
#         balanced_diet_counts[4] += 1

#     elif row['Balanced Diet'] == 6 and row['Level'] == 2:
#         balanced_diet_counts[5] += 1

#     elif row['Balanced Diet'] == 7 and row['Level'] == 2:
#         balanced_diet_counts[6] += 1

# print(balanced_diet_counts)

# """// Wheezing"""

# # Obesity = set()

# # for i in df['Fatigue']:
# #   Obesity.add(i)

# # print(Obesity)

# new_df = df[['Wheezing', 'Level']]

# wheezing_counts = [0, 0, 0, 0, 0, 0, 0, 0]

# for index, row in new_df.iterrows():
#     if row['Wheezing'] == 1 and row['Level'] == 0:
#         wheezing_counts[0] += 1

#     elif row['Wheezing'] == 2 and row['Level'] == 0:
#         wheezing_counts[1] += 1

#     elif row['Wheezing'] == 3 and row['Level'] == 0:
#         wheezing_counts[2] += 1

#     elif row['Wheezing'] == 4 and row['Level'] ==0:
#         wheezing_counts[3] += 1

#     elif row['Wheezing'] == 5 and row['Level'] == 0:
#         wheezing_counts[4] += 1

#     elif row['Wheezing'] == 6 and row['Level'] == 0:
#         wheezing_counts[5] += 1

#     elif row['Wheezing'] == 7 and row['Level'] == 0:
#         wheezing_counts[6] += 1

#     elif row['Wheezing'] == 8 and row['Level'] == 0:
#         wheezing_counts[7] += 1

# print(wheezing_counts)

# # Create Dictionary Function

# def createDs(colName , df):
#   dictionary = dict()
#   for i in df[colName]:
#     dictionary.update({i:0})

#   return dictionary

# # count values of given Level wrt to given colName

# def countValues(colName , Lev, ds , df):

#   for index, row in df.iterrows():
#     if row["Level"] == Lev:
#       value = row[colName]
#       ds[value] = ds.get(value, 0) + 1


#   return ds

# df.info()

# colName = "Air Pollution"
# new_df = df[[colName, 'Level']]

# ds0 = (createDs(colName , new_df))
# ds1 = (createDs(colName , new_df))
# ds2 = (createDs(colName , new_df))

# ds0 = countValues(colName , 0 , ds0 , new_df)
# ds1 = countValues(colName , 1 , ds1 , new_df)
# ds2 = countValues(colName , 2 , ds2 , new_df)

# print(f"level 0 of {colName} is " , ds0)
# print(f"level 1 of {colName} is " , ds1)
# print(f"level 2 of {colName} is " , ds2)