#！/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time  : 18/10/2022  16:50
@Author: Skye
@File  : (Haozhe TANG) Final_project_code.py 
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

names = ["age", "workclass", "final-wt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"]
df_train = pd.read_csv('adult.train.csv', names=names)
df_test = pd.read_csv('adult.test.csv', names=names)

## Function for labeling
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2. -0.08, 1.01*height, "%s" % int(height), size=20, family="Times new roman")

# Data analytics
# The number of people earning more than 50k (Original data)
more_than_50k = len([salary for salary in df_train['salary'] if salary == " >50K"]) + len([salary for salary in df_test['salary'] if salary == " >50K"])
less_than_50k = len(df_train) + len(df_test) - more_than_50k
ratio_more_than_50k = more_than_50k/(more_than_50k + less_than_50k)
ratio_less_than_50k = less_than_50k/(more_than_50k + less_than_50k)
salary_list = [more_than_50k, less_than_50k]
ratio_list = [ratio_more_than_50k, ratio_less_than_50k]

#Draw the graph
#Bar
ax1 = plt.figure(figsize=(10, 10))
plt.suptitle("Income distribution of original data", fontsize=15)
plt.subplot(1, 2, 1)
plt.title("Rumber of the distribution of people's income", fontsize=15)
plt.xlabel('Category', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=10)
fig_salary = plt.bar(['more than 50k', 'less than 50k'], salary_list, color=['r', 'b'])
autolabel(fig_salary)

#Pie
plt.subplot(1, 2, 2)
plt.title("Ratio of the distribution of people's income", fontsize=15)
plt.pie(ratio_list, labels=['more than 50k', 'less than 50k'],autopct="%.2f%%")
plt.savefig("Income distribution of original data.png")
ax1.show()


#For global process, we should delete the lines with "space"
df_global_train = df_train.dropna()
df_global_test = df_test.dropna()

# Numeric mapping
map_workclass = {
    " Private": 0,
    " Self-emp-not-inc": 1,
    " Self-emp-inc": 2,
    " Federal-gov": 3,
    " Local-gov": 4,
    " State-gov": 5,
    " Without-pay": 6,
    " Never-worked": 7
}
map_edu = {
    " Preschool": 0,
    " 1st-4th": 1,
    " 5th-6th": 2,
    " 7th-8th": 3,
    " 9th": 4,
    " 10th": 5,
    " 11th": 6,
    " 12th": 7,
    " HS-grad": 8,
    " Some-college": 9,
    " Prof-school": 10,
    " Assoc-acdm": 11,
    " Assoc-voc": 12,
    " Bachelors": 13,
    " Masters":14,
    " Doctorate":15
}
map_marry = {
    " Married-civ-spouse": 0,
    " Divorced": 1,
    " Never-married": 2,
    " Separated": 3,
    " Widowed": 4,
    " Married-spouse-absent": 5,
    " Married-AF-spouse": 6
}
map_job = {
    " Tech-support": 0,
    " Craft-repair": 1,
    " Sales": 2,
    " Exec-managerial": 3,
    " Prof-specialty": 4,
    " Handlers-cleaners": 5,
    " Machine-op-inspct": 6,
    " Adm-clerical": 7,
    " Farming-fishing": 8,
    " Transport-moving": 9,
    " Priv-house-serv": 10,
    " Protective-serv": 11,
    " Armed-Forces": 12,
    " Other-service": 13,
}
map_relation = {
    " Wife": 0,
    " Own-child": 1,
    " Husband": 2,
    " Not-in-family": 3,
    " Other-relative": 4,
    " Unmarried": 5
}
map_race = {
    " White": 0,
    " Black": 1,
    " Asian-Pac-Islander": 2,
    " Amer-Indian-Eskimo": 3,
    " Other": 4
}
map_sex = {
    " Female": 0,
    " Male": 1
}
map_country = {
    " United-States": 1,
    " Cambodia": 2,
    " England": 3,
    " Puerto-Rico": 4,
    " Canada": 5,
    " Germany": 6,
    " Outlying-US(Guam-USVI-etc)": 7,
    " India": 8,
    " Japan": 9,
    " Greece": 10,
    " South": 11,
    " China": 12,
    " Cuba": 13,
    " Iran": 14,
    " Honduras": 15,
    " Philippines": 16,
    " Italy": 17,
    " Poland": 18,
    " Jamaica": 19,
    " Vietnam": 20,
    " Mexico": 21,
    " Portugal": 22,
    " Ireland": 23,
    " France": 24,
    " Dominican-Republic": 25,
    " Laos": 27,
    " Ecuador": 28,
    " Taiwan": 29,
    " Haiti": 30,
    " Columbia": 31,
    " Hungary": 32,
    " Guatemala": 33,
    " Nicaragua": 34,
    " Scotland": 35,
    " Thailand": 36,
    " Yugoslavia": 37,
    " El-Salvador": 38,
    " Trinadad&Tobago": 39,
    " Peru": 40,
    " Hong": 41,
    " Holand-Netherlands": 42
}

#Mapping replacement
df_global_train['workclass'] = df_global_train['workclass'].map(map_workclass)
df_global_train['education'] = df_global_train['education'].map(map_edu)
df_global_train['marital-status'] = df_global_train['marital-status'].map(map_marry)
df_global_train['occupation'] = df_global_train['occupation'].map(map_job)
df_global_train['relationship'] = df_global_train['relationship'].map(map_relation)
df_global_train['race'] = df_global_train['race'].map(map_race)
df_global_train['sex'] = df_global_train['sex'].map(map_sex)
df_global_train['native-country'] = df_global_train['native-country'].map(map_country)

df_global_test['workclass'] = df_global_test['workclass'].map(map_workclass)
df_global_test['education'] = df_global_test['education'].map(map_edu)
df_global_test['marital-status'] = df_global_test['marital-status'].map(map_marry)
df_global_test['occupation'] = df_global_test['occupation'].map(map_job)
df_global_test['relationship'] = df_global_test['relationship'].map(map_relation)
df_global_test['race'] = df_global_test['race'].map(map_race)
df_global_test['sex'] = df_global_test['sex'].map(map_sex)
df_global_test['native-country'] = df_global_test['native-country'].map(map_country)

'''
Process the target prediction by using One-Hot Coding
Then, the result will be like:
If the income is over 50K, the rusult is 1; otherwise, it is 0.
'''

###Training set
income_train_raw = df_global_train['salary']
income_train_raw.unique()

y_train = income_train_raw.apply(lambda x: int(x==" >50K"))


###Testing set
income_test_raw = df_global_test['salary']
income_test_raw.unique()


y_test = income_test_raw.apply(lambda x: int(x==" >50K"))


############Models############
'''
Logistic Regression:
1. Original;
2. Optimized
'''
#### 1. Normal Model
print("***********Normal Model of Logistic Regression**********")
X_df_global_train, X_df_global_test = df_global_train.iloc[:, :-1], df_global_test.iloc[:, :-1]
#Train the model
lr_normal = linear_model.LogisticRegression()

print("Training...")
performance_global = lr_normal.fit(X_df_global_train, y_train)

#Predict and evaluate the model
print("Testing...\n")
prediction_global = lr_normal.predict(X_df_global_test)


## Intercept and coefficiency
intercept_global = performance_global.intercept_
coef_global = performance_global.coef_

print("Coef: {0}, \nIntercept:{1}".format(coef_global,intercept_global))

####2. Optimized model
print("***********Optimized Model of Logistic Regression**********")
lr_op = linear_model.LogisticRegression(penalty="l1", solver="liblinear", tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1)
# lr_op = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
#Train the model
print("Training...")
performance_glo_op = lr_op.fit(X_df_global_train, y_train)

#Predict and evaluate the model
print('Testing...\n')
prediction_glo_op = lr_op.predict(X_df_global_test)

## Intercept and coefficiency
intercept_glo_op = performance_glo_op.intercept_
coef_glo_op = performance_glo_op.coef_

print("Coef: {0}, \nIntercept:{1}".format(coef_glo_op, intercept_glo_op))


#Data of prediction result
#Prediction Ratio
pre_glo_Counter = Counter(prediction_global)
pre_glo_op_Counter = Counter(prediction_glo_op)

# Accuracy
as_global = round(accuracy_score(y_test,prediction_global), 4)
as_glo_op = round(accuracy_score(y_test,prediction_glo_op), 4)

#Precision score
ps_global = round(precision_score(y_test, prediction_global), 4)
ps_glo_op = round(precision_score(y_test, prediction_glo_op), 4)

#Recall score
rs_global = round(recall_score(y_test, prediction_global, average='binary', pos_label=1), 4)
rs_glo_op = round(recall_score(y_test, prediction_glo_op, average='binary', pos_label=1), 4)

#f1-score
fs_global = round(f1_score(y_test, prediction_global, average='binary', pos_label=1), 4)
fs_glo_op = round(f1_score(y_test, prediction_glo_op, average='binary', pos_label=1), 4)

#classification_report
report_glo_normal = classification_report(y_test, prediction_global)
report_glo_op = classification_report(y_test, prediction_glo_op)

#cross_val
cv_glo_normal = cross_val_score(lr_normal, X_df_global_train, y_train)
cv_glo_op = cross_val_score(lr_op, X_df_global_train, y_train)

##Report Printing
print("The data and report for normal model of Logistic Regression...\n")
print("Ratio of the prediction: \n '>50K':%.2f%%, '<=50K':%.2f%%"% (pre_glo_Counter.get(1) / len(y_test) * 100, pre_glo_Counter.get(0) / len(y_test) * 100))
print("Score of the model:{0}".format(cv_glo_normal.mean()))
print(report_glo_normal)
print('Data of the prediction: 1. Accuracy Score: {0}, 2.Precision Score:{1}, 3.recall_score:{2}, 4.f1_score:{3}.\n'.format(as_global, ps_global, rs_global, fs_global))

print("The data and report for optimized model of Logistic Regression...\n")
print("Ratio of the prediction: \n '>50K':%.2f%%, '<=50K':%.2f%%"% (pre_glo_op_Counter.get(1) / len(y_test) * 100, pre_glo_op_Counter.get(0) / len(y_test) * 100))
print("Score of the model:{0}".format(cv_glo_op.mean()))
print(report_glo_op)
print('Data of the prediction: 1. Accuracy Score: {0}, 2.Precision Score:{1}, 3.recall_score:{2}, 4.f1_score:{3}.'.format(as_glo_op, ps_glo_op, rs_glo_op, fs_glo_op))




########## Decision Tree
# Utilize the minmaxscaler into numeric data (here: age, final-wt, capital-gain, capital-loss, hours-per-week)
numerical = ['age', 'final-wt', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = MinMaxScaler()
df_global_train[numerical] = scaler.fit_transform(df_global_train[numerical])
df_global_test[numerical] = scaler.fit_transform(df_global_test[numerical])
# Build tree model
print("***********Desision Tree**********")
#Split Data
X_glo_tree_train, X_glo_tree_test = df_global_train.iloc[:, :-1], df_global_test.iloc[:, :-1]

#Model Fit
print("Training...")
clf_global = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=15, random_state=4)
clf_global.fit(X_glo_tree_train, y_train)

#Stucture of the decision tree
plt.figure(figsize=(12,9),dpi=80)
tree_strucure = tree.export_text(clf_global, feature_names=X_glo_tree_train.columns.tolist())
with open('tree_structure.txt', 'w', encoding='utf-8') as f:
    f.write(tree_strucure)
    f.close()
# print("The structure of the decision tree:", tree_strucure)

#Predict
print("Testing...\n")
prediction_tree = clf_global.predict(X_glo_tree_test)



########## Random Forest
print("***********Random Forest**********")
#Split the data
X_glo_forest_train, X_glo_forest_test = df_global_train.iloc[:, :-1], df_global_test.iloc[:, :-1]

print("Training...")
#Build the model
clf_forest = RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=15, max_features="sqrt")
clf_forest.fit(X_glo_forest_train, y_train)
print("Testing..\n")
prediction_forest = clf_forest.predict(X_glo_forest_test)


######Data of prediction result
#Ratio of the prediction
pre_tree_Counter = Counter(prediction_tree)
pre_forest_Counter = Counter(prediction_forest)
#Accuracy Score
as_global_tree = round(accuracy_score(y_test, prediction_tree), 4)
as_glo_forest = round(accuracy_score(y_test, prediction_forest), 4)

#Precision score
ps_global_tree = round(precision_score(y_test, prediction_tree), 4)
ps_glo_forest = round(precision_score(y_test, prediction_forest), 4)

#Recall score
rs_global_tree = round(recall_score(y_test, prediction_tree, average='binary', pos_label=1), 4)
rs_glo_forest = round(recall_score(y_test, prediction_forest, average='binary', pos_label=1), 4)

#f1-score
fs_global_tree = round(f1_score(y_test, prediction_tree, average='binary', pos_label=1), 4)
fs_glo_forest = round(f1_score(y_test, prediction_forest, average='binary', pos_label=1), 4)

#classification_report
report_glo_tree = classification_report(y_test, prediction_tree)
report_glo_forest = classification_report(y_test, prediction_forest)

#cross_val
cv_tree = cross_val_score(clf_global, X_glo_tree_train, y_train)
cv_forest = cross_val_score(clf_forest, X_glo_forest_train, y_train)

##Report Printing
print("The data and report for Decision Tree...\n")
print("Ratio of the prediction: \n '>50K':%.2f%%, '<=50K':%.2f%%"% (pre_tree_Counter.get(1) / len(y_test) * 100, pre_tree_Counter.get(0) / len(y_test) * 100))
print("Score of the model:{0}".format(cv_tree.mean()))
print(report_glo_tree)
print('Data of the prediction: 1. Accuracy Score: {0}, 2.Precision Score:{1}, 3.recall_score:{2}, 4.f1_score:{3}.\n'.format(as_global_tree, ps_global_tree, rs_global_tree, fs_global_tree))

print("The data and report for Random Forest...\n")
print("Ratio of the prediction: \n '>50K':%.2f%%, '<=50K':%.2f%%"% (pre_forest_Counter.get(1) / len(y_test) * 100, pre_forest_Counter.get(0) / len(y_test) * 100))
print("Score of the model:{0}".format(cv_forest.mean()))
print(report_glo_forest)
print('Data of the prediction: 1. Accuracy Score: {0}, 2.Precision Score:{1}, 3.recall_score:{2}, 4.f1_score:{3}.'.format(as_glo_forest, ps_glo_forest, rs_glo_forest, fs_glo_forest))

# Data analytics for test dataset
# The number of people earning more than 50k (Original data)
more_than_50k_test = len([salary for salary in df_global_test['salary'] if salary == " >50K"])
less_than_50k_test = len(df_global_test) - more_than_50k_test
ratio_more_than_50k_test = more_than_50k_test/(more_than_50k_test + less_than_50k_test)
ratio_less_than_50k_test = less_than_50k_test/(more_than_50k_test + less_than_50k_test)
salary_list_test = [more_than_50k_test, less_than_50k_test]
ratio_list_test = [ratio_more_than_50k_test, ratio_less_than_50k_test]

#Draw the graph
#Bar
ax1 = plt.figure(figsize=(10, 10))
plt.suptitle("Income distribution of test dataset data", fontsize=15)
plt.subplot(1, 2, 1)
plt.title("Rumber of the distribution of people's income", fontsize=15)
plt.xlabel('Category', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=10)
fig_salary = plt.bar(['more than 50k', 'less than 50k'], salary_list_test, color=['r', 'b'])
autolabel(fig_salary)

#Pie
plt.subplot(1, 2, 2)
plt.title("Ratio of the distribution of people's income", fontsize=15)
plt.pie(ratio_list_test, labels=['more than 50k', 'less than 50k'], autopct="%.2f%%")
plt.savefig("Income distribution of test data.png")
ax1.show()