from collections import Counter

from numpy import mean
from pandas import read_csv
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve


def load_dataset(full_path):
    dataframe = read_csv(full_path, na_values='?')
    # dataframe.info(verbose=True)
    dataframe = dataframe.drop(columns=['Unnamed: 0', 'X'], axis=1)
    X = dataframe.loc[:, 'age':'hours.per.week']
    y = dataframe['income']
    num_ix = X.select_dtypes(include=['int64']).columns
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    y = LabelEncoder().fit_transform(y)
    return X, y, cat_ix, num_ix


def sum_lable(target):
    # summarize the class distribution
    print(target)
    counter = Counter(target)
    for k, v in counter.items():
        per = v / len(target) * 100
        print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))


def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


def acc(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return float(correct) / y_true.shape[0]


def _plot_roc_curve(fpr, tpr, auc_):
    plt.title('Receiver Operating Characteristic - ROC')
    plt.plot(fpr, tpr, 'b-',label = 'AUC = %0.2f' % auc_)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


# From data
X, y, cat_ix, num_ix = load_dataset("C:\\Users\\MSI\Desktop\\vnpt-project\\python\\data\\data_adult.csv")
sum_lable(y)
cat_ix = cat_ix.drop(['race','gender'])
print(cat_ix)

X = pd.concat([X, pd.get_dummies(X[cat_ix])], axis=1)
X = X.drop(['workclass', 'education', 'marital.status', 'race', 'gender'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


print(X_train.values)
print(X_train.shape)
print(y_train.shape)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.values)

# model = RandomForestClassifier(n_estimators=50)
# model = model.fit(X_train, y_train)
# scores = evaluate_model(X_train, y_train, model)
# y_pre = model.predict(X_test)
#
# print('Scores: ', mean(scores))
# print('Accuracy: ', accuracy_score(y_test, y_pre))
# print('F1-score: ', f1_score(y_test, y_pre))

# model = RandomForestClassifier(n_estimators=100,
#                                class_weight={0: 0.4,
#                                               1: 0.6})

model1 = RandomForestClassifier(n_estimators=50,
                                max_depth=10,
                                min_samples_split=400,
                                class_weight={
                                    0: 0.45,
                                    1: 0.55
                                },
                                max_features=10)

# model2 = RandomForestClassifier(n_estimators=100,
#                                 max_depth=8,
#                                 min_samples_split=100,
#                                 class_weight="balanced",
#                                 max_features="auto")
#
# model3 = RandomForestClassifier(n_estimators=200,
#                                 max_depth=5,
#                                 min_samples_split=200,
#                                 class_weight="balanced",
#                                 max_features="sqrt")

model1 = model1.fit(X_train, y_train)
scores1 = evaluate_model(X_train, y_train, model1)
y_pre1 = model1.predict(X_test)
fpr1, tpr1, thres = metrics.roc_curve(y_test, y_pre1)
auc_ = auc(fpr1,tpr1)
print('Scores1: ', mean(scores1))
print('Accuracy1: ', accuracy_score(y_test, y_pre1))
print('F1-score1: ', f1_score(y_test, y_pre1))
print('AUC1: ', auc(fpr1, tpr1))
_plot_roc_curve(fpr1, tpr1, auc_)
# plot_roc_curve(model1, X_test, y_test)
plt.show()

# _plot_roc_curve(fpr, tpr, thres)

# model2 = model2.fit(X_train, y_train)
# scores2 = evaluate_model(X_train, y_train, model2)
# y_pre2 = model2.predict(X_test)
# fpr2, tpr2, thres2 = metrics.roc_curve(y_test, y_pre1)
# print('Scores2: ', mean(scores2))
# print('Accuracy2: ', accuracy_score(y_test, y_pre2))
# print('F1-score2: ', f1_score(y_test, y_pre2))
# print('AUC2: ', auc(fpr2, tpr2))
#
# model3 = model3.fit(X_train, y_train)
# scores3 = evaluate_model(X_train, y_train, model3)
# y_pre3 = model3.predict(X_test)
# fpr3, tpr3, thres3 = metrics.roc_curve(y_test, y_pre1)
# print('Scores3: ', mean(scores3))
# print('Accuracy3: ', accuracy_score(y_test, y_pre3))
# print('F1-score3: ', f1_score(y_test, y_pre3))
# print('AUC3: ', auc(fpr3, tpr3))
