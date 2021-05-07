from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, KMeansSMOTE
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.under_sampling import RandomUnderSampler
from numpy import mean
from pandas import read_csv
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier


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
    plt.plot(fpr, tpr, 'b-', label='AUC = %0.2f' % auc_)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


# From data
X, y, cat_ix, num_ix = load_dataset("C:\\Users\\MSI\Desktop\\vnpt-project\\python\\data\\data_adult.csv")
sum_lable(y)
cat_ix = cat_ix.drop(['race', 'gender'])
print(cat_ix)

X = pd.concat([X, pd.get_dummies(X[cat_ix])], axis=1)
X = X.drop(['workclass', 'education', 'marital.status', 'race', 'gender'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.values)
print(X_train)
print(y_train)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.values)

model1 = XGBClassifier(n_estimators=50,
                       learning_rate=0.1,
                       max_depth=10,
                       min_samples_split=400,
                       max_features=10
                       )

print('----------------------------------------------------')
print(Counter(y_train))
# oversample = KMeansSMOTE(sampling_strategy=0.5, k_neighbors=3)
# X_train, y_train = oversample.fit_resample(X_train, y_train)
print(Counter(y_train))
print(X_train)
print(y_train)
model1 = model1.fit(X_train, y_train)
scores1 = evaluate_model(X_train, y_train, model1)
y_pre1 = model1.predict(X_test)
fpr, tpr, thres = metrics.roc_curve(y_test, y_pre1)
auc_ = auc(fpr, tpr)

print('Scores: ', mean(scores1))
print('Accuracy: ', accuracy_score(y_test, y_pre1))
print('F1-score: ', f1_score(y_test, y_pre1))
print('AUC: ', auc_)
