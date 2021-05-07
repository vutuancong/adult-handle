from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import mean
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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


def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# From data
X, y, cat_ix, num_ix = load_dataset("/data/data_adult.csv");
X = pd.concat([X, pd.get_dummies(X[cat_ix])], axis=1)
X = X.drop(['workclass', 'education', 'marital.status', 'race', 'gender'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.values)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.values)
rdFrt = RandomForestClassifier(n_estimators=100)
rdFrt.fit(X_train, y_train)
scores = evaluate_model(X_train, y_train, rdFrt)

arr = np.stack((X.columns.array, rdFrt.feature_importances_), axis=1)
dict_feature = {}
for index in arr:
    if "_" not in index[0]:
        dict_feature[index[0]] = index[1]
    else:
        key = index[0].split('_')[0]
        if key in dict_feature:
            value = dict_feature[key]
            value = value + index[1]
            dict_feature[key] = value
        else:
            dict_feature[key] = index[1]

dict_feature = OrderedDict(sorted(dict_feature.items(), key=lambda x: x[1], reverse=True))
print('Scores: ', mean(scores))
print(dict_feature)

x_axis = list(dict_feature.keys())
y_axis = list(dict_feature.values())
plt.bar(x_axis, y_axis)
plt.show()

# print(sum(y_axis))
# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf.fit(X_train, y_train)
# forest = RandomForestClassifier(random_state=0)
# forest.fit(X_train, y_train)

# print(X_test)

# start_time = time.time()
# result = permutation_importance(
#     forest, X_test.todense(), y_test, n_repeats=10, random_state=42, n_jobs=2)
# elapsed_time = time.time() - start_time
# print(f"Elapsed time to compute the importances: "
#       f"{elapsed_time:.3f} seconds")
#
# import matplotlib.pyplot as plt
#
# # print(feature_names)
# # print(result.importances_mean)
# forest_importances = pd.Series(result.importances_mean, index=feature_names)
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
# ax.set_title("Feature importances using permutation on full model")
# ax.set_ylabel("Mean accuracy decrease")
# fig.tight_layout()
# plt.show()


# feature_names = [f'feature {i}' for i in X.columns]
# steps = [('c', OneHotEncoder(handle_unknown='ignore'), cat_ix), ('n', MinMaxScaler(), num_ix)]
# ct = ColumnTransformer(steps)
# X_train = ct.fit_transform(X_train)
# # steps = [('c', OneHotEncoder(handle_unknown='ignore'), cat_ix), ('n', MinMaxScaler(), num_ix)]
# # ct = ColumnTransformer(steps)
# # X_test = ct.fit_transform(X_test)

# pipe1 = make_pipeline(StandardScaler(), rdFrt)
# acc_rdf = cross_val_score(rdFrt, X_train, y_train, scoring='accuracy', cv=5).mean()
# model = pipe1.steps[1][1]
# print('Accuracy theo random forest:', acc_rdf)
# print('Feature name: ', X.columns.array)
# print('Feature importance: ')
# print(rdFrt.feature_importances_)
# print(mean(scores))
