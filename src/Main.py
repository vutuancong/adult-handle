import pandas as pd
from matplotlib import pyplot
from numpy import mean, std
from pandas import read_csv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_dataset(full_path):
    dataframe = read_csv(full_path, na_values='?')
    dataframe.info(verbose=True)
    dataframe = dataframe.drop(columns=['Unnamed: 0', 'X'], axis=1)
    X = dataframe.loc[:, 'age':'hours.per.week']
    y = dataframe['income']
    num_ix = X.select_dtypes(include=['int64']).columns
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    y = LabelEncoder().fit_transform(y)
    # X, y = dataframe[[2, 3, 4, 5, 6, 7, 8, 9]], dataframe[10]
    # cat_ix = X[[2, 5, 9]].columns
    # num_ix = X[[3, 4, 6, 7, 8]].columns
    # y = LabelEncoder().fit_transform(y)
    # print("cat_ix: ", cat_ix)
    # print("num_ix: ", num_ix)
    return X, y, cat_ix, num_ix


def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# define models to test
def get_models():
    models, names = list(), list()
    # CART
    models.append(DecisionTreeClassifier())
    names.append('CART')
    # SVM
    models.append(SVC(gamma='scale'))
    names.append('SVM')
    # Bagging
    models.append(BaggingClassifier(n_estimators=100))
    names.append('BAG')
    # RF
    models.append(RandomForestClassifier(n_estimators=100))
    names.append('RF')
    # GBM
    models.append(GradientBoostingClassifier(n_estimators=100))
    names.append('GBM')
    return models, names

# From data
X, y, cat_ix, num_ix = load_dataset("/data/data_adult.csv");
X = pd.concat([X, pd.get_dummies(X[cat_ix])], axis=1)
X = X.drop(['workclass', 'education', 'marital.status', 'race', 'gender'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GradientBoostingClassifier()
model.fit()


# models, names = get_models()
# results = list()
# for i in range(len(models)):
#     # define steps
#     steps = [('c', OneHotEncoder(handle_unknown='ignore'), cat_ix), ('n', MinMaxScaler(), num_ix)]
#     # one hot encode categorical, normalize numerical
#     ct = ColumnTransformer(steps)
#     # wrap the model i a pipeline
#     pipeline = Pipeline(steps=[('t', ct), ('m', models[i])])
#     print(models[i])
#     # evaluate the model and store results
#     scores = evaluate_model(X, y, pipeline)
#     # print(pipeline)
#     results.append(scores)
#     # summarize performance
#     print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
# # plot the results
# pyplot.boxplot(results, labels=names, showmeans=True)
# pyplot.show()


# model = DummyClassifier(strategy='most_frequent')
# model1 = RandomForestClassifier();
# # evaluate the model
# scores = evaluate_model(X, y, model1)
# # summarize performance
# print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# model = GradientBoostingClassifier(n_estimators=100)
# # one hot encode categorical, normalize numerical
# ct = ColumnTransformer([('c', OneHotEncoder(), cat_ix), ('n', MinMaxScaler(), num_ix)])
# # define the pipeline
# pipeline = Pipeline(steps=[('t', ct), ('m', model)])
# # fit the model
# pipeline.fit(X, y)

# #one hot transfer
# data_cat_ix = X[cat_ix]
# data_cat_ix = data_cat_ix.values
# enc = preprocessing.OneHotEncoder()
# enc.fit(data_cat_ix)
# one_hot_cat_ix = enc.transform(data_cat_ix).toarray()
# print(one_hot_cat_ix.shape)
#
# #min max scalse
# data_min_max = X[num_ix]
# data_min_max = data_min_max.values
# scaler = MinMaxScaler()
# scaler.fit(data_min_max)
# data_min_max = scaler.transform(data_min_max)
# print(data_min_max.shape)