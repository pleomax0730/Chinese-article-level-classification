from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from pandas_ml import ConfusionMatrix
import sklearn.metrics
from sklearn.metrics import classification_report
import sklearn.naive_bayes as sk_bayes
import sklearn.svm as sk_svm
import mlflow
import numpy as np
import pandas as pd
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score


def self_test(xkernel, testset):
    return xkernel.predict(testset)


def train_model(balance_data, defaultype='decisiontree'):
    # print('settings.LEVEL_PREDICTION_DATASET', settings.LEVEL_PREDICTION_DATASET)
    # balance_data = pd.read_csv(settings.LEVEL_PREDICTION_DATASET, sep=',', header=0)

    # Lv.1,Pondlet_0102_001910,83,16,5,3,0,0,2,0,2
    # X = balance_data.values[:, 2:]   # old
    X = balance_data.values[:, 2:] # old
    Y = balance_data.values[:, 0:2]  # old
    # Y = balance_data.values[:, 0:1]
    print('X', X)
    print('Y', Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    mlflow.log_param("train_records", '%s' % len(X_train))
    mlflow.log_param("valid_records", '%s' % len(y_test))

    """
    for custom label: y_test[:, 1]
    for target class: y_test[:, 0]
    """

    mlflow.sklearn.autolog()
    if defaultype == 'XGBClassifier':
        model = XGBClassifier()
    elif defaultype == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif defaultype == "AdaBoostClassifier":
        model = AdaBoostClassifier()
    elif defaultype == "svm":
        model = sk_svm.SVC(C=1.0, kernel='rbf', gamma='auto')
    elif defaultype == "BaggingClassifier":
        model = BaggingClassifier()
    elif defaultype == "LGBMClassifier":
        model = LGBMClassifier()
    elif defaultype == "VotingClassifier":
        estimators = [('rf', RandomForestClassifier()),
                      ('dt', DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=10, min_samples_leaf=5)), 
                      ('bg', BaggingClassifier()), ('sg', XGBClassifier()), ('lg', LGBMClassifier())]
        model = VotingClassifier(estimators=estimators, voting='hard')
    # elif defaultype == "StackingClassifier":
    #    estimators = [('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier())]
    #    model = StackingClassifier(estimators=estimators)
    elif defaultype == 'naive_bayes':
        model = sk_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    # DecisionTreeClassifier
    else:
        model = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=10, min_samples_leaf=5)


    scores = cross_val_score(model, X, Y[:, 0], cv=6, scoring='accuracy')
    mlflow.log_param(".r__model name", '%s' % defaultype)
    mlflow.log_param(".r__K fold score", '%s' % scores)
    print('K fold score', scores)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
    mlflow.log_param(".r__K fold score avg", "%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
    '''
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X, Y):
        # create model
        # Fit the model
        model.fit(X[train], Y[train])
        # evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
    '''

    time.sleep(5)

    model.fit(X_train, y_train[:, 0])
    model_score = model.score(X_train, y_train[:, 0])
    """ Calculate the misclassified """
    missed_index = np.where(y_test[:, 0] != model.predict(X_test))
    instance = X_test[missed_index]
    label = y_test[missed_index]
    predicted = model.predict(X_test[missed_index])
    misclassified = list(zip(predicted.tolist(), label.tolist(), instance.tolist()))
    y_pred = model.predict(X_test)

    # [['Lv.3' 'Gloss_00017']
    y_true = [x[0] for x in y_test]

    dt = {}
    dt['kernel'] = model
    dt['accuracy(train)'] = model_score * 100
    dt['accuracy(test)'] = accuracy_score(y_test[:, 0], self_test(model, X_test)) * 100
    dt['misclassified'] = misclassified

    # label : [['Lv.3' 'Gloss_00017']
    # y_true = [x[0] for x in label]
    # y_pred = predicted
    data = {'y_Actual': y_true, 'y_Predicted': y_pred}
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    Confusion_Matrix = ConfusionMatrix(df['y_Actual'], df['y_Predicted'])
    # 印出混淆矩陣, F1 score,
    Confusion_Matrix.print_stats()
    # 儲存成混謠矩陣
    rs = Confusion_Matrix._str_stats()
    # F1 score
    testset_f1_report = classification_report(y_true, y_pred, digits=6)
    print(testset_f1_report)
    with open('./reports/confustion_martix_trainmodel-validsets.txt', 'w') as f:
        f.write(testset_f1_report)
        f.write(rs)
    return dt
