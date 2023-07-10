import sys
sys.path.append('../PROJECT-02-BUG-PREDICTION-TOMMASO9999-MAIN')

import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.dummy import DummyClassifier
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import decimal

params_tree = [2, 3, 4, 5, 100]
params_svm = [1, 2, 5, 10, 100]
params_mlp = [(50, 50), (100, 100),(150, 150), (200, 200), (250, 250)]
params_rf = [5, 10, 50, 100, 200]
params_naive = [0, 1, 2, 3, 4]


def train_classifiers(csv, folder_name):

    data = pd.read_csv(csv)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    features = data.iloc[:, 2:9]
    target = data.iloc[:, -1]

    results = []
    for i in range(0, 5):

        classifiers = [
            (tree.DecisionTreeClassifier(max_depth=params_tree[i]), 'Decision Tree'),
            (GaussianNB(var_smoothing=params_naive[i]), 'Naive Bayes'),
            (svm.SVC(kernel='linear', C=params_svm[i]), 'SVM'),
            (make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=params_mlp[i], max_iter=2000)), 'MLP'),
            (RandomForestClassifier(n_estimators=params_rf[i]), 'Random Forest'),
            (DummyClassifier(strategy='constant', constant=True), 'Biased Classifier')
        ]


        for j in range(20):

            kf = KFold(n_splits=5, shuffle=True)  

            for train_index, test_index in kf.split(features):  
                train_features, test_features = features.iloc[train_index], features.iloc[test_index]
                train_target, test_target = target.iloc[train_index], target.iloc[test_index]

                fold_results = []

                for classifier, name in classifiers:
                    classifier.fit(train_features, train_target)
                    predictions = classifier.predict(test_features)
                    metrics = precision_recall_fscore_support(test_target, predictions, average='binary', zero_division=0)
                    result = {
                        'Classifier Name': name,
                        'Hyperparameters': params_tree[i] if classifier.__class__ == tree.DecisionTreeClassifier else
                          params_svm[i] if classifier.__class__ == svm.SVC else 
                          params_rf[i] if classifier.__class__== RandomForestClassifier else 
                          params_naive[i] if classifier.__class__ == GaussianNB else 
                          "Dummy" if classifier.__class__ == DummyClassifier else 
                          params_mlp[i]
                          ,
                        'Precision': metrics[0],
                        'Recall': metrics[1],
                        'F-score': metrics[2],
                    }
                    fold_results.append(result)

                results.extend(fold_results)

        metrics_df = pd.DataFrame(results)
        metrics_df.to_csv(os.path.join(folder_name, 'results_different_params.csv'))


if __name__ == "__main__":
    csv = f'feature_vectors/feature_vectors_labeled.csv'
    folder_name = r'models_train_results_different_params'
    train_classifiers(csv, folder_name)





