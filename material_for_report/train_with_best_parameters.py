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

def train_classifiers(csv, folder_name):
    data = pd.read_csv(csv)

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    features = data.iloc[:, 2:9]
    target = data.iloc[:, -1]


    classifiers = [
        (tree.DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None, splitter="best"), 'Decision Tree'),
        (GaussianNB(priors=None, var_smoothing=1e-9), 'Naive Bayes'),
        (svm.SVC(kernel='linear', C=0.1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='scale', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False), 'SVM'),
        (make_pipeline(StandardScaler(copy=True, with_mean=True, with_std=True), MLPClassifier(
            activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
            beta_2=0.999, early_stopping=False, epsilon=1e-08,
            hidden_layer_sizes=(100,), learning_rate='constant',
            learning_rate_init=0.001, max_fun=15000, max_iter=2000,
            momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
            power_t=0.5, random_state=None, shuffle=True, solver='adam',
            tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False
        )), 'MLP'),
        (RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None, criterion='gini', max_depth=None, max_features='sqrt', max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None, oob_score=False, random_state=None, verbose=0, warm_start=False), 'Random Forest'),
        (DummyClassifier(strategy='constant', constant=True), 'Biased Classifier')
    ]


    results = []


    for i in range(20):

        kf = KFold(n_splits=5, shuffle=True)  # Initialize 5-fold cross-validation

        for train_index, test_index in kf.split(features):  # Iterate over the cross-validation splits
            train_features, test_features = features.iloc[train_index], features.iloc[test_index]
            train_target, test_target = target.iloc[train_index], target.iloc[test_index]

            fold_results = []

            for classifier, name in classifiers:
                classifier.fit(train_features, train_target)
                predictions = classifier.predict(test_features)
                metrics = precision_recall_fscore_support(test_target, predictions, average='binary', zero_division=0)
                result = {
                    'Classifier Name': name,
                    'HyperParameters':"-",
                    'Precision': metrics[0],
                    'Recall': metrics[1],
                    'F-score': metrics[2],
                }
                fold_results.append(result)

            results.extend(fold_results)

        metrics_df = pd.DataFrame(results)
        metrics_df.to_csv(os.path.join(folder_name, 'results.csv'))

if __name__ == "__main__":
    csv = f'/Users/tommasoverzegnassi/Desktop/project-02-bug-prediction-Tommaso9999-main/feature_vectors/feature_vectors_labeled.csv'
    folder_name = r'models_train_results_best_configuration'
    train_classifiers(csv, folder_name)

