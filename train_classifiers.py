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
        (tree.DecisionTreeClassifier(max_depth=5), 'Decision Tree'),
        (GaussianNB(), 'Naive Bayes'),
        (svm.SVC(kernel='linear', C=1), 'SVM'),
        (make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=2000, random_state=42)), 'MLP'),
        (RandomForestClassifier(n_estimators=50), 'Random Forest'),
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
                    'Precision': metrics[0],
                    'Recall': metrics[1],
                    'F-score': metrics[2],
                }
                fold_results.append(result)

            results.extend(fold_results)

        metrics_df = pd.DataFrame(results)
        metrics_df.to_csv(os.path.join(folder_name, 'results.csv'))

if __name__ == "__main__":
    csv = r'feature_vectors/feature_vectors_labeled.csv'
    folder_name = r'models_train_results'
    train_classifiers(csv, folder_name)





