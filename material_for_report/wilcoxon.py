import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os 


folder_name = r"results_of_classifiers_against_biased_classifier"
if not os.path.exists(folder_name):
        os.mkdir(folder_name)



classifiers = ["Decision Tree", "Naive Bayes", "SVM", "MLP", "Random Forest", "Biased Classifier"]

def evaluate(csv):
   
    data = pd.read_csv(csv)
   
    df = pd.DataFrame(data, columns=['Classifier Name', 'Precision', 'Recall', 'F-score'])
    RD_df = df[df['Classifier Name'] == "Random Forest"]
    MLP_df = df[df['Classifier Name'] == "MLP"]


    _, p_value = stats.wilcoxon(RD_df['F-score'], MLP_df['F-score'])

    print(p_value)

 



if __name__ == "__main__":
    csv = r'models_train_results/results.csv'
    evaluate(csv)
