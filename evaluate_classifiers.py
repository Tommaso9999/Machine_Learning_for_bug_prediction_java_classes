import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os 


folder_name = r"results_of_classifiers_against_biased_classifier"
if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def evaluate(csv):
    classifiers = ["Decision Tree", "Naive Bayes", "SVM", "MLP", "Random Forest", "Biased Classifier"]
    data = pd.read_csv(csv)
    result_df = pd.DataFrame(columns=['Classifier'])  

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))  

    for classifier, ax in zip(classifiers, axes.flatten()):
        df = pd.DataFrame(data, columns=['Classifier Name', 'Precision', 'Recall', 'F-score'])
        classifier_df = df[df['Classifier Name'] == classifier]

        boxprops = dict(facecolor='skyblue', edgecolor='black')  
        medianprops = dict(color='red')  


        classifier_df.boxplot(column=['Precision', 'Recall', 'F-score'], patch_artist=True,
                              boxprops=boxprops, medianprops=medianprops, ax=ax)

        ax.set_xlabel('Metric')
        ax.set_ylabel('Values')
        ax.set_title(classifier + " results")
        ax.set_ylim(0, 1.1)
        ax.grid(False)

        biased_classifier_df = df[df['Classifier Name'] == "Biased Classifier"]

        median_df = round(classifier_df['F-score'].median(),4)
        median_biased = round(biased_classifier_df['F-score'].median(),4)
        median_difference = round(median_df-median_biased, 4)

        mean_df = round(classifier_df['F-score'].mean(),4)
        mean_biased = round(biased_classifier_df['F-score'].mean(),4)
        mean_difference = mean_df-mean_biased

        p_value="-"

        if classifier != "Biased Classifier":
        
            _, p_value = stats.wilcoxon(classifier_df['F-score'], biased_classifier_df['F-score'])


        result_df = result_df.append({'Classifier': classifier, 
                                          'mean F score classifier':mean_df, 'classifier mean F - biased mean F': mean_difference,
                                          'Wilcoxon test p-value': p_value}, ignore_index=True)

    plt.tight_layout()  
    plt.savefig(r'results_of_classifiers_against_biased_classifier/classifier_results.png')  
    plt.show()

    result_df.to_csv(r'results_of_classifiers_against_biased_classifier/classifiers_vs_biased_classifier_results.csv')  



if __name__ == "__main__":
    csv = r'models_train_results/results.csv'
    evaluate(csv)
