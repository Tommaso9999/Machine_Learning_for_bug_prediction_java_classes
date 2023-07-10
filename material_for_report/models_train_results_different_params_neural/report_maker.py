import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import inspect

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

        boxplot_data = [classifier_df['F-score']]
        labels = ['F-score']

        ax.set_xlabel('Metric')
        ax.set_ylabel('Values')
        ax.set_title(classifier + " results")
        ax.set_ylim(0, 1.1)
        ax.grid(False)

        std_df = round(classifier_df['F-score'].std(), 4)
        mean_df = classifier_df['F-score'].mean()
        distance = (std_df - mean_df) / ax.get_ylim()[1]  # Calculate the normalized distance

        ax.boxplot(boxplot_data, patch_artist=True, boxprops=boxprops, medianprops=medianprops, labels=labels, showmeans=True)

        positions = [1]
        ax.plot(positions, mean_df - std_df, 'ro')  # Red dot below the mean
        ax.plot(positions, mean_df + std_df, 'ro')  # Red dot above the mean

        ax.text(1.1, mean_df - std_df*0.95,"-std "+ str(f'{std_df:.2f}'), ha='center', va='top', fontsize=8)
        ax.text(1.1, mean_df + std_df*0.95,"+std "+ str(f'{std_df:.2f}'), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save the plot as a PNG file in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    save_path = os.path.join(script_dir, 'results_plot_fscore.png')
    plt.savefig(save_path)

    plt.show()

if __name__ == "__main__":
    csv = r'models_train_results/results.csv'
    evaluate(csv)