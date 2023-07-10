import pandas as pd



df = pd.read_csv(r'results_grid_search.csv')

max_detriti_index = df.groupby(['Classifier Name'])['F-score']

result = df.loc[max_detriti_index]


result.to_csv(r'best_parameters_gridsearch_2.csv')

print(result)
