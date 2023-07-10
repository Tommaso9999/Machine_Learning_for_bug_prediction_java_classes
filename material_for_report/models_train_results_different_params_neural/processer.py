
import pandas as pd


input_file = pd.read_csv(r'results.csv')
grouped_cols = input_file.groupby(['Classifier Name', 'Hyperparameters'])
output_file = grouped_cols.mean().reset_index()

grouped = output_file.groupby('Classifier Name')

print(output_file)



for name, group in grouped:

    group.to_csv(f'{name}_data.csv', index=False)




