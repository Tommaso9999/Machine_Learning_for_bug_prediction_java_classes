import os
import re
import pandas as pd


def label(csv):
    df = pd.read_csv(csv)

    directory = r'resources/modified_classes'
    class_names = []

    for filename in os.listdir(directory):
        if filename.endswith(".src"):
            with open(os.path.join(directory, filename), "r") as f:
                for line in f:
                    class_name = line.split(".")[-1].strip()
                    class_names.append(class_name)


    dfnew = df
    dfnew['buggy'] = dfnew['class_name'].apply(lambda x: 1 if x in class_names else 0)

    dfnew = dfnew.drop(df.columns[0], axis=1)
    dfnew.to_csv(r'feature_vectors/feature_vectors_labeled.csv')



if __name__=="__main__":
    csv = r'feature_vectors/feature_vectors_not_labeled.csv'
    label(csv)
