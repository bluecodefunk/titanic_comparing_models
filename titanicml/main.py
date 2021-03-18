from helper_functions.dataclean import Processing
import pandas as pd


def main():
    trainfile = '../train.csv'
    dataset = pd.read_csv(trainfile)
    print(dataset.head())
    #print(Processing.missingfeatures(dataset))
    #dataset_missing_drop = Processing.missingdrop(dataset, 0.4)
    processed_dataset = Processing.groupimputedrop(dataset)
    print(processed_dataset.isnull().sum())


if __name__ == "__main__":
    main()
