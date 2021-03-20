""" Machine learning pipeline """
import pandas as pd
from helperfunctions.featureprocessing import MissingVals, FeatureExtract


def main():
    trainfile = '../train.csv'
    dataset = pd.read_csv(trainfile)
    dataset_copy = dataset.copy()
    print(dataset_copy.head())
    #dataset_missing_drop = Processing.missingdrop(dataset_copy, 0.4)
    dataset_clr = MissingVals.groupimputedrop(dataset_copy)
    dataset_clr['NameTitle'] = dataset_clr['Name'].apply(FeatureExtract.search_title)
    dummy_features, high_dim, dataset_dummy = FeatureExtract.finddummy(dataset_clr, group_size=5)
    # Returns dataset with dummy encode features
    dataset_dummy = FeatureExtract.dummyfy(dataset_dummy, dummy_features)
    dataset_dummy = dataset_dummy.drop(high_dim, axis=1)



if __name__ == "__main__":
    main()
