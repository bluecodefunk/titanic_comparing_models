""" Machine learning pipeline """
import pandas as pd
from helperfunctions.featureprocessing import MissingVals, FeatureExtract


def main():
    trainfile = '../train.csv'
    dataset = pd.read_csv(trainfile)
    print(dataset.head())
    #dataset_missing_drop = Processing.missingdrop(dataset, 0.4)
    dataset_clr = MissingVals.groupimputedrop(dataset)
    dataset_clr['name_title'] = dataset_clr['Name'].apply(lambda x: FeatureExtract.(x))
    dataset_clr = FeatureExtract.dummyfy(dataset_clr, ['Pclass','Sex','Embarked','name_title'])
    

if __name__ == "__main__":
    main()
