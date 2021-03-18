import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.impute import SimpleImputer
class Processing:

    @classmethod
    def missingfeatures(cls, traindata:pd.DataFrame):
        cls.traindata = traindata
        nullseries = traindata.isnull().mean()[traindata.isnull().mean() > 0]
        return nullseries


    @classmethod
    def missingdrop(cls, traindata:pd.DataFrame, missing:float):
        cls.traindata = traindata
        cls.missing = missing
        nullseries = cls.missingfeatures(traindata)
        nullseriesdrop = list(nullseries[nullseries > missing].index.values)
        traindatadrop = traindata.drop(columns=nullseriesdrop)
        return traindatadrop

    @classmethod
    def groupmeanimpute(cls, method: str, cols: list):
        




def main():
    trainfile = '../train.csv'
    dataset = pd.read_csv(trainfile)
    print(dataset.head())
    print(Processing.missingfeatures(dataset))
    dataset_missing_drop = Processing.missingdrop(dataset, 0.4)
    print(dataset_missing_drop.columns)

if __name__ == "__main__":
    main()
