import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
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
    def missingfeatures(cls, traindata: pd.DataFrame):
        cls.traindata = traindata
        nullseries = traindata.isnull().mean()[traindata.isnull().mean() > 0]
        return nullseries

    @classmethod
    def missingdrop(cls, traindata: pd.DataFrame, missing: float):
        cls.traindata = traindata
        cls.missing = missing
        nullseries = cls.missingfeatures(traindata)
        nullseriesdrop = list(nullseries[nullseries > missing].index.values)
        traindatadrop = traindata.drop(columns=nullseriesdrop)
        return traindatadrop

    @classmethod
    def groupimputedrop(cls, dataset: pd.DataFrame):
        '''

        :param dataset: pandas dataframe
        :param method_col_dict:
        :return:
        '''
        cls.dataset = dataset
        nullseries = cls.missingfeatures(dataset)
        nullseries[nullseries <= 0.1] = 1
        nullseries[(nullseries > 0.1) & (nullseries <= 0.5)] = 2
        nullseries[(nullseries > 0.5) & (nullseries < 1)] = 3
        replacedict = {1: 'impute', 2: 'group', 3: 'drop'}
        nullseries = nullseries.replace(replacedict)
        method_col_dict = {nullseries.index[i]: nullseries.values[i] for i in range(len(nullseries))}
        print(method_col_dict)
        for key in method_col_dict:
            if method_col_dict[key] == 'group':
                dataset[key] = np.where(dataset[key].dtype != 'object',
                                        dataset[key].fillna(dataset[key].mean()),
                                        dataset[key].fillna(dataset[key].mode()))
            if method_col_dict[key] == 'impute':
                dataset[key] = np.where(dataset[key].dtype != 'object',
                                        dataset[key].fillna(dataset[key].mean()),
                                        dataset[key].fillna(dataset[key].mode()))
            if method_col_dict[key] == 'drop':
                dataset = dataset.drop(columns=key)
            else:
                print("Wrong input: No treatement performed")
        return dataset
