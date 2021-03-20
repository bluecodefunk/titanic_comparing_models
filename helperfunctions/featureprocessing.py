""" Basic data preparation for training a model
Classes:

    MissingVals
    FeatureExtract

Functions:
    missingfeatures(pd.DataFrame) -> pd.Series
    missingdrop(pd.DataFrame, float) -> pd.DataFrame
    groupimputedrop(pd.DataFrame, dict) -> pd.DataFrame
    search_title(str) -> str
    dummyfy(pd.DataFrame,list) -> pd.DataFrame


"""
import re
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np


class MissingVals:
    """ This class has method for missing value treatment"""

    @classmethod
    def missingfeatures(cls, traindata: pd.DataFrame):
        """
        param traindata: pandas dataframe
        return: pandas series
        """
        cls.traindata = traindata
        nullseries = traindata.isnull().mean()[traindata.isnull().mean() > 0]
        return nullseries

    @classmethod
    def missingdrop(cls, traindata: pd.DataFrame, missing: float):
        """Drop columns with missing percentage greater than threshold"""
        cls.traindata = traindata
        cls.missing = missing
        nullseries = cls.missingfeatures(traindata)
        nullseriesdrop = list(nullseries[nullseries > missing].index.values)
        traindatadrop = traindata.drop(columns=nullseriesdrop)
        return traindatadrop

    @classmethod
    def groupimputedrop(cls, dataset: pd.DataFrame, dictuserinput: dict = None):
        """
        :rtype pd.DataFrame
        :param dictuserinput:
        :param dataset: pd.DataFrame
        """
        cls.dataset = dataset
        cls.dictuserinput = dictuserinput
        nullseries = cls.missingfeatures(dataset)
        nullseries[nullseries <= 0.05] = 1
        nullseries[(nullseries > 0.05) & (nullseries <= 0.2)] = 2
        nullseries[(nullseries > 0.2) & (nullseries < 1)] = 3
        replacedict = {1: 'drop_row', 2: 'drop_col', 3: 'impute'}
        nullseries = nullseries.replace(replacedict)
        if dictuserinput is None:
            method_col_dict = {nullseries.index.values[i]: nullseries.values[i]
                               for i in range(len(nullseries))}
        else:
            method_col_dict = dictuserinput
        print(method_col_dict)
        for key in method_col_dict:
            print(key)
            if method_col_dict[key] == 'drop_col':
                dataset = dataset.drop(columns=key)
            elif method_col_dict[key] == 'drop_row':
                dataset = dataset.dropna(subset=[key], axis=0)
            elif dataset[key].dtype == object and dataset[key].isnull().sum() > 0:
                dataset[key] = dataset[key].fillna(dataset[key].mode())
            elif method_col_dict[key] == 'group':
                dataset[key] = dataset[key].fillna(dataset[key].mean())
            elif method_col_dict[key] == 'impute':
                dataset[key] = dataset[key].fillna(dataset[key].mean())
            else:
                print("Wrong input: No treatement performed")
        return dataset


class FeatureExtract:
    """This class has various functions to extract features    """

    @classmethod
    def search_title(cls, text: str):
        """Return title from name string"""
        cls.text = text
        name_title = re.search(', (.+?)\.', text)
        if name_title:
            return name_title.group(1)

    @classmethod
    def finddummy(cls, dataset: pd.DataFrame, max_uniques: int = 20, group_size: int = None):
        """
        Get dummy features list, high dimensional features and modify feature bins

        Returns dummy_list if group_size = None else returns dummy_features, highdimfeatures
        and modifies features with (5 < colunique < 20) to (colunique <=5)
        :rtype: list, list, pd.DataFrame
        """

        cls.dataset = dataset
        cls.max_uniques = max_uniques
        cls.group_size = group_size
        discreetcols = dataset.select_dtypes(exclude=['int64', 'float64']).columns
        dummylist = []
        for colname in discreetcols:
            colunique = dataset[colname].nunique()
            if colunique <= max_uniques and isinstance(colname, str):
                dummylist.append(colname)
        highdimfeatures = list(np.setdiff1d(discreetcols, dummylist))
        if isinstance(group_size, int):
            for colname in dummylist:
                colunique = dataset[colname].nunique()
                if colunique > group_size:
                    mainlabels = list(dataset[colname].value_counts()[:group_size].index.values)
                    dataset[colname] = np.where(dataset[colname].isin(mainlabels),
                                                dataset[colname],
                                                "other bin")

            return dummylist, highdimfeatures, dataset
        else:
            return dummylist

    @classmethod
    def dummyfy(cls, dataset: pd.DataFrame, cols: list = None):
        """Dummy encode categorical features"""
        cls.dataset = dataset
        cls.cols = cols
        if not cols:
            cols = cls.finddummy(dataset, max_uniques=20, group_size=None)
        for colname in cols:
            one_hot = pd.get_dummies(dataset[colname], drop_first=True, prefix=colname)
            # Drop column B as it is now encoded
            dataset = dataset.drop(colname, axis=1)
            # Join the encoded df
            dataset = dataset.join(one_hot)
        return dataset
