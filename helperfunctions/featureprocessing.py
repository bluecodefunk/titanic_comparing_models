""" This is a library with functions useful for missing value treatment and ...
"""
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

class MissingVals:
    """ This class has method for missing value treatment
    """

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
    def groupimputedrop(cls, dataset: pd.DataFrame, dictuserinput: dict = None):
        """
        :param dictuserinput:
        :param dataset: pandas dataframe
        :return:
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
                dataset = dataset.dropna(subset = [key], axis=0   )
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
        cls.text = text
        m = re.search(', (.+?)\.', text)
        if m:
            return m.group(1)
        else:
            return None
    @classmethod
    def dummyfy(cls, dataset: pd.DataFrame, cols: list):
        self.dataset = dataset
        sel
        for cols in ['Pclass', 'Sex', 'Embarked', 'name_title']:
            one_hot = pd.get_dummies(dataset[cols], drop_first=True)
            # Drop column B as it is now encoded
            dataset = dataset.drop(cols, axis=1)
            # Join the encoded df
            dataset = dataset.join(one_hot)

