""" Class implements model training steps - data split, fitting model, return metrics

Classes:


Functions:

"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd


class TrainClassifier:
    """Outputs model results for the dataset and model passed"""

    @classmethod
    def datasplit(cls, dataset: pd.DataFrame, id_columns: list, label_column: str,
                  scale: bool = False):
        """Ready the dataframe to fit the model"""
        cls.dataset = dataset
        cls.id_columns = id_columns
        cls.label_column = label_column

        if not isinstance(id_columns, list):
            id_columns = [str(id_columns)]
        if scale:
            scale_fit = StandardScaler()
            features = dataset.drop([label_column] + id_columns, axis=1)
            label = dataset[label_column]
            features = scale_fit.fit_transform(features)
        x_train, x_test, y_train, y_test = train_test_split(
            features, label, test_size = 0.33, random_state = 42)
        return [x_train, x_test, y_train, y_test]


    @classmethod
    def fitmetrics(cls, model, dataset_list):
        """Prints the model metrics"""
        cls.model = model
        cls.dataset_list = dataset_list
        x_train, x_test, y_train, y_test = dataset_list
        del x_train, y_train
        y_pred = model.predict(x_test)
        accuracy = model.score(x_test, y_test)
        aucval = roc_auc_score(y_test, y_pred)

        print("Model accuracy is: ", accuracy)
        print("Model AUC :", aucval)
        print("Model confusion matrix:\n",confusion_matrix(y_test, y_pred))
        return accuracy,aucval


    @classmethod
    def fitscore(cls, dataset_list: list,
                 model_func, params: dict):
        """Creates the estimator and call function to output metrics"""
        cls.dataset_list = dataset_list
        cls.model_func = model_func
        cls.params = params
        x_train, x_test, y_train, y_test = dataset_list
        del x_test, y_test
        model_list = [LogisticRegressionCV(), DecisionTreeClassifier(), RandomForestClassifier(),
                      AdaBoostClassifier(), SVC()]
        for model_iter in model_list:
            if model_func == model_iter.__str__():
                estimator = model_iter
                estimator.set_params(**params)
                estimator.fit(x_train, y_train)
                cls.fitmetrics(estimator, dataset_list)
                break
