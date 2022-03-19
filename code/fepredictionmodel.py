"""
This file defines the FEModel class.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col


"""
Runs fixed-effects regression models
"""


class FEPredictionModel:
    """
    Initialize a model.
    """

    def __init__(self, datafile, graph_output, table_output, non_numeric_features):
        self.df = pd.read_csv(datafile)
        self.graph_output = graph_output
        self.table_output = table_output

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.non_numeric_features = non_numeric_features
        self.numeric_features = None

    # """
    # Divide dataset randomly into a training and testing set.
    # """
    # def split_train_test_random(self,  y_col, test_percent=0.25, seed=7):
    #     y = self.df[y_col]  # split labels
    #     x = self.df.drop(columns=[y_col])  # drop labels from original dataset
    #     self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,
    #                                                                             y,
    #                                                                             test_size=test_percent,
    #                                                                             random_state=seed)

    """
    Divide dataset into a training and testing set such that each contains the same distribution of entities.
    """

    def split_train_test(self, y_col, entity_var, time_var, test_percent=0.25, seed=7):
        observations_per_entity = self.df[entity_var].value_counts().mode()

        # this ensures that when we generate dummies, col titles are strings
        self.df[entity_var] = self.df[entity_var].astype(str)

        if False in self.df[entity_var].value_counts().isin([observations_per_entity]):  # if this triggers, panel not balanced
            print("ERROR: Panel not balanced.")
            return
        dataset = self.df.sort_values([entity_var, time_var])

        testing_obs_per_group = int(observations_per_entity * test_percent)  # will test on test_percent% of obs from each FIPS code
        training_obs_per_group = observations_per_entity - testing_obs_per_group  # train on the rest

        testing_data = dataset.groupby(entity_var).tail(testing_obs_per_group)  # select test_percent% of obs from each FIPS code
        training_data = dataset.drop(index=testing_data.index)  # drop those observations from the original dataset to get the traiing data

        self.y_train = training_data[y_col]  # separate labels from training date
        self.x_train = training_data.drop(columns=y_col)

        self.y_test = testing_data[y_col]  # separate labels from testing data
        self.x_test = testing_data.drop(columns=y_col)

        # keep track of the numeric feature names in the dataset
        self.x_train = self.x_train.drop(columns='Unnamed: 0')
        self.x_test = self.x_test.drop(columns='Unnamed: 0')
        self.numeric_features = [x for x in list(self.x_train.columns) if x not in self.non_numeric_features]


    """
    Run fixed-effects LASSO regression and produce summary statistics. 
    """

    def run_LASSO(self, fixed_effects_var):
        # generate one dummy variable for each entity except for one
        self.x_train = pd.concat([self.x_train, pd.get_dummies(self.x_train[fixed_effects_var],
                                                               drop_first=True)],
                                 axis=1)
        self.x_test = pd.concat([self.x_test, pd.get_dummies(self.x_test[fixed_effects_var],
                                                             drop_first=True)],
                                axis=1)

        # normalize x_train and x_test
        ct = ColumnTransformer([('Standardizer', StandardScaler(), self.numeric_features)],
                               remainder='passthrough')
        self.x_train = pd.DataFrame(data=ct.fit_transform(self.x_train), columns=self.x_train.columns)
        self.x_test = pd.DataFrame(data=ct.transform(self.x_test), columns=self.x_test.columns)

        # add a constant term
        # self.x_train = sm.add_constant(self.x_train)
        # self.x_test = sm.add_constant(self.x_test)
        print(self.numeric_features)
        print(self.x_train[self.numeric_features].mean(skipna=True))
        print(self.x_train[self.numeric_features].var(skipna=True))
        self.x_train[self.numeric_features].to_csv("~/Desktop/test.csv")
        # # run LASSO with multiple alphas and pick the best
        # alphas = np.linspace(0.01, 5, num=51)
        # for alpha in alphas:
        #     # LASSO regression is a special case of elastic net, where the weight on L1 term is 1
        #     lasso = sm.regression.linear_model.OLS.fit_regularized(method='elastic_net',
        #                                                            alpha=alpha,
        #                                                            L1_wt=1.0)


