"""
This file defines the FEModel class.
"""
import numpy as np
import pandas as pd
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

    def __init__(self, datafile, graph_output, table_output):
        self.df = pd.read_csv(datafile)
        self.graph_output = graph_output
        self.table_output = table_output

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.regression = None

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

    def split_train_test_panel_data(self, y_col, entity_var, time_var, test_percent=0.25, seed=7):
        observations_per_entity = self.df[entity_var].value_counts().mode()

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

    """
    Run fixed-effects LASSO regression and produce summary statistics. 
    """

    def run_LASSO(self, fixed_effects_var):
        # we will need to edit features by adding dummy variables, normalizing, and adding a constant, so work with copies
        train_features = self.x_train.copy()
        test_features = self.x_test.copy()

        # generate one dummy variable for each entity except for one
        print("generating dummies")
        for entity in self.df[fixed_effects_var].keys()[0:-1]:
            # for each entity, generate a column = 1 if the row corresponds to that entity and 0 otherwise
            train_features[entity] = np.where(train_features[fixed_effects_var] == entity, 1, 0)
            test_features[entity] = np.where(test_features[fixed_effects_var] == entity, 1, 0)

        # normalize x_train and x_test
        print("Normalizing!")
        standardizer = StandardScaler
        train_features = standardizer.fit_transform(X=train_features)
        test_features = standardizer.transform(X=test_features)

        # add a constant term
        print("Adding a constant!")
        train_features = sm.add_constant(train_features)
        test_features = sm.add_constant(test_features)

        print(train_features.mean())
        print(test_features.mean())


        # # run LASSO with multiple alphas and pick the best
        # alphas = np.linspace(0.01, 5, num=51)
        # for alpha in alphas:
        #     # LASSO regression is a special case of elastic net, where the weight on L1 term is 1
        #     lasso = sm.regression.linear_model.OLS.fit_regularized(method='elastic_net',
        #                                                            alpha=alpha,
        #                                                            L1_wt=1.0)


