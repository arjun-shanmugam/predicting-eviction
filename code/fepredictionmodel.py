"""
This file defines the FEModel class.
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
import dataframe_image as dfi




"""
Runs fixed-effects regression models
"""


class FEPredictionModel:
    """
    Initialize a model.
    """

    def __init__(self, datafile, graph_output, table_output, non_numeric_features, model_name):
        self.df = pd.read_csv(datafile)
        self.graph_output = graph_output
        self.table_output = table_output
        self.model_name = model_name

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.optimal_coefficients = None
        self.optimal_mse = None

        self.non_numeric_features = non_numeric_features
        self.numeric_features = None

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
    Run fixed-effects ridge regression and produce summary statistics. 
    """
    def run_ridge(self, fixed_effects_var):

        # generate one dummy variable for each entity except for one
        train_dummies = pd.get_dummies(self.x_train[fixed_effects_var],
                                       drop_first=True)
        self.x_train = pd.concat([self.x_train, train_dummies],
                                 axis=1)
        test_dummies = pd.get_dummies(self.x_test[fixed_effects_var],
                                      drop_first=True)
        self.x_test = pd.concat([self.x_test, test_dummies],
                                axis=1)
        dummy_col_names = train_dummies.columns

        # drop categorical features
        self.x_train = self.x_train.drop(columns=self.non_numeric_features)
        self.x_test = self.x_test.drop(columns=self.non_numeric_features)
        self.x_train.to_csv("~/Desktop/test.csv")
        # run LASSO with multiple alphas and pick the best
        alphas = np.linspace(0.01, 5, num=51)
        errors = []
        coefficients = []
        for alpha in alphas:
            ridge = Ridge(alpha=alpha, fit_intercept=True, normalize=True, random_state=7)
            ridge.fit(self.x_train, self.y_train)
            predicted_y = ridge.predict(self.x_test)
            predicted_y = np.where(predicted_y < 0, 0, predicted_y)
            errors.append(np.mean((predicted_y - self.y_test) ** 2))
            coefficients.append(ridge.coef_)

        # plot MSE for different values of alpha
        figure1 = plt.figure()
        plt.xlabel(r'$\alpha$')
        plt.ylabel("Mean squared prediction error")
        plt.title(self.model_name + ": Ridge Prediction Error by Penalty Parameter " + "(" + r'$\alpha$' + ")")
        plt.plot(alphas, errors, '.')
        figure1.savefig(os.path.join(self.graph_output, self.model_name + '_mse_plot.png'))

        # store optimal coefficients and prediction error
        self.optimal_mse = np.min(errors)
        self.optimal_coefficients = coefficients[np.argmin(errors)]
        # self.optimal_coefficients = pd.DataFrame(self.optimal_coefficients, columns=self.x_train.columns)

    """
    Produces a KDE of the distribution of a certain variable.
    """
    def get_kde_plot(self, column, plot_title, xlabel):
        figure1 = plt.figure(1)
        # kde plot for training dataset
        ax1 = pd.concat([self.x_train, self.y_train], axis=1)[column].plot.kde(title=plot_title,
                                                                               label='Training dataset')
        # kde plot for testing dataset
        ax1 = pd.concat([self.x_test, self.y_test], axis=1)[column].plot.kde(label='Testing dataset')
        figure1.legend(loc='center')
        plt.xlim([-10, 40])
        plt.xlabel(xlabel)
        plt.figure(1).savefig(os.path.join(self.graph_output, 'kde_plot_' + column + ".png"))

    def get_summary_statistics(self, variables, labels):
        rename_dict = {}  # create dictionary to rename the columns
        for variable, label in zip(variables, labels):
            rename_dict[variable] = label
        training_statistics = pd.concat([self.x_train[self.numeric_features], self.y_train], axis=1).describe()
        testing_statistics = pd.concat([self.x_test[self.numeric_features], self.y_test], axis=1).describe()
        training_statistics = training_statistics.rename(columns=rename_dict)
        testing_statistics = testing_statistics.rename(columns=rename_dict)

        dfi.export(training_statistics.transpose()[['count', 'mean', 'std', '50%']], os.path.join(self.table_output, 'train_summary_stats.png'))
        dfi.export(testing_statistics.transpose()[['count', 'mean', 'std', '50%']], os.path.join(self.table_output, 'test_summary_stats.png'))

