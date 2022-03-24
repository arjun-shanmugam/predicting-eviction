"""
This file defines the FEModel class.
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
import dataframe_image as dfi
from sklearn.metrics import r2_score

"""
Runs fixed-effects regression models
"""


class FEPredictionModel:
    """
    Initialize a model.
    """

    def __init__(self, datafile, graph_output, table_output, non_numeric_features, model_name, float_format='{:.5f}'):
        pd.options.display.float_format = float_format.format
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
        self.optimal_r_squared = None
        self.output_table = None

        self.non_numeric_features = non_numeric_features
        self.numeric_features = None
        self.fe_dummies = None

    """
    Divide dataset into a training and testing set such that each contains the same distribution of entities.
    """

    def split_train_test(self, y_col, entity_var, time_var, test_percent=0.25):
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

    def run_ridge(self, fixed_effects_var, exclude_variables=None, include_dummies_in_table=False):

        # generate one dummy variable for each entity except for one
        train_dummies = pd.get_dummies(self.x_train[fixed_effects_var], drop_first=True, prefix=fixed_effects_var, prefix_sep="_")
        self.x_train = pd.concat([self.x_train, train_dummies], axis=1)
        test_dummies = pd.get_dummies(self.x_test[fixed_effects_var], drop_first=True, prefix=fixed_effects_var, prefix_sep="_")
        self.x_test = pd.concat([self.x_test, test_dummies], axis=1)
        self.fe_dummies = train_dummies.columns

        # drop categorical features
        self.x_train = self.x_train.drop(columns=self.non_numeric_features)
        self.x_test = self.x_test.drop(columns=self.non_numeric_features)

        # drop researcher-specified features
        if exclude_variables is not None:
            self.x_train = self.x_train.drop(columns=exclude_variables)
            self.x_test = self.x_test.drop(columns=exclude_variables)

        # run ridge with multiple alphas and pick the best
        alphas = np.linspace(0.01, 5, num=51)
        errors = []
        coefficients = []
        r_squareds = []
        for alpha in alphas:
            ridge = Ridge(alpha=alpha, normalize=True, random_state=7)
            ridge.fit(self.x_train, self.y_train)
            predicted_y = ridge.predict(self.x_test)
            predicted_y = np.where(predicted_y < 0, 0, predicted_y)
            errors.append(np.mean((predicted_y - self.y_test) ** 2))
            coefficients.append(ridge.coef_)
            r_squareds.append(r2_score(self.y_test, predicted_y))

        # plot MSE for different values of alpha
        figure1 = plt.figure()
        plt.xlabel(r'$\alpha$')
        plt.ylabel("Mean squared prediction error")
        plt.title(self.model_name + ": Ridge Prediction Error by Penalty Parameter " + "(" + r'$\alpha$' + ")")
        plt.plot(alphas, errors, '.')
        figure1.savefig(os.path.join(self.graph_output, self.model_name + '_mse_plot.png'))

        # store optimal coefficients, prediction error, R2
        self.optimal_coefficients = coefficients[np.argmin(errors)]
        self.optimal_mse = np.min(errors)
        self.optimal_r_squared = r_squareds[np.argmin(errors)]
        self.output_table = pd.DataFrame({"Variable/Metric": self.x_train.columns,
                                          "Coefficient/Value": self.optimal_coefficients})

        # produce pdf of coefficient values
        figure = plt.figure()
        self.output_table.loc[self.output_table["Variable/Metric"].isin(self.fe_dummies)]['Coefficient/Value'].plot.kde(
            title="Distribution of coefficients on " + fixed_effects_var + " dummies")
        plt.xlabel("Coefficient value")
        plt.xlim(-3, 3)
        figure.savefig(os.path.join(self.graph_output, self.model_name + "_coefficients_distribution.png"))

        # drop rows corresponding to dummy variable coefficients
        self.output_table = self.output_table.drop(self.output_table.loc[self.output_table['Variable/Metric'].isin(self.fe_dummies)].index)

        # append MSE and R^2 to the output table
        self.output_table = self.output_table.append({'Variable/Metric': "MSE",
                                                      'Coefficient/Value': self.optimal_mse}, ignore_index=True)
        self.output_table = self.output_table.append({'Variable/Metric': "R^2",
                                                      'Coefficient/Value': self.optimal_r_squared}, ignore_index=True)

        # save output table as PNG
        self.output_table.set_index('Variable/Metric')
        dfi.export(self.output_table, os.path.join(self.table_output, self.model_name + "_regression_output.png"))

    """
    Produces a KDE of the distribution of a certain variable.
    """

    def get_kde_plot(self, column, plot_title, xlabel):
        figure1 = plt.figure(1)
        # kde plot for training dataset
        pd.concat([self.x_train, self.y_train], axis=1)[column].plot.kde(title=plot_title,
                                                                         label='Training dataset')
        # kde plot for testing dataset
        pd.concat([self.x_test, self.y_test], axis=1)[column].plot.kde(label='Testing dataset')
        figure1.legend(loc='center')
        plt.xlim([-10, 40])
        plt.xlabel(xlabel)
        plt.figure(1).savefig(os.path.join(self.graph_output, 'kde_plot_' + column + ".png"))

    """
    Produce a table of summary statistics for select variables in PNG format.
    """

    def get_summary_statistics(self, variables, labels):

        training_statistics = pd.concat([self.x_train[self.numeric_features], self.y_train], axis=1)[variables].describe()
        testing_statistics = pd.concat([self.x_test[self.numeric_features], self.y_test], axis=1)[variables].describe()

        # transpose summary tables and add variable labels
        training_summary = training_statistics.transpose()
        testing_summary = testing_statistics.transpose()

        print(training_summary.index)
        print(labels)

        training_summary['Label'] = pd.Series(labels, index=training_summary.index)
        testing_summary['Label'] = pd.Series(labels, index=training_summary.index)


        print(pd.Series(labels))
        print(training_summary['Label'])
        print(testing_summary['Label'])

        dfi.export(training_summary[['Label', 'count', 'mean', 'std', '50%']], os.path.join(self.table_output, 'train_summary_stats.png'))
        dfi.export(testing_summary[['Label', 'count', 'mean', 'std', '50%']], os.path.join(self.table_output, 'test_summary_stats.png'))
