"""
This file tests the FEPredictionModel class.
"""
from fepredictionmodel import FEPredictionModel
from math import isclose

path_to_data = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/cleaned_data/cleaned_dataset.csv"
path_to_graph_outputs = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model1/graphs"
path_to_table_outputs = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model1/tables"
non_numeric_features = ['fips', 'month', 'cz', 'czname', 'county']
# model_1 = FEPredictionModel(path_to_data, path_to_graph_outputs, path_to_table_outputs, non_numeric_features, "Model 1")
# model_1.split_train_test('filings', 'fips', 'month')
# model_1.get_kde_plot('filings')
# model_1.get_summary_statistics()
# model_1.run_ridge('county')

path_to_graph_outputs = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model2/graphs"
path_to_table_outputs = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model2/tables"
model_2 = FEPredictionModel(path_to_data, path_to_graph_outputs, path_to_table_outputs, non_numeric_features, "Model 2")
model_2.split_train_test('filings', 'fips', 'month')
model_2.run_ridge('fips')