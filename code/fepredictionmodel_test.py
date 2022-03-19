"""
This file tests the FEPredictionModel class.
"""
from fepredictionmodel import FEPredictionModel

path_to_data = "/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/cleaned_data/cleaned_dataset.csv"
path_to_graph_outputs = "/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/output/graphs"
path_to_table_outputs = "/Users/arjunshanmugam/Documents/School/Brown/Semester6/ECON1680/project1/project1-arjun-shanmugam/output/tables"
model_1 = FEPredictionModel(path_to_data, path_to_graph_outputs, path_to_table_outputs)
model_1.split_train_test_panel_data('filings', 'fips', 'month')

# test that x_train and x_test have identical columns
assert set(model_1.x_train.columns) == set(model_1.x_test.columns)

model_1.run_LASSO('fips')

