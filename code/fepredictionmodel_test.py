"""
This file tests the FEPredictionModel class.
"""
from fepredictionmodel import FEPredictionModel
from math import isclose

path_to_data = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/cleaned_data/cleaned_dataset.csv"
path_to_graph_outputs = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model1/graphs"
path_to_table_outputs = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model1/tables"
non_numeric_features = ['tract', 'month', 'cz', 'czname', 'county']
# model_1 = FEPredictionModel(path_to_data, path_to_graph_outputs, path_to_table_outputs, non_numeric_features, "Model 1")
# model_1.split_train_test('filings', 'fips', 'month')
# model_1.get_kde_plot('filings')
# model_1.get_summary_statistics()
# model_1.run_ridge('county')

path_to_graph_outputs = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model2/graphs"
path_to_table_outputs = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model2/tables"
model_2 = FEPredictionModel(path_to_data, path_to_graph_outputs, path_to_table_outputs, non_numeric_features, "Model 2")
model_2.split_train_test('filings', 'tract', 'month')
labels = ['Eviction filings',
          'CNIP (previous month)',
          'Unemployment rate (previous month)',
          'Eviction filings  (previous month)',
          '% of workers with <15 min. commute',
          '% single parent households (2010)',
          '% single parent households (1990)',
          '% single parent households (2000)',
          'Mean household income (2000)',
          'Mean commute time (2000)',
          'Portion with bachelor\'s or above (2000)',
          'Portion with bachelor\'s or above (2010)',
          'Portion of residents foreign born',
          'Median household income (1990)',
          'Median household income (2016)',
          'Population density (2000)',
          'Portion below federal poverty line (2010)',
          'Portion below federal poverty line (2000)',
          'Portion below federal poverty line (1990)',
          'Portion white (2010)',
          'Portion Black (2010)',
          'Portion Hispanic (2010)',
          'Portion Asian (2010)',
          'Portion white (2000)',
          'Portion Black (2000)',
          'Portion Hispanic (2000)',
          'Portion Asian (2000)',
          'Mean 3rd grade math test scores (2013)',
          'Median gross 2-bedroom rent (2015)',
          'Rate of employment (2000)',
          'Wage growth for high school graduates',
          'Total number of jobs within 5 miles',
          'Number of high paying jobs within 5 miles',
          'Population density (2010)',
          'Average annualized job growth (2004-2013)',
          'Number of jobs per square mile',
          'Census return rate (2010)']
variables = ['filings',
             'L1_cnip',
             'L1_unemployment_rate',
             'L1_filings',
             'traveltime15_2010',
             'singleparent_share2010',
             'singleparent_share1990',
             'singleparent_share2000',
             'hhinc_mean2000',
             'mean_commutetime2000',
             'frac_coll_plus2000',
             'frac_coll_plus2010',
             'foreign_share2010',
             'med_hhinc1990',
             'med_hhinc2016',
             'popdensity2000',
             'poor_share2010',
             'poor_share2000',
             'poor_share1990',
             'share_white2010',
             'share_black2010',
             'share_hisp2010',
             'share_asian2010',
             'share_white2000',
             'share_black2000',
             'share_hisp2000',
             'share_asian2000',
             'gsmn_math_g3_2013',
             'rent_twobed2015',
             'emp2000',
             'ln_wage_growth_hs_grad',
             'jobs_total_5mi_2015',
             'jobs_highpay_5mi_2015',
             'popdensity2010',
             'ann_avg_job_growth_2004_2013',
             'job_density_2013',
             'mail_return_rate2010']
model_2.get_summary_statistics(variables=variables,
                               labels=labels)
model_2.run_ridge('county')