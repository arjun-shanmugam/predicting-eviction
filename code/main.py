"""
Runs the models and produces figures.
"""
import os
import numpy as np
import pandas as pd
from fepredictionmodel import FEPredictionModel
from urllib.request import urlopen
import json
import plotly.express as px
import dataframe_image as dfi

# Set paths here
merged_data = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/cleaned_data/cleaned_dataset.csv"
model1_graph_output = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model1/graphs"
model1_tables_output = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model1/tables"
model2_graph_output = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model2/graphs"
model2_tables_output = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model2/tables"
model3_graph_output = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model3/graphs"
model3_tables_output = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model3/tables"
model4_graph_output = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model4/graphs"
model4_tables_output = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output/model4/tables"
output_general = "/Users/arjunshanmugam/Documents/GitHub/project1-arjun-shanmugam/output"

### Model 1
model_1 = FEPredictionModel(datafile=merged_data,
                            graph_output=model1_graph_output,
                            table_output=model1_tables_output,
                            non_numeric_features=['tract', 'month', 'cz', 'czname', 'county'],
                            model_name="Model 1")
model_1.split_train_test(y_col='filings',
                         entity_var='tract',
                         time_var='month')

# Figure 1: Map of observed counties
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
counties_df = pd.concat([model_1.df['county'], pd.Series(np.ones(len(model_1.df['county'])), name='ones')], axis=1)
fig = px.choropleth_mapbox(counties_df,
                           title="Map of Observed Census Tracts (Aggregated to County)",
                           geojson=counties,
                           locations='county',
                           color='ones',
                           mapbox_style="open-street-map",
                           zoom=4.5,
                           center={"lat": 37.248198, "lon": -95.5},
                           opacity=0.5).update(layout=dict(title=dict(x=0.5)))
fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
fig.update(layout_coloraxis_showscale=False)

fig.write_image(os.path.join(model1_graph_output, 'observed_counties_map.png'), width=1400, height=800)

# Figure 2
model_1.get_kde_plot('filings', "KDE of Eviction Filings at Census Tract-Month Level", "Number of filings")

# Table 1
labels = ['Eviction filings',
          'Civilian non-institutionalized population (previous month)',
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
          'Census mail return rate (2010)',
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
          'Number of jobs per square mile']
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
             'mail_return_rate2010',
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
             'job_density_2013']
model_1.get_summary_statistics(variables=variables,
                               labels=labels)
model_1.run_ridge('tract')


### Model 2
model_2 = FEPredictionModel(datafile=merged_data,
                            graph_output=model2_graph_output,
                            table_output=model2_tables_output,
                            non_numeric_features=['tract', 'month', 'cz', 'czname', 'county'],
                            model_name="Model 2")
model_2.split_train_test(y_col='filings',
                         entity_var='tract',
                         time_var='month')
model_2.get_summary_statistics(variables=variables,
                               labels=labels)
model_2.run_ridge('county')

### Model 3
model_3 = FEPredictionModel(datafile=merged_data,
                            graph_output=model3_graph_output,
                            table_output=model3_tables_output,
                            non_numeric_features=['tract', 'month', 'cz', 'czname', 'county'],
                            model_name="Model 3")
model_3.kmeans()
model_3.split_train_test(y_col='filings',
                         entity_var='tract',
                         time_var='month')
model_3.get_summary_statistics(variables=variables,
                               labels=labels)
model_3.run_ridge('cluster')

# generate concatenated regression output table
labels = model_1.output_table['Label']
model_1_output = pd.Series(model_1.output_table['Coefficient/Value'], name="Model 1")
model_2_output = pd.Series(model_2.output_table['Coefficient/Value'], name="Model 2")
model_3_output = pd.Series(model_3.output_table['Coefficient/Value'], name="Model 3")
all_models_output = pd.concat([labels, model_1_output, model_2_output, model_3_output], axis=1)
dfi.export(all_models_output, os.path.join(output_general, 'all_models_reg_output.png'))

### compare model 1 estimates with and without covariates
# model 4 runs tract fixed effects regressio
model_4 = FEPredictionModel(datafile=merged_data,
                            graph_output=model4_graph_output,
                            table_output=model4_tables_output,
                            non_numeric_features=['tract', 'month', 'cz', 'czname', 'county'],
                            model_name="Model 4")
model_4.split_train_test(y_col='filings',
                         entity_var='tract',
                         time_var='month')
model_4.get_summary_statistics(variables=variables,
                               labels=labels)
model_4.run_ridge('tract', exclude_variables=['traveltime15_2010',
                                              'mail_return_rate2010',
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
                                              'job_density_2013'])
model_4_output = pd.Series(model_4.output_table['Coefficient/Value'], name="Model 1 w/o covariates")
model_1_and_4_output = pd.concat([labels, model_1_output, model_4_output], axis=1)
model_1_and_4_output.fillna('')
dfi.export(model_1_and_4_output, os.path.join(output_general, 'model1_without_oi_compare.png'))
print(model_1.optimal_abs_error)