from kgs_inferential_statistics import KGsInferentialStatistics
from variables import QUALITY_DIMENSIONS
from generate_heatmap import GenerateHeatmap
import os
import utils    

here = os.path.dirname(os.path.abspath(__file__))

analysis_dates= ['2024-01-07','2024-04-07','2024-08-04','2024-12-01','2025-04-06']

available_ids = utils.get_always_sparql_up(analysis_dates)
print("Always available KGs: ",len(available_ids))

for analysis_date in analysis_dates:
    correlation_matrix_spearman = KGsInferentialStatistics(os.path.join(here,f'../data/quality_analysis_results/{analysis_date}.csv'),f'Spearman_correlation_matrix_dimensions_{analysis_date}_Always_UP')
    correlation_matrix_spearman.calculate_spearman_correlation_matrix(QUALITY_DIMENSIONS,True,filter_ids=available_ids)

    heatmap = GenerateHeatmap(os.path.join(here,f'../data/inferential_statistics_results/Spearman_correlation_matrix_dimensions_{analysis_date}_Always_UP.csv'),f'../data/charts/heatmap/Spearman_{analysis_date}_Always_UP',f'Spearman correlation matrix with quality data computed on {analysis_date} on KGs with SPARQL endpoint always UP')
    heatmap.draw_heatmap()

always_monitored = utils.get_always_observed_ids(analysis_dates[0])

print("Always monitored KGs: ",len(always_monitored))
for analysis_date in analysis_dates:
    correlation_matrix_spearman = KGsInferentialStatistics(os.path.join(here,f'../data/quality_analysis_results/{analysis_date}.csv'),f'Spearman_correlation_matrix_dimensions_{analysis_date}_ALL')
    correlation_matrix_spearman.calculate_spearman_correlation_matrix(QUALITY_DIMENSIONS,False,filter_ids=always_monitored)

    heatmap = GenerateHeatmap(os.path.join(here,f'../data/inferential_statistics_results/Spearman_correlation_matrix_dimensions_{analysis_date}_ALL.csv'),f'../data/charts/heatmap/Spearman_{analysis_date}_ALL',f'Spearman correlation matrix with quality data computed on {analysis_date} on All KGs')
    heatmap.draw_heatmap()