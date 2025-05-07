import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap
import os
matplotlib.use('Agg')

here = os.path.dirname(os.path.abspath(__file__))

class GenerateHeatmap:
    def __init__(self,correlation_data,output_dir,plot_title) -> None:
        self.correlation_data = pd.read_csv(correlation_data,index_col=0)
        self.output_dir = os.path.join(here,output_dir)
        self.plot_title = plot_title
    

    def draw_heatmap(self, replace = False):
        correlation_matrix =  self.correlation_data.loc[:, ~ self.correlation_data.columns.str.contains('_p-value')]
        p_value_matrix =  self.correlation_data.filter(like='_p-value')

        p_value_numeric = p_value_matrix.replace({'\*\*\*': 0.001, '\*\*': 0.01, '\*': 0.05}, regex=True)
        p_value_numeric = p_value_numeric.apply(pd.to_numeric, errors='coerce')

        if replace:
            for col in correlation_matrix:
                    correlation_matrix[col] = correlation_matrix[col].str.replace(',', '.')
                    correlation_matrix[col] = pd.to_numeric(correlation_matrix[col], errors='coerce') 

        masked_correlation_matrix = correlation_matrix.copy()
        masked_correlation_matrix[p_value_numeric > 0.05] = np.nan

        annotations = masked_correlation_matrix.copy().astype(str)
        for i in range(p_value_matrix.shape[0]):
            for j in range(p_value_matrix.shape[1]):
                if not pd.isna(masked_correlation_matrix.iloc[i, j]):
                    if not pd.isna(p_value_matrix.iloc[i, j]):  
                        annotations.iloc[i, j] += f" {p_value_matrix.iloc[i, j]}"

        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        cmap = LinearSegmentedColormap.from_list('RedGreenRed', ['blue', 'white', 'blue'])

        plt.figure(figsize=(20, 10))
        ax = sns.heatmap(
            masked_correlation_matrix.astype(float), 
            annot=annotations.values,                
            fmt="",                                 
            cmap=cmap,                               
            cbar=True,                               
            norm=norm,                               
            linewidths=0.5,                         
            linecolor="gray"                         
        )

        plt.title(self.plot_title, fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir, dpi=300)
        plt.close()