import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import os
import csv
import numpy as np
from scipy.stats import ttest_ind
import utils

here = os.path.dirname(os.path.abspath(__file__))

class KGsInferentialStatistics:
    def __init__(self, file_path, output_file):
        '''
            Save the path to the file with the KGs quality data.
            
            :param file_path: path to the file with the KGs data
            :param output_file: name of the file in which to write the results
        '''
        self.analysis_result = file_path
        self.output_file = os.path.join(here,'../data/inferential_statistics_results/' + output_file)
    
    def calculate_pearson_on_metrics(self,first_variable,second_variable, sparql_status = False):
        """
            Calculate the Pearson correlation on two metrics or dimensions (columns) from the quality data in the CSV file.

            :param first_variable: column name in the CSV file to use as the first variable to check correlation.
            :param second_variable: column name in the CSV file to use as the second variable to check correlation.
            :param sparql_status: include or exclude KGs with SPARQL endpoit online. False only KGs with SPARQL endpoint DOWN, True for only KGs with SPARQL endpoint UP.
        """
        df = pd.read_csv(self.analysis_result,usecols=['Sparql endpoint',first_variable,second_variable])
        if sparql_status:
            df = df[(df["Sparql endpoint"] == "Available")] 
        else:
            df = df[(df["Sparql endpoint"] != "Available")] 

        # Exclude rows where the quality value could not be calculated
        df = df[(df[first_variable] != '-') & (df[second_variable] != '-')]

        first_variable_values = df[first_variable]
        second_variable_values = df[second_variable]

        first_variable_values = pd.to_numeric(first_variable_values, errors='coerce')
        second_variable_values = pd.to_numeric(second_variable_values, errors='coerce')

        pearson_result = pearsonr(first_variable_values, second_variable_values)

        self.write_correlation_data_on_csv(first_variable,second_variable,pearson_result,'pearson',sparql_status)
    
    def calculate_spearman_on_metrics(self,first_variable, second_variable,sparql_status = False):
        """
            Calculate the Spearman correlation on two metrics or dimensions (columns) from the quality data in the CSV file.

            :param first_variable: column name in the CSV file to use as the first variable to check correlation.
            :param second_variable: column name in the CSV file to use as the second variable to check correlation.
            :param sparql_status: include or exclude KGs with SPARQL endpoit online. False only KGs with SPARQL endpoint DOWN, True for only KGs with SPARQL endpoint UP.
        """
        df = pd.read_csv(self.analysis_result,usecols=[first_variable,second_variable])
        if sparql_status:
            df = df[(df["Sparql endpoint"] == "Available")] 
        else:
            df = df[(df["Sparql endpoint"] != "Available")] 

        first_variable_values = df[first_variable]
        second_variable_values = df[second_variable]

        first_variable_values = pd.to_numeric(first_variable_values, errors='coerce')
        second_variable_values = pd.to_numeric(second_variable_values, errors='coerce')

        spearman_result = spearmanr(first_variable_values, second_variable_values,nan_policy='omit')

        self.write_correlation_data_on_csv(first_variable,second_variable,spearman_result,'spearman',sparql_status)

    
    def calculate_t_test(self,variable_to_verify):

        df = pd.read_csv(self.analysis_result,usecols=['Sparql endpoint',variable_to_verify])
        
        kgs_with_sparql_up = df[(df["Sparql endpoint"] == "Available")] 
        kgs_with_sparql_down = df[(df["Sparql endpoint"] != "Available")] 

        kgs_with_sparql_up_values = kgs_with_sparql_up[variable_to_verify]
        kgs_with_sparql_down_values = kgs_with_sparql_down[variable_to_verify]

        ttest_result = ttest_ind(kgs_with_sparql_up_values,kgs_with_sparql_down_values)

        self.write_ttest_data_on_csv(variable_to_verify,ttest_result)

    def write_correlation_data_on_csv(self, first_variable, second_variable, correlation_result,correlation_type, sparql_status):
        """
            Write the correlation data on a CSV file.

            :param first_variable: name of the first variable used to calculate the correlation
            :param second_variable: name of the second variable used to calculate the correlation
            :param pearson_result: value obtained from the execution of the spearmanr method from shipy
            :param correlation_type: type of correlation used (the possible values are: pearson and spearman)
            :param sparql_status: type of KGs considered in the correlation calculation. False only KGs with SPARQL endpoint DOWN, True for only KGs with SPARQL endpoint UP.
        """
        output_file = self.output_file + f'_sparql_status_{sparql_status}.csv'
        mode = 'a' if os.path.exists(output_file) else 'w'
        
        correlation, p_value = correlation_result

        with open(output_file, mode, newline='') as csv_file:
            writer = csv.writer(csv_file)
            if mode == 'w':
                if correlation_type == 'pearson':
                    writer.writerow(['First variable','Second variable','Pearson correlation value','p-value'])
                if correlation_type == 'spearman':
                    writer.writerow(['First variable','Second variable','Spearman correlation value','p-value'])
                if correlation_type == 'ttest':
                    writer.writerow(['First group','Second group','Ttest correlation value','p-value'])
                writer.writerow([first_variable,second_variable,correlation,p_value])
            else:
                writer.writerow([first_variable,second_variable,correlation,p_value])
    
    def write_ttest_data_on_csv(self,variable,correlation_result):
        """
            Write the ttest correlation data on a CSV file.

            :param name of the first variable used to calculate the correlation
            :param correlation_result: value obtained from the execution of the spearmanr method from shipy
        """
        output_file = self.output_file + f'.csv'
        mode = 'a' if os.path.exists(output_file) else 'w'

        correlation, p_value = correlation_result

        with open(output_file, mode, newline='') as csv_file:
            writer = csv.writer(csv_file)
            if mode == 'w':
                writer.writerow(['Variable used for ttest','Ttest correlation value','p-value'])
                writer.writerow([variable,correlation,p_value])
            else:
                writer.writerow([variable,correlation,p_value])

        
    def calculate_pearson_correlation_matrix(self,columns_to_use,sparql_status = True):
        '''
            Generate the Pearson Correlation matrix by using the values in the columns columns_to_use from the CSV file.      

            :param columns_to_use: list of strings representing the names of the columns from which to take values to measure correlation.
        '''
        if not 'LUZZU' in self.output_file and not 'merged' in self.output_file and not 'SPARQLES' in self.output_file:
            columns_to_use.append('Sparql endpoint')
            df = pd.read_csv(self.analysis_result,usecols=columns_to_use)
            if sparql_status:
                df = df[(df["Sparql endpoint"] == "Available")] 
            else:
                df = df[(df["Sparql endpoint"] != "Available")] 
                
            # Delete the column to avoid errors
            df = df.drop('Sparql endpoint', axis=1)
        else:
            df = pd.read_csv(self.analysis_result,usecols=columns_to_use)
        
        # Replace '-' with NaN and drop rows with NaN values
        df.replace('-', np.nan, inplace=True)
        df.dropna(inplace=True)

        rho = df.corr('pearson')
        pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
        p = pval.map(add_significance_stars)
        
        final_matrix = pd.DataFrame()

        # Iterate through the columns to create pairs of correlation values and stars
        for col in rho.columns:
            # Append correlation values
            final_matrix[col] = rho[col].round(2)  
            # Append corresponding significance stars
            final_matrix[f"{col}_p-value"] = p[col]

        final_matrix.to_csv(self.output_file + '.csv')

    def calculate_spearman_correlation_matrix(self,columns_to_use,sparql_status = True, replace_columns = False, filter_ids=None, metrics = False):
        '''
            Generate the Spearman Correlation matrix by using the values in the columns columns_to_use from the CSV file.      

            :param columns_to_use: list of strings representing the names of the columns from which to take values to measure correlation.
            :param replace_columns: if True, columns that have a list or a boll value as their value will be transformed into a float
        '''
        if not 'LUZZU' in self.output_file and not 'Yummy' in self.output_file and not 'SPARQLES' in self.output_file and not 'luzzu' in self.output_file and not 'sparqles' in self.output_file and not 'yummy' in self.output_file:
            columns_to_use.append('Sparql endpoint')
            columns_to_use.append('KG name')
            columns_to_use.append('KG id')
            df = pd.read_csv(self.analysis_result,usecols=columns_to_use)

            #df = self.add_analysis_time_to_df(df)

            if replace_columns:
                df = utils.convert_metrics_to_float(df, metrics_as_list=['New terms defined in the dataset','metadata-media-type','Languages (query)',
                                                                'External links','Regex uri'],bool_metrics=['Ontology Hijacking problem'])
            for metric in columns_to_use:
                try:
                    df[metric] = df[metric].replace("insufficient data", np.nan)
                except KeyError:
                    continue

            if sparql_status:
                df = df[(df["Sparql endpoint"] == "Available")] 
            #else:
            #    df = df[(df["Sparql endpoint"] != "Available")] 
            
            if filter_ids is not None:
                df = df[df['KG id'].isin(filter_ids)]

            df = df.sort_index(axis=1)

            if metrics:
                df = df.replace('-', np.nan)
                df = df.replace('Available',1)
                df = df.replace('offline',0)
                df = df.replace('True',1)
                df = df.replace('False',0)
                df = df.replace(True,1)
                df = df.replace(False,0)
                df = df.apply(pd.to_numeric, errors='coerce')
            else:
            # Delete the column to avoid errors
                columns_to_drop = ['Sparql endpoint', 'KG name', 'SPARQL endpoint URL', 'Dataset URL', 'KG id']
                df = df.drop(columns=columns_to_drop, errors='ignore')

        else:
            if 'Sparql endpoint' in columns_to_use:
                columns_to_use = [item for item in columns_to_use if item != "Sparql endpoint"] 
            if 'KG name' in columns_to_use:
                columns_to_use = [item for item in columns_to_use if item != "KG name"] 
            df = pd.read_csv(self.analysis_result,usecols=columns_to_use)
            df.replace('-', np.nan, inplace=True)

        if 'Volatility score' in df.columns:
            df.rename(columns={'Volatility score': 'Timeliness'}, inplace=True)
        
        if 'Representational-Consistency score' in df.columns:
            df.rename(columns={'Representational-Consistency score': 'Interoperability'}, inplace=True)

        if 'Representational-Conciseness score' in df.columns:
            df.rename(columns={'Representational-Conciseness score': 'Rep. Conciseness'}, inplace=True)
        if 'Representational-Conc.' in df.columns:
            df.rename(columns={'Representational-Conc.': 'Rep. Conciseness'}, inplace=True)


        if 'SPARQL endpoint URL' in df.columns:
            df.drop(columns=['SPARQL endpoint URL'], inplace=True)
        if 'Dataset URL' in df.columns:
            df.drop(columns=['Dataset URL'], inplace=True)
        if 'KG name' in df.columns:
            df.drop(columns=['KG name'], inplace=True)
        #if 'Sparql endpoint' in df.columns:
        #    df.drop(columns=['Sparql endpoint'], inplace=True)

        df.columns = df.columns.str.replace('score', '', regex=False)
        df.columns = df.columns.str.strip()
        if metrics:
            df = df.dropna(axis=1, how='all')

            df = df.dropna(axis=0, how='all')

        df_clean = df.dropna()
        rho = df.corr('spearman')
        pval = df.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*rho.shape)
        p = pval.map(add_significance_stars)
        
        final_matrix = pd.DataFrame()

        # Iterate through the columns to create pairs of correlation values and stars
        for col in rho.columns:
            # Append correlation values
            final_matrix[col] = rho[col].round(2)  
            # Append corresponding significance stars
            final_matrix[f"{col}_p-value"] = p[col]

        final_matrix.to_csv(f'{self.output_file}.csv')

    
    def calculate_spearman_correlation_matrix_metrics(self,columns_to_use,sparql_status = True, replace_columns = False):
        '''
            Generate the Spearman Correlation matrix by using the values in the columns columns_to_use from the CSV file.      

            :param columns_to_use: list of strings representing the names of the columns from which to take values to measure correlation.
            :param replace_columns: if True, columns that have a list or a boll value as their value will be transformed into a float
        '''
        if not 'LUZZU' in self.output_file and not 'Yummy' in self.output_file and not 'SPARQLES' in self.output_file and not 'luzzu' in self.output_file:
            columns_to_use.append('Sparql endpoint')
            columns_to_use.append('KG name')
            df = pd.read_csv(self.analysis_result,usecols=columns_to_use)

            #df = self.add_analysis_time_to_df(df)

            if replace_columns:
                df = utils.convert_metrics_to_float(df, metrics_as_list=['New terms defined in the dataset','metadata-media-type','Languages (query)',
                                                                'External links','Regex uri'],bool_metrics=['Ontology Hijacking problem'])
            for metric in columns_to_use:
                try:
                    df[metric] = df[metric].replace("insufficient data", np.nan)
                except KeyError:
                    continue

            if sparql_status:
                df = df[(df["Sparql endpoint"] == 1)] 
            else:
                df = df[(df["Sparql endpoint"] != 1)] 

            df = df.sort_index(axis=1)
                
            # Delete the column to avoid errors
            df = df.drop('Sparql endpoint', axis=1)
            df = df.drop('KG name',axis=1)
        else:
            df = pd.read_csv(self.analysis_result,usecols=columns_to_use)
            df.replace('-', np.nan, inplace=True)

        if 'Volatility score' in df.columns:
            df.rename(columns={'Volatility score': 'Timeliness'}, inplace=True)
        
        if 'Representational-Consistency score' in df.columns:
            df.rename(columns={'Representational-Consistency score': 'Interoperability'}, inplace=True)

        if 'Representational-Conciseness score' in df.columns:
            df.rename(columns={'Representational-Conciseness score': 'Rep. Conciseness'}, inplace=True)

        df.columns = df.columns.str.replace('score', '', regex=False)
        df_clean = df.dropna()
        rho = df.corr('spearman')
        pval = df.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*rho.shape)
        p = pval.map(add_significance_stars)
        
        final_matrix = pd.DataFrame()

        # Iterate through the columns to create pairs of correlation values and stars
        for col in rho.columns:
            # Append correlation values
            final_matrix[col] = rho[col].round(2)  
            # Append corresponding significance stars
            final_matrix[f"{col}_p-value"] = p[col]

        final_matrix.to_csv(f'{self.output_file}.csv')

def add_significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''  
