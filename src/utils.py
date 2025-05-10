import ast
import pandas as pd
import numpy as np
import os
from scipy.stats import shapiro
from variables import QUALITY_DIMENSIONS

here = os.path.dirname(os.path.abspath(__file__))

def convert_metrics_to_float(df,metrics_as_list,bool_metrics):
    """
        Convert the values in the pandas dataframe columns into a float to execute the calculation about the correlation.
        Lists are trasformed in integer (list size) and True/False into 1 or 0.

        :param df: padas dataframe with the columns to trasform
    """
    for metric in metrics_as_list:
        for idx, list_string in enumerate(df[metric]):
            try:
                list_elements = ast.literal_eval(list_string)  
                if isinstance(list_elements, list):
                    #df.at[idx, metric] = len(list_elements)
                    df.at[idx, metric] = 1
                else:
                    df.at[idx, metric] = 0
            except Exception as error:
                df.at[idx, metric] = 0 if list_string == '[]' else 1
        
        df[metric] = df[metric].replace("HTTP Error 502: Bad Gateway", np.nan)
        df[metric] = df[metric].replace("HTTP Error 504: Gateway Time-out", np.nan)
        df[metric] = df[metric].replace("HTTP Error 504: Gateway Timeout", np.nan)
        df[metric] = df[metric].replace("HTTP Error 502: Proxy Error", np.nan)
        df[metric] = df[metric].replace("insufficient data'", np.nan)
        
    # Create the P1 metric
    df['P1'] = df.apply(lambda row: 1 if (row['Author (metadata)'] != 'False' or (row['Publisher'] != '-' and row['Publisher'] != '[]' and row['Publisher'] != 'absent')) else 0, axis=1)
    df['P1'] = pd.to_numeric(df['P1'], errors='coerce')

    df.drop('Author (metadata)',axis=1,inplace=True)
    df.drop('Publisher',axis=1,inplace=True)

    for idx, value in enumerate(df['Extensional conciseness']):
        if isinstance(value,str):
            conc_value = value.split(' ')[0]
            df.at[idx, 'CN2'] = conc_value
    df['CN2'] = pd.to_numeric(df['CN2'], errors='coerce')

    df.drop('Extensional conciseness',axis=1,inplace=True)

    # Trasform the True/False values into 1 or 0
    for metric in bool_metrics:
        df[metric] = df[metric].map({'True': 1, 'False': 0}).fillna(0).astype(int)

    for metric in metrics_as_list:
        df.loc[df[metric] == 'False', metric] = 0

    # Trasform license string into 1 or 0
    df.loc[~df['License machine redeable (metadata)'].isin(['False', False, 'License not specified - notspecified -']), 'License machine redeable (metadata)'] = 1
    df.loc[~df['License human redeable'].isin(['-', 'absent','False',False]), 'License human redeable'] = 1
    df.loc[df['License human redeable'] == 'False', 'License human redeable'] = 0
    df.loc[df['License machine redeable (metadata)'] == 'False', 'License machine redeable (metadata)'] = 0
    df.loc[df['License machine redeable (metadata)'] == 'License not specified - notspecified -', 'License machine redeable (metadata)'] = 0
    df.loc[df['License machine redeable (metadata)'] == 'absent', 'License machine redeable (metadata)'] = 0
    df.loc[df['License human redeable'] == 'absent', 'License human redeable'] = 0
    df.loc[df['Regex uri'] == 'absent', 'Regex uri'] = 0
    df['Regex uri'] = df['Regex uri'].astype(str)
    df.loc[df['Regex uri'].str.contains(r'\W', regex=True, na=False), 'Regex uri'] = 1

    df.replace('-', np.nan, inplace=True)

    return df

def verify_normal_distribution(csv_file_path,columns_to_verify):
    df = pd.read_csv(csv_file_path)
    for col in columns_to_verify:
        data = df[col].dropna() 
        stat, p = shapiro(data)
        print(f'Statistic: {stat}, p-value: {p}')
        if p > 0.05:
            print(f"Data for {col} appears to be normally distributed (fail to reject H0).")
        else:
            print(f"Data for  {col} does NOT appear to be normally distributed (reject H0).")

def remove_sparql(s):
    if isinstance(s, str):  # Ensure it's a string before processing
        return s.replace('sparql', '')
    return s 

def p_value_to_numeric(p_symbol):
    """
    Converts p-value symbols to numeric values.
    
    :param p_symbol: String representation ('***', '**', '*', or '')
    :return: Corresponding numeric p-value
    """
    mapping = {'***': 0.001, '**': 0.01, '*': 0.05, '': 1.0}  
    return mapping.get(p_symbol, 1.0)

def get_always_sparql_up(csv_files):
    # Dictionary to keep track of availability per ID
    availability_tracker = {}

    # Process each file
    for file in csv_files:
        df = pd.read_csv(f'../data/quality_analysis_results/{file}.csv')
        
        # Make sure necessary columns exist
        if 'KG id' not in df.columns or 'Sparql endpoint' not in df.columns:
            continue  

        for _, row in df.iterrows():
            kg_id = row['KG id']
            status = row['Sparql endpoint']
            
            if kg_id not in availability_tracker:
                availability_tracker[kg_id] = True
            
            if status != 'Available':
                availability_tracker[kg_id] = False

    always_available_ids = [kg_id for kg_id, is_available in availability_tracker.items() if is_available]

    return always_available_ids

def get_always_observed_ids(first_analysis):
    all_ids = {}

    df = pd.read_csv(f'../data/quality_analysis_results/{first_analysis}.csv')

    if 'KG id' not in df.columns:
        return []  

    for _, row in df.iterrows():
        kg_id = row['KG id']
        
        if kg_id not in all_ids:
            all_ids[kg_id] = True

    return list(all_ids.keys())

def recover_other_info_from_ids(kgs_id,date_for_recover_info):
    print(kgs_id)
    result = {}
    df = pd.read_csv(f'../data/quality_analysis_results/{date_for_recover_info}.csv')
    for _, row in df.iterrows():
        kg_id = row['KG id']
        if kg_id in kgs_id:
            result[kg_id] = {
                'KG id': kg_id,
                'KG name': row.get('KG name', ''),
                'SPARQL endpoint URL': row.get('SPARQL endpoint URL', ''),
            }
    
    output_data =  list(result.values())
    pd.DataFrame(output_data).to_csv('../data/KGs_always_up_info.csv', index=False)