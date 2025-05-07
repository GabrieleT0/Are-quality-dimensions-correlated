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
    df = pd.read_csv(csv_file_path)
    categories = {
        "Accessibility" : {
            "Availability" : 0,
            "Licensing" : 0,
            "Interlinking" : 0
        },
        "Representational" : {
            "Representational-Conc." : 0,
            "Interoperability" : 0,
            "Interpretability" : 0,
            "Versatility" : 0,
            "Understandability" : 0
        },
        "Trust": {
            "Believability" : 0
        },
        "Intrinsic": {
            "Conciseness" : 0,
            "Consistency" : 0,
            "Accuracy" : 0,
        }
    }
    for key in categories:
        category = categories[key]
        dimensions = category.keys()
        if len(dimensions) > 1:
            for metric in dimensions:
                df[metric] = df[metric].astype(float)
            df[key] = df[dimensions].sum(axis=1) / len(dimensions)
        else:
            df[key] = df[dimensions]

    columns = list(categories.keys())
    columns.append('Namespace')

    df.to_csv(output_file,columns=columns,index=False)

def merge_dimensions_to_categories(csv_file_path):
    QUALITY_DIMENSIONS.append('Sparql endpoint')
    QUALITY_DIMENSIONS.append('KG name')
    QUALITY_DIMENSIONS.append('SPARQL endpoint URL')
    QUALITY_DIMENSIONS.append('Dataset URL')
    df = pd.read_csv(csv_file_path,usecols=QUALITY_DIMENSIONS)
    file_name = os.path.basename(csv_file_path).split('.')[0]
    categories = {
        "Accessibility" : {
            "Availability score" : 0,
            "Licensing score" : 0,
            "Interlinking score" : 0,
            "Security score" : 0,
            "Performance score" : 0
        },
        "Representational" : {
            "Representational-Conciseness score" : 0,
            "Representational-Consistency score" : 0,
            "Interpretability score" : 0,
            "Versatility score" : 0,
            "Understandability score" : 0
        },
        "Trust": {
            "Believability score" : 0,
            "Reputation score" : 0,
            "Verifiability score" : 0
        },
        "Intrinsic": {
            "Conciseness score" : 0,
            "Consistency score" : 0,
            "Accuracy score" : 0,
        },
        "Dataset dynamicity" :{
            "Currency score" : 0,
            "Volatility score" : 0
        },
        "Contextual" : {
            "Completeness score" : 0,
            "Amount of data score" : 0,
        }
    }

    for key in categories:
        category = categories[key]
        dimensions = category.keys()
        if len(dimensions) > 1:
            for metric in dimensions:
                df[metric] = df[metric].astype(float)
            df[key] = df[dimensions].sum(axis=1) / len(dimensions)
        else:
                df[key] = df[dimensions]
    columns_to_write = list(categories.keys())
    columns_to_write.append('Sparql endpoint')
    columns_to_write.append('KG name')
    columns_to_write.append('SPARQL endpoint URL')
    columns_to_write.append('Dataset URL')
    df.to_csv(os.path.join(here,f'../data/quality_analysis_results/KGHB/{file_name}_only_categories.csv'),columns=columns_to_write,index=False)

    df = pd.read_csv(csv_file_path)
    categories = {
        "Accessibility" : {
            "Availability" : 0,
            "Interlinking" : 0,
            "Performance" : 0,
            "Licensing" : 0,
        },
        "Representational" : {
            "Interoperability" : 0,
            "Versatility" : 0,
            "Representational-Conc." : 0
        },
        "Contextual" : {
            "Amount of data" : 0
        },
        "Dataset dynamicity" : {
            "Currency" : 0
        }
    }
    for key in categories:
        category = categories[key]
        dimensions = category.keys()
        if len(dimensions) > 1:
            for metric in dimensions:
                df[metric] = df[metric].astype(float)
            df[key] = df[dimensions].sum(axis=1) / len(dimensions)
        else:
            df[key] = df[dimensions]
    
    columns = list(categories.keys())
    columns.append('description_url')
    columns.append('endpoint_url')

    df.to_csv(os.path.join(here,f'../data/quality_analysis_results/YummyData/YummyData_analysis_data_by_category.csv'),columns=columns,index=False)


def merge_metrics_to_dim_kghb(csv_file_path):
    df = pd.read_csv(csv_file_path)
    categories = {
        "Availability" : {
            "URIs Deferenceability" : 0,
        },
        "Interlinking" : {
            "sameAs chains" : 0,
        },
        "Performance" : {
            "latency" : 0
        },
        "Accuracy" : {
            "malformend datatype" : 0,
            "inverse functional propery violation" : 0
        },
        "Conciseness" : {
            "extensional conciseness" : 0
        },
        "Consistency" : {
            "disjoint value" : 0
        },
        "Understandability": {
            "vocabularies" : 0,
            "labels" : 0,
            "URI regex" : 0,
        },
        "Interpretability" : {
            "blank nodes" : 0,
            "Use of RDF structures" : 0,
        },
        "Rep. Conciseness" : {
            "uris length" : 0
        },
        "Currency" : {
            "modification date" : 0
        },
        "Verifiability" : {
            "publisher" : 0
        },
        "Believability" : {
            "trust value" : 0
        }
    }
    for key in categories:
        category = categories[key]
        dimensions = category.keys()
        if len(dimensions) > 1:
            for metric in dimensions:
                df[metric] = df[metric].astype(float)
            df[key] = df[dimensions].sum(axis=1) / len(dimensions)
        else:
            df[key] = df[dimensions]
    
    categories['Sparql endpoint'] = {}
    categories['License machine redeable'] = {}
    categories['License human redeable'] = {}
    categories['number of triples'] = {}
    categories['number of entities'] = {}
    categories['number of properties'] = {}
    categories['new vocabularies'] = {}
    categories['new terms'] = {}
    categories['serialization formats'] = {}
    categories['langauges'] = {}
    categories['KG name'] = {}
    #categories['SPARQL endpoint URL'] = {}
    #categories['Dataset URL'] = {}

    df.to_csv(os.path.join(here,f'../data/quality_analysis_results/KGHB/KGHB_common_dim-metrics.csv'),columns=categories.keys(),index=False)



    df = pd.read_csv(csv_file_path)
    categories = {
        "Availability" : {
            "A3" : 0
        },
        "Rep. Conciseness" : {
            "RC1" : 0,
        },
        "Interpretability" : {
            "RC2" : 0,
            "IN4" : 0,
        },
        "Believability" : {
            "P1" : 0,
        },
        "Understandability" : {
            "U1" : 0,
            "U3" : 0,
            "U3" : 0
        },
        "Conciseness" : {
            "CN2" : 0
        },
        "Consistency": {
            "CS1" : 0,
            "CS6" : 0,
        },
        "Accuracy" : {
            "CS9" : 0,
        },
        "Performance" : {
            "PE3" : 0
        },
    }
    df.replace('-', np.nan, inplace=True)
    for key in categories:
        category = categories[key]
        dimensions = category.keys()
        if len(dimensions) > 1:
            for metric in dimensions:
                df[metric] = df[metric].astype(float)
            df[key] = df[dimensions].sum(axis=1) / len(dimensions)
        else:
            df[key] = df[dimensions]
    
    categories['IO1'] = {}
    categories['V1'] = {}
    categories['V2'] = {}
    categories['L1'] = {}
    categories['L2'] = {}

    columns = list(categories.keys())
    columns.append('Namespace')

    df.to_csv(os.path.join(here,f'../data/quality_analysis_results/LUZZU/LUZZU_common_dim-metrics.csv'),columns=columns,index=False)

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

def max_corr_highest_p(data):
    """
    Finds the tuple with the highest absolute correlation value, but prioritizes correlation significance (also if p-value is higher).

    :param data: List of tuples (correlation, p-value)
    :return: Tuple with max absolute correlation and highest p-value
    """
    if not data:
        return None  # Handle empty list case
    
    # Filter out tuples with NaN correlation values
    valid_data = [t for t in data if not np.isnan(t[0][0]) and not isinstance(t[1][0],float)]

    if len(valid_data) == 0:
        valid_data = [t for t in data if not np.isnan(t[0][0])]
        if len(valid_data) == 0:
            return None
        max_corr = max(valid_data, key=lambda t: abs(t[0][0]))[0]
        max_corr_tuples = [t for t in data if abs(t[0][0]) == abs(max_corr[0])]
        return max(max_corr_tuples, key=lambda t: p_value_to_numeric(t[1][0]))

    # Find the maximum absolute correlation value
    max_corr = max(valid_data, key=lambda t: abs(t[0][0]))[0]

    # Filter tuples that have this maximum absolute correlation
    max_corr_tuples = [t for t in valid_data if abs(t[0][0]) == abs(max_corr[0])]

    # Return the one with the highest p-value (largest numeric value)
    return min(max_corr_tuples, key=lambda t: p_value_to_numeric(t[1][0]))

def max_corr_lowest_p(data):
    """
    Finds the tuple with the highest absolute correlation value, but prioritizes statistical significance (lowest p-value).

    :param data: List of tuples (correlation, p-value)
    :return: Tuple with max absolute correlation among those with the lowest p-value
    """
    # Filter out tuples with NaN correlation values
    valid_data = [t for t in data if not np.isnan(t[0][0]) and not isinstance(t[1][0],float)]

    if len(valid_data) == 0:
        valid_data = [t for t in data if not np.isnan(t[0][0])]
        if len(valid_data) == 0:
            return None
        # If p-value is NaN, evaluate only the correlation and get the value with the highest correlation
        max_corr = max(valid_data, key=lambda t: abs(t[0][0]))[0]
        max_corr_tuples = [t for t in data if abs(t[0][0]) == abs(max_corr[0])]
        return max(max_corr_tuples, key=lambda t: p_value_to_numeric(t[1][0]))

    # Find the tuple with the lowest p-value (most statistically significant)
    min_p_value = min(valid_data, key=lambda t: p_value_to_numeric(t[1][0]))[1]

    # Get all tuples with this minimum p-value
    min_p_tuples = [t for t in valid_data if p_value_to_numeric(t[1][0]) == p_value_to_numeric(min_p_value[0])]

    # Among those, select the one with the highest absolute correlation
    return max(min_p_tuples, key=lambda t: abs(t[0][0]))

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