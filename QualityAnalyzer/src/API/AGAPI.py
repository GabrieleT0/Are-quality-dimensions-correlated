import requests
import os
import json
import utils
import base64

ENCODED_URL_TEMPLATE = "aHR0cDovL3d3dy5pc2lzbGFiLml0OjEyMjgwL2tnc2VhcmNoZW5naW5lL2JydXRhbFNlYXJjaD9rZXl3b3JkPQ=="
def decode_url_template():
    return base64.b64decode(ENCODED_URL_TEMPLATE).decode('utf-8')

def getMetadati(idKG):
    url_template = decode_url_template()
    url = url_template % idKG
    try:
        response = requests.get(url, verify=False)    
        if response.status_code == 200:
            response = response.json()
            results = response.get('results')
            return results
        else:
            print("Connection failed to AGAPI")
            return False
    except:
        print('Connection failed to AGAPI')
        return False

def getAllKg():
    url_template = decode_url_template()
    url = url_template % ''
    try:
        response = requests.get(url, verify=False)    
        if response.status_code == 200:
            print("Connection to API successful and data recovered")
            response = response.json()
            results = response.get('results')
            return results
        else:
            print("Connection failed")
            return False
    except:
        print('Connection failed')
        return False

def getSparqlEndpoint(metadata):
    if isinstance(metadata, dict):
        sparqlInfo = metadata.get('sparql')
        if not sparqlInfo:
            return False
        accessUrl = sparqlInfo.get('access_url')
        return accessUrl

def getNameKG(metadata):
    if isinstance(metadata, dict):
        title = metadata.get('title')
        return title
    else: 
        return False

def getIdByName(keyword):
    url_template = decode_url_template()
    url = url_template + keyword
    try:
        response = requests.get(url, verify=False)    
        if response.status_code == 200:
            print("Connection to API successful and data recovered")
            response = response.json()
            results = response.get('results')
            kgfound = []
            for i in range(len(results)):
                d = results[i]
                id = d.get('id')
                name = d.get('title')
                kgfound.append((id, name))
            return kgfound
        else:
            try: 
                return utils.load_kgs_metadata_from_snap()
            except:
                return False 
    except:
        try: 
            return utils.load_kgs_metadata_from_snap()
        except:
            return False