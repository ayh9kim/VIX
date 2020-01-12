# -*- coding: utf-8 -*-
"""
@author: ayh9kim
"""
# Import
import os
import time
from datetime import datetime
import requests
import pandas as pd

# generic request function
def request_url(req_endpoints, req_headers):
    for url in req_endpoints:
        # measure duration to slow down the loop to avoid jam
        t0 = time.time()
        
        # session if needed
        '''
        session = requests.Session()
        session.post(request_session_url, data=request_session_credentials)
        r = session.get(url)
        '''
        
        # no session needed
        # try requesting
        try:
            resp= requests.get(url, headers=req_headers) #timeout=10 means wait up to 10 seconds
            
        # error message
        except:
            print("Error: ")
        
        # get the difference between before and after the request, in seconds
        resp_delay = time.time() - t0
        
        # wait 10x longer than it took them to respond
        time.sleep(10 * resp_delay)
        
    # output response
    return(resp)

# vix data parser
def vix_data_parser(resp):
    # convert the response JSON into a structured python `dict()`
    response_data = resp.json()
    
    # For VIX: Date, F1-F2, F4-F7, Spot-F1 RollYield, Vix Spot, F1, F2, F4, F7
    lstCol = ['Date', 'F1-F2 Contango', 'F4-F7 Contango', 'Spot-F1 RY', 'Spot', 'F1', 'F2', 'F4', 'F7']
    parsed_data = pd.DataFrame.from_records(response_data, columns=lstCol) 
    
    parsed_data.iloc[:, 1:] = parsed_data.iloc[:,1:].apply(pd.to_numeric)
    
    return(parsed_data)
    

if __name__ == "__main__":
    # Init
    # os.chdir()
    
    # Request parameters
    request_headers = {"User-Agent": "Test"} 
    request_endpoints = ["http://vixcentral.com/ajax_get_contango_data/"]
    
    resp = request_url(request_endpoints, request_headers)
    data = vix_data_parser(resp)
    
    data.to_csv('data_VIX_' + datetime.today().strftime('%Y%m%d') + '.csv')
    
