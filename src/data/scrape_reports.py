import pandas as pd
import numpy as np
from pathlib import Path
import requests

def scrape_reports(df_full, df_part, path_output):
    """This function scrapes pdf reports corresponding to the urls in the column 'CSR_URL' in df_part.
    It also updates the column CSR_Filename in df_full with the filename of the respective pdf file.

    Args:
        df_full (pandas dataframe): full dataframe containing the urls to be scraped.
        df_part (pandas dataframe): part of df_full containing the urls to be scraped in one iteration.
        path_output (str): path to store the scraped pdf files.

    Returns:
        None

    """

    # Specify request headers: needed to prevent some pdfs from not downloading correctly
    headers = {
        "User-Agent": "PostmanRuntime/7.20.1",
        "Accept": "*/*",
        "Cache-Control": "no-cache",
        "Postman-Token": "8eb5df70-4da6-4ba1-a9dd-e68880316cd9,30ac79fa-969b-4a24-8035-26ad1a2650e1",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "cache-control": "no-cache",
    }

    for index, row in df_part.iterrows():
        filename = row['CSR_Period_Relative'] + '_' + row['Identifier'] + '_' + '.pdf'
        pathname = Path(path_output + filename)
        try:
            response = requests.get(row['CSR_URL'], headers=headers, verify=False)
            # Check if link leads to a pdf file
            if 'application/pdf' in response.headers.get('content-type'):
                # If yes, scrape and add filename to df_full
                pathname.write_bytes(response.content)
                df_full.loc[index, 'CSR_Filename'] = filename
        except:
            df_full.loc[index, 'CSR_Filename'] = 'Error'





