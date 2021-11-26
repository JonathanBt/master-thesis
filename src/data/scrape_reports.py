import pandas as pd
from pathlib import Path
import requests
from bs4 import BeautifulSoup


def scrape_pdf_reports(df, path_output):
    """This function scrapes pdf reports corresponding to the urls in the column 'CSR_URL' in df.
    It also updates the column CSR_Filename in df with the filename of the respective pdf file.

    Args:
        df (pandas dataframe): dataframe containing the urls to be scraped in the column 'CSR_URL'.
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

    for index, row in df.iterrows():
        filename = str(row['ID']) + '_' + str(row['Identifier']) + '_' + str(row['Financial_Period_Absolute']) + '.pdf'
        pathname = Path(path_output + filename)
        try:
            response = requests.get(row['CSR_URL'], headers=headers, verify=False, timeout=10)
            # Check if link leads to a pdf file
            if 'application/pdf' in response.headers.get('content-type'):
                # If yes, scrape and add filename to df_full
                pathname.write_bytes(response.content)
                df.loc[index, 'CSR_Filename'] = filename
            else:
                df.loc[index, 'CSR_Filename'] = 'Error'
        except:
            df.loc[index, 'CSR_Filename'] = 'Error'


def scrape_urls_responsibilityreports_website():
    """This function scrapes the website responsibilityreports.com and returns URLs to all reports for all companies.
    In addition to the URLs, it also stores the link to the page, company name, ticker, and year of the report in a df.

    Args:

    Returns:
        df (pandas dataframe): dataframe containing the urls of the CSRs.

    """

    # Get links to all company pages
    url = "https://www.responsibilityreports.com/Companies?a=#"
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    links = [
        "https://www.responsibilityreports.com" + a["href"]
        for a in soup.select('a[href^="/Company"]')
    ]

    # Iterate over all companies
    list_ = []
    for link in links:
        soup = BeautifulSoup(requests.get(link).content, "html.parser")
        # Get company name
        company_name = soup.select_one("h1").get_text(strip=True)
        # Get ticker
        ticker = soup.select_one(".ticker_name")
        if ticker:
            ticker = ticker.get_text(strip=True)
        else:
            ticker = ''
        # Get exchange
        exchange = soup.select_one(".right")
        if exchange:
            exchange = exchange.get_text(strip=True).replace('Exchange', '').replace('More', '')
        else:
            exchange = ''
        # Get most recent report
        try:
            CSR_Year = int(soup.select_one(".bold_txt").text[0:4])
        except:
            CSR_Year = ''
        try:
            CSR_URL = 'https://www.responsibilityreports.com' + soup.select_one(".btn_form_10k")['href']
        except:
            CSR_URL = ''
        # Add row for most recent report
        dict_ = {
            "Link": link,
            "Company_Name": company_name,
            "Ticker": ticker,
            "Exchange": exchange,
            "Financial_Period_Absolute": CSR_Year,
            "CSR_URL": CSR_URL
        }
        list_.append(dict_)
        # Add rows for historic reports (if available)
        for div in soup.select(".text_block"):
            try:
                CSR_Year = int(div.select_one('span').text[0:4])
            except:
                CSR_Year = ''
            try:
                CSR_URL = 'https://www.responsibilityreports.com' + div.select('a')[0]['href']
            except:
                CSR_URL = ''
            dict_ = {
                "Link": link,
                "Company_Name": company_name,
                "Ticker": ticker,
                "Exchange": exchange,
                "Financial_Period_Absolute": CSR_Year,
                "CSR_URL": CSR_URL
            }
            list_.append(dict_)

    # Create df
    df = pd.DataFrame(list_)

    return df
