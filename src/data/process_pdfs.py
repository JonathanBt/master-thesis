from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import pikepdf
import string
import re
import nltk
import spacy 


def convert_pdf_to_txt(path_report):
    """This function extracts the text from a pdf file.

    Args:
        path_report (str): Path to the pdf file to be processed.

    Returns:
        text (str): Extracted text.

    """

    try:
        fp = open('data/pdf_reports/' + path_report, 'rb')
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        retstr = StringIO()
        codec = 'utf-8'
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        pages = PDFPage.get_pages(fp)

        for page in pages:
            interpreter.process_page(page)

        text = retstr.getvalue()

        fp.close()
        device.close()
        retstr.close()

    except:
        try:
            # Solve encrypted pdf error: Decrypt and overwrite
            pdf = pikepdf.open('data/pdf_reports/' + path_report, allow_overwriting_input=True)
            pdf.save('data/pdf_reports/' + path_report)

            fp = open('data/pdf_reports/' + path_report, 'rb')
            rsrcmgr = PDFResourceManager()
            laparams = LAParams()
            retstr = StringIO()
            codec = 'utf-8'
            device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            pages = PDFPage.get_pages(fp)

            for page in pages:
                interpreter.process_page(page)

            text = retstr.getvalue()

            fp.close()
            device.close()
            retstr.close()

        except:
            text = 'Error'

    return text


def clean_text(text):
    """This function performs basic cleaning of text extracted from a pdf file.

    Args:
        text (str): Text to be cleaned.

    Returns:
        text (str): Cleaned text.

    """

    # Remove non ASCII characters
    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    # Words may be split between lines, link them back together
    text = re.sub(r'\s?-\s?', '-', text)
    # Remove spaces prior to punctuation
    text = re.sub(r'\s?([,:;\.])', r'\1', text)
    # Remove URLs
    text = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', r' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove multiple dots
    text = re.sub(r'\.+', '.', text)
    # Remove trailing spaces
    text = text.strip()
      
    return text


def mask_companies_spacy(text):
    """This function masks company names using SpaCys Named Entity Recognizer.

    Args:
        text (str): Text to be masked.

    Returns:
        newString (str): Masked text.

    """

    nlp.max_length = len(text) + 100
    doc = nlp(text)
    newString = text
    for ent in reversed(doc.ents): # reversed to not modify the offsets of other entities when substituting
        if ent.label_ =='ORG':
            newString = newString[:ent.start_char] + ent.label_ + newString[ent.end_char:]
            
    return newString


def transform_text(text):
    """This function tokenizes text and transforms text to lowercase, removes everything except alphabetical characters, removes stopwords and single characters and performs lemmatization.

    Args:
        text (str): Text to be tokenized.

    Returns:
        text (str): Cleaned and tokenized text.

    """
    
    # Convert to lowercase
    text = text.lower()   
    # Remove everything except alphabetical characters 
    text = re.sub("[^a-zA-Z]"," ",text)     
    # Tokenize (convert from string to list)
    lst_text = text.split()
    # Remove Stopwords
    lst_text = [word for word in lst_text if word not in nltk.corpus.stopwords.words("english")]
    # Remove single characters
    lst_text = [word for word in lst_text if len(word)>1]
    # Lemmatisation (convert the word into root word)
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    lst_text = [lem.lemmatize(word) for word in lst_text]
    # Convert back to string from list
    text = " ".join(lst_text)
    
    return text