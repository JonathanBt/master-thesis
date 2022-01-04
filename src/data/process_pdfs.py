from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


def convert_pdf_to_txt(path_report):
    """This function extracts the text from a pdf file.

    Args:
        path_report (str): path to the pdf file to be processed

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
        text = 'Error'
    return text