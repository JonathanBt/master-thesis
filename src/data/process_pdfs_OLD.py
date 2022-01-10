from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from datetime import timedelta, datetime
import pikepdf


def convert_pdf_to_txt(path_report):
    """This function extracts the text from a pdf file.

    Args:
        path_report (str): path to the pdf file to be processed

    Returns:
        text (str): Extracted text.

    """
    # Limit maximum time for text extraction to prevent pdfminer from getting stuck (5 minutes per pdf)
    endtime = datetime.utcnow() + timedelta(seconds=300)

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
            if datetime.utcnow() <= endtime:
                interpreter.process_page(page)
            else:
                break

        if datetime.utcnow() <= endtime:
            text = retstr.getvalue()
        else:
            text = 'Error'

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
                if datetime.utcnow() <= endtime:
                    interpreter.process_page(page)
                else:
                    break

            if datetime.utcnow() <= endtime:
                text = retstr.getvalue()
            else:
                text = 'Error'

            fp.close()
            device.close()
            retstr.close()

        except:
            text = 'Error'

    return text