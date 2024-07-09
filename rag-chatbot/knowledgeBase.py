import getpass
import os

os.environ["MISTRAL_API_KEY"] = getpass.getpass()

from langchain_mistralai import ChatMistralAI

from io import BytesIO, StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser



llm = ChatMistralAI(model="mistral-large-latest")



# For PDF parsing
def getDataFromPDF(pdf_files):
    rag_docs = []
    for pdf_file in pdf_files:
        pdf_data = pdf_file.read()

        text_paras = []
        parser = PDFParser(BytesIO(pdf_data))
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        for page in PDFPage.create_pages(doc):
            output_string = StringIO()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            interpreter.process_page(page)
            page_text = output_string.getvalue()
            text_paras.extend(re.split(r'\n\s*\n', page_text))

        rag_docs_data = StringIterableReader().load_data(text_paras)
        rag_docs.extend(rag_docs_data)

    return rag_docs