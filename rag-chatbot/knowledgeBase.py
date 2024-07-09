from io import BytesIO, StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


import os
import getpass

# setting the user_agent fo webBaseloader
os.environ['USER_AGENT'] = 'Mozilla/5.0'
os.environ["OPENAI_API_KEY"] = getpass.getpass()

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub




# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

len(docs[0].page_content)
# print(docs[0].page_content)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))
print(len(all_splits[0].page_content))
print(all_splits[10].metadata)


vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("What are the approaches to Task Decomposition?")

print(len(retrieved_docs))
print(retrieved_docs[0].page_content)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()
example_messages
print(example_messages[0].content)


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