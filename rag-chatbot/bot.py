import os
import getpass
import openai
import logging
# openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
load_dotenv()
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, get_response_synthesizer
# from llama_index.core= import Response
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_parse import LlamaParse
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="princeton-nlp/sup-simcse-roberta-base")

filename_fn = lambda filename: {"file_name": filename} # setting meta data for each document automatically. can add other details to this dictionary.
documents = SimpleDirectoryReader("./data", file_metadata=filename_fn).load_data()
# model.encode(documents)
# documents2 = LlamaParse(result_type="markdown").load_data("./data") # takenote of the second parameter. not given in document



# Setting our custom settings for chunking our document
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=15) # we are using the above text splitting settings locally, for setting a global setting refer the documentations


# saving to disk
db = chromadb.PersistentClient(path="./chroma_db") # initialising client
chroma_collection = db.get_or_create_collection("rag-chatbot") # get collection
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)# assign chroma as the vector_store to the context
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, transformations=[text_splitter], 
    storage_context=storage_context, 
    show_progress=True, 
    embed_model=embed_model,
) # generating embedding for the 1st time and storing them in chroma db

# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("rag-chatbot")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

# this one is for the each vector index.
individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index.as_query_engine(),
        metadata=ToolMetadata(
            name=f"RAGToolForDifferentArticleInADocument",
            description=f"useful for when you want to answer queries about the given documents",
        ),
    )
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools,
    llm=OpenAI(model="gpt-3.5-turbo"),
)

# This one is for the sub-question that is a part of the query
query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="subQuestionQueryQngine",
        description="useful for when you want to answer queries that require analyzing all the articles in the document",
    ),
)

tools = individual_query_engine_tools + [query_engine_tool]

agent = OpenAIAgent.from_tools(tools, verbose=True)
# agent = OpenAIAgent.from_tools(tools)  # verbose=False by default

while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = agent.chat(text_input)
    print(f"Agent: {response}")

"""
# The below code for a simple rag

# A breakup of index.as_query_engine()
# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

# query
response = query_engine.query("What did the author do growing up?")
print(response)
"""