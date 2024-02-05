import sys
import os

# Get the directory of the current file (__file__ refers to load_and_process.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the root of the project)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Now you can import config

from config import EMBEDDING_MODEL, PG_COLLECTION_NAME
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

loader = DirectoryLoader(
    os.path.abspath("../source_docs"),
    glob="**/*.pdf",
    # use_multithreading=True,
    show_progress=True,
    max_concurrency=50,
    loader_cls=UnstructuredPDFLoader,
    sample_size=1
)

docs = loader.load()

embeddings= OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
)

text_splitter= SemanticChunker(
    embeddings=OpenAIEmbeddings()
)

chunks = text_splitter.split_documents(docs)

PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=PG_COLLECTION_NAME,
    connection_string="postgresql+psycopg://postgres@localhost:5432/pdf_rag_vectors",
    pre_delete_collection=True,
    )
