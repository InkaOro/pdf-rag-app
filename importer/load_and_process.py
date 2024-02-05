import os

from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings

from config import EMBEDDING_MODEL

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
print(docs)

OpenAIEmbeddings(
    model=EMBEDDING_MODEL,

)

