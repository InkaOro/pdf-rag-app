import sys
import os

# Get the directory of the current file (__file__ refers to load_and_process.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the root of the project)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from dotenv import load_dotenv
load_dotenv()

from config import PG_COLLECTION_NAME
from operator import itemgetter
from typing import TypedDict


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.pgvector import PGVector

vector_store = PGVector(
    collection_name=PG_COLLECTION_NAME,
    connection_string=os.getenv("POSTGRES_URL"),
    embedding_function=OpenAIEmbeddings(),
)

template = """
Answer given the following context:
{context}
Question: {question}
"""

ANSWER_PROMPT=ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", streaming=True)

class RagInput(TypedDict):
    question: str

final_chain = (
    {
        "context": itemgetter("question") | vector_store.as_retriever(), "question": itemgetter("question")}
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
).with_types(input_type=RagInput)

FINAL_CHAIN_INVOKE = final_chain.invoke({"question":"What is the theme of the auctions?"})

print(FINAL_CHAIN_INVOKE)
