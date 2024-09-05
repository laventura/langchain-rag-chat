# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain

import openai

from dotenv import load_dotenv
import os
import shutil

# load env variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "./chroma"
DATA_PATH = "data/books" 


def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents) 
    save_to_chroma(chunks)

def load_documents():
    print(f"Loading documents from {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    for d in documents:
        print(d.metadata)
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                   chunk_overlap=200,
                                                   length_function=len,
                                                   add_start_index=True)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    return chunks

def save_to_chroma(chunks: list[Document]):
    # clear out the chroma directory first
    if os.path.exists(CHROMA_PATH):
        print(f"Clearing out {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)
    
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings, 
                                         persist_directory=CHROMA_PATH)
    # vector_store.save(CHROMA_PATH)
    vector_store.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

if __name__ == "__main__":
    main()