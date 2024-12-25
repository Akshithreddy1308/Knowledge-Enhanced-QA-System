import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load PDF files
def load_pdf_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, filename))
            documents.extend(loader.load())
    return documents

# Load CSV files
def load_csv_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename))
            for _, row in df.iterrows():
                documents.append(str(row.to_dict()))
    return documents

# Load Text files
def load_text_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                documents.append(file.read())
    return documents

# Process documents
def process_documents(documents):
    texts = []
    for doc in documents:
        if isinstance(doc, Document):
            texts.append(doc.page_content)
        elif isinstance(doc, str):
            texts.append(doc)
        else:
            raise ValueError(f"Unsupported document type: {type(doc)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_texts = text_splitter.create_documents(texts)
    return split_texts

# Embed and store in FAISS
def store_embeddings(texts):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(texts, embeddings)
    return db

# Dynamic pipeline for preprocessing uploaded documents
def process_and_store():
    docs = []
    docs.extend(load_pdf_files(UPLOAD_FOLDER))
    docs.extend(load_csv_files(UPLOAD_FOLDER))
    docs.extend(load_text_files(UPLOAD_FOLDER))
    processed_docs = process_documents(docs)
    db = store_embeddings(processed_docs)
    return db

if __name__ == "__main__":
    print("Starting preprocessing...")
    db = process_and_store()
    print("Preprocessing completed successfully!")