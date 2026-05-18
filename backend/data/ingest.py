import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_pdfs(raw_dir: str = "data/raw", persist_dir: str = "data/vectorstore"):
    print("Starting PDF ingestion...")
    pdf_files = glob.glob(os.path.join(raw_dir, "*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in data/raw")
        return
        
    documents = []
    for file in pdf_files:
        print(f"Loading {file}")
        loader = PyPDFLoader(file)
        docs = loader.load()
        # Add source metadata
        for doc in docs:
            doc.metadata["source_file"] = os.path.basename(file)
        documents.extend(docs)
        
    print(f"Loaded {len(documents)} pages.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    
    print("Generating embeddings and building FAISS index...")
    # Using local HuggingFace embeddings so it works regardless of OpenAI/Anthropic API keys
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)
    print(f"FAISS index saved to {persist_dir}")

if __name__ == "__main__":
    ingest_pdfs()
