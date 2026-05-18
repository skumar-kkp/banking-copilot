from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

DATA_FOLDER="data/raw"

def load_documents():

    docs=[]

    pdf_files=[
        "RetailLoan_ProductsGuide_NovaNorth.pdf",
        "AccountTerms_Conditions_NovaNorth.pdf",
        "KYC_AML_Policy_NovaNorth.pdf",
        "FairPracticesCode_CustomerRights_NovaNorth.pdf",
        "RBI_RegulatoryGuidelines_Summary_NovaNorth.pdf",
        "FraudAlert_Types_ResolutionGuide_NovaNorth.pdf",
        "DigitalBanking_UPI_FeatureGuide_NovaNorth.pdf",
        "Banking_Finance_Glossary_NovaNorth.pdf"
    ]

    for file in pdf_files:

        loader=PyPDFLoader(
            os.path.join(DATA_FOLDER,file)
        )

        pages=loader.load()

        for page in pages:

            page.metadata["source"]=file

            if "Loan" in file:
                page.metadata["intent"]="loan"

            elif "KYC" in file:
                page.metadata["intent"]="kyc"

            elif "Fraud" in file:
                page.metadata["intent"]="fraud"

            elif "RBI" in file:
                page.metadata["intent"]="regulatory"

            else:
                page.metadata["intent"]="general"

        docs.extend(pages)

    return docs


def create_chunks(documents):

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    chunks=splitter.split_documents(documents)

    return chunks


def build_faiss():

    docs=load_documents()

    chunks=create_chunks(docs)

    embeddings=OpenAIEmbeddings()

    db=FAISS.from_documents(
        chunks,
        embeddings
    )

    db.save_local("faiss_index")

    print("FAISS created")


if __name__=="__main__":
    build_faiss()
