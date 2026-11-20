import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="Docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    #check if this file exist
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist")

    folder_path = Path("Docs")
    documents = [
        Document(
            page_content=file.read_text(encoding="utf-8", errors="ignore"),
            metadata={
                "source": str(file),
                "filename": file.name,
            },
        )
        for file in folder_path.glob("*.txt")
    ]


    print(f"Loaded {len(documents)} documents.")

    # #load all files from doc directory
    # loader= Document(
    #     path=docs_path,
    #     glob="*.txt",
    #     page_content=
    #     loader_cls=TextLoader
    # )
    # documents= loader.load()

    # if len(documents)==0:
    #     raise FileNotFoundError(f"No .txt file found in {docs_path}")

    for i, doc in enumerate(documents[:3]):
       print(f"\n Document {i+1}:")
       print(f" Source : {doc.metadata['source']}")
       print(f" Content length : {len(doc.page_content)} characters")
       print(f" Content Preview : {doc.page_content[:100]}...")
       print(f" metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into Chunks")

    text_splitter= CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks= text_splitter.split_documents(documents)

    if chunks:
        for i,chunk in enumerate(chunks[:5]):
            print(f"\n--- chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content")
            print(chunk.page_content)
            print("-" *50)
        if len(chunks)>5 :
            print(f"\n... and {len(chunks)-5} more chunks")
    return chunks

def create_vector_store(chunks,persist_directory="db/chromadb"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing into vector DB...")

    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")

    #creating ChromaDB vector Store
    print("----Creating vector Store----")
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("-------Finished Creating Vector Store----------")

    print(f" Vector Store created and saved to {persist_directory}")

    return vectorstore

def main():
    print("Main Function")
     #1.Loading the file
    documents=load_documents(docs_path="Docs")
    # print(documents)

    #2.chunking the files
    chunks= split_documents(documents)

    #3.embedding and store into vector DB
    vectorstore=create_vector_store(chunks)

if __name__=="__main__":
     main()