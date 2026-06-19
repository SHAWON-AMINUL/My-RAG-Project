import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
# CharacterTextSplitter পরিবর্তন করে RecursiveCharacterTextSplitter ব্যবহার করা হয়েছে
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from{docs_path}...")

    #check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} docs not exist. Please create it and add your company files.")
    
    #Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    
    for i, doc in enumerate(documents[:2]):  #Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} charecters")

        # টার্মিনাল ক্র্যাশ রোধ করতে এনকোডিং সেফ প্রিন্ট
        preview = doc.page_content[:100].encode('utf-8', errors='replace').decode('utf-8')
        print(f"  Content preview: {preview}...")
        print(f"  metadata: {doc.metadata}")
 

        #print(f"  Content preview: {doc.page_content[:100]}...")
        #print(f"  metadata: {doc.metadata}")

    return documents




def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    #text_splitter = CharacterTextSplitter(
    #  এখানে CharacterTextSplitter কেটে RecursiveCharacterTextSplitter বসিয়ে দিন
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,


        # Recursive splitter বুদ্ধিমান উপায়ে প্যারাগ্রাফ, নতুন লাইন, ডট বা স্পেস দেখে লেখা ছোট করে
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} charecters")
            print(f"Contect:")
            #  FIXED: এখানে টার্মিনাল যাতে Unicode এরর না দেয় তার ব্যবস্থা করা হয়েছে
            #safe_content = chunk.page_content.encode('utf-8', errors='replace').decode('utf-8')
            #print(safe_content)
            #print(chunk.page_content)
            print("_" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

        return chunks
    

def create_vector_store(chunks,persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore

def main():
    print("Main Function")

    #1. Loading the files
    documents = load_documents(docs_path="docs")

    #2. Chunking the files 
    chunks = split_documents(documents)

    #3. Embedding and Storing in Vector DB
    vectorstore = create_vector_store(chunks)



if __name__ == "__main__":
    main()
