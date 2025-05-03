import os

from dotenv import load_dotenv

# text loader is used to load the text file
from langchain_community.document_loaders import TextLoader

# embeddings are used to convert the text into a vector
from langchain_openai import OpenAIEmbeddings

# pinecone is used to store the embeddings
from langchain_pinecone import PineconeVectorStore

# text splitter is used to split the text into chunks
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":

    # Step 1: Ingesting the data
    # please checkout this image for the flow of the ingestion
    # ![Ingestion Flow](/data/ingestion_flow.png)
    print("Ingesting started...")
    # overall flow:
    # 1. load the text file,
    # 2. split the text into chunks,
    # 3. convert the text into a vector,
    # 4. store the embeddings in pinecone

    # load the text file
    loader = TextLoader("data/mediumblog1.txt")
    documents = loader.load()

    # split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # convert the text into a vector
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # store the embeddings in pinecone
    PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("Ingesting completed")

    # Step 2: Retrieving the data
    # please checkout this image for the flow of the retrieval
    # ![Retrieval Flow](/data/retrieve.png)
    print("Retrieving started...")
    # overall flow:
    # 1. embedding user query
    # 2. search the pinecone index
    # 3. prompt the LLM to answer the user query
    # 4. generate the response
