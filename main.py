"""Main module for running retrieval-augmented generation (RAG) with Pinecone and LangChain."""

import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":

    def retrieve_and_answer_query(query: str) -> str:
        """
        Retrieve relevant documents and answer a query using RAG.

        This function implements a retrieval-augmented generation (RAG) pipeline that takes a
        user query, searches relevant documents from a Pinecone vector store using embeddings,
        and generates an answer using GPT-4. It combines document retrieval with LLM-based
        question answering to provide accurate, context-aware responses.
        """
        print("Retrieving started...")
        # overall flow:
        # 1. embedding user query
        # 2. search the pinecone index
        # 3. prompt the LLM to answer the user query
        # 4. generate the response

        # 1. embedding user query
        # Initialize OpenAI embeddings and LLM, create a simple chain to process the query
        # and print results
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Initialize Pinecone vector store with our index name and embeddings
        vectorstore = PineconeVectorStore(
            index_name=os.environ["INDEX_NAME"], embedding=embeddings
        )

        # Get a pre-made prompt template for retrieval QA from LangChain's hub
        # This template is optimized for question-answering using retrieved documents
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        # Create a chain that combines retrieved documents with the prompt
        # This chain will:
        # 1. Take the retrieved documents
        # 2. Combine them with the user's question using the prompt template
        # 3. Send to the LLM for answering
        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

        # Create the full retrieval chain that will:
        # 1. Take a user question
        # 2. Retrieve relevant documents from Pinecone
        # 3. Pass those docs to the combine_docs_chain created above
        retrieval_chain = create_retrieval_chain(
            retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
        )

        # Run the chain with our query and print the answer
        # The chain will:
        # 1. Take our query about Pinecone
        # 2. Find relevant docs in our Pinecone database
        # 3. Combine those docs with our query in a prompt
        # 4. Get GPT to answer based on the retrieved information
        response = retrieval_chain.invoke(input={"input": query})
        return response["answer"]

    def custom_template_based_rag(query: str) -> str:
        """
        Custom template-based RAG implementation.
        This is a RAG (Retrieval Augmented Generation) implementation because it:
        1. Retrieves relevant documents from a vector store (Pinecone)
        2. Augments the prompt with the retrieved context
        3. Generates a response using an LLM (GPT-4) based on the retrieved context
        """

        # 1. embedding user query
        # Initialize OpenAI embeddings and LLM, create a simple chain to process the query
        # and print results
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Initialize Pinecone vector store with our index name and embeddings
        vectorstore = PineconeVectorStore(
            index_name=os.environ["INDEX_NAME"], embedding=embeddings
        )

        template = """Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say "I don't know". don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.

        {context}
        Question: {question}

        Helpful Answer:"""

        custom_rag_prompt = PromptTemplate.from_template(template)

        rag_chain = (
            {
                "context": vectorstore.as_retriever() | format_docs,
                "question": RunnablePassthrough(),
            }
            | custom_rag_prompt
            | llm
        )

        answer = rag_chain.invoke(query)
        return answer.content

    # Example usage
    query = "What is Vector Database in machine learning?"
    # answer = retrieve_and_answer_query(query)
    # print(answer)

    answer = custom_template_based_rag(query)
    print(answer)
