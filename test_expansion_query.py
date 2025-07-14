"""
Simple script to test the RAG model. There are no comments about the 
openAI requests since it's very well documented. Waste of space anyway. 

Author: Rayhaan Perin
Date: 13-07-2025

Changelist: 
            13-07-2025 --- Creation of script.
"""

from openai import OpenAI
from dotenv import load_dotenv
import os 
from chromadb.utils import embedding_functions
import chromadb

def initial_query(client, query, model = "gpt-4.1-nano"):
    prompt = """You are a helpful assistance who is going to be explaining what you know about Rayhaan Perin a physics 
    PhD student. The questions will be on CV and resume topics such as work experience, skills, technical experience, education etc and other stuff related to a job. This is almost as if it's a virtual interview."""

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    response = client.chat.completions.create(
        model = model,
        messages = messages,
    )

    content = response.choices[0].message.content

    return content

def refined_query(client, query, chunks, model = "gpt-4.1-nano"):
    chunks = "\n\n".join(chunks)
    prompt = """You are a helpful assistance who is going to be explaining what you know about Rayhaan Perin a physics 
    PhD student. The questions will be on CV and resume topics such as work experience, skills, technical experience, education etc and other stuff related to a job. This is almost as if it's a virtual interview. Do not lie. If you genuinly don't know then say so.\n""" + "Context:\n" + chunks + "\n" + "Question:\n" + query

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    response = client.chat.completions.create(
        model = model,
        messages = messages,
    )

    content = response.choices[0].message.content

    return content

def main():
    # Load API key
    load_dotenv() 
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Create client
    client = OpenAI(api_key = OPENAI_API_KEY)

    # Specify naive query and await response
    query = "What relevant skills does Rayhaan Perin have?"
    response = initial_query(client = client, query = query)

    # Combined query with naive response
    combinedQueryResponse = f"{query} {response}"

    # Specify embedding function
    embeddingFunction = embedding_functions.OpenAIEmbeddingFunction(api_key = OPENAI_API_KEY, model_name="text-embedding-3-small")

    # Connect to chroma database
    chromaClient  = chromadb.PersistentClient(path = "chroma_persistant_path")
    collectionName = "Rayhaan_CV"
    chromaCollection = chromaClient.get_collection(collectionName, embedding_function = embeddingFunction)
    
    # Query said chroma database
    queryDatabase = chromaCollection.query(query_texts = combinedQueryResponse, n_results = 3, include = ["documents"])
    queriedDocuments = queryDatabase["documents"][0]

    # Get refined RAG response
    refinedAnswer = refined_query(client = client, chunks = queriedDocuments, query = query)

    # Print answer
    print(f"Question:\n{query}")
    print("\n\n")
    print(f"Answer:\n{refinedAnswer}")

if __name__ == "__main__":
    main()