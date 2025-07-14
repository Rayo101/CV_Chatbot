"""
Simple script to read some text data about my CV, split the text, 
compute the embeddings and then save all of this to a chroma database.

Author: Rayhaan Perin
Date: 13-07-2025

Changelist: 
            13-07-2025 --- Creation of script.
"""

import os 
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

def load_files(path: str) -> list[dict]:
    """
    Loads in the CV text data. 

    Input: 
        path (str) - The path to the folder with the text files.
    
    Return:
        (list) - The list od the text data with both the file name stored as "id" and the text under the "text" field.
    
    """
    documents = []
    for file_name in os.listdir(path):
        if file_name.endswith(".txt"):
            with open(file = path + "/" + file_name, mode = "r", encoding = "utf-8") as file:
                documents.append({"id": file_name, "text": file.read()})
    return documents

def token_text_splits(documents: list[dict], token_splitter: SentenceTransformersTokenTextSplitter) -> list:
    """
    This splits the data from the text file into chunks with a specific chunk size and chunk overlap size.

    Input: 
        documents (list) - The list of document text data
        token_splitter (SentenceTransformersTokenTextSplitter) - The text splitter from langchain.

    Return:
        (list) - The chuked documents

    """
    documentsChunk = []
    for doc in documents:
        documentsChunk.extend(token_splitter.split_text(doc['text']))
    return documentsChunk


def get_openAI_embeddings(text: str, client: OpenAI) -> str:
    """
    coverts text to embeddings using a specific model.

    Input:
        text (str) - The text to convert into embeddings. 
        client (openAI) - openAI client instance

    Return:
        (str) - 
    """
    response = client.embeddings.create(input = text, model = "text-embedding-3-small")
    return response.data[0].embedding

def create_embeddings(documents, client) -> None:
    """
    Adds extra field to the dictionary which contains the text embeddings. 

    Input:
        documents (list) - The list containing the id, text
        client (openAI) - openAI client instance

    Return:
        None
    """
    for doc in documents:
        doc["embedding"] = get_openAI_embeddings(doc['text'], client)

def insert_embeddings_to_db(documents: list[dict], collection: Collection) -> None:
    """
    This takes the documents "id" "text" and "embeddings" and saves the content to a persistant chroma database.

    Input: 
        documents (list) - The list containing the id, text
        collection (Collection) - a grouping mechanism for storing and managing embeddings, documents, and metadata

    Return:
        None
    """
    for doc in documents:
        collection.upsert(ids = [str(doc['id'])], documents = [doc['text']], embeddings = [doc["embedding"]])

def main():
    # Get api key
    load_dotenv() 
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Create client 
    client = OpenAI(api_key = OPENAI_API_KEY)

    # Specify embedding function
    embeddingFunction = embedding_functions.OpenAIEmbeddingFunction(api_key = OPENAI_API_KEY, model_name="text-embedding-3-small")

    # Create chroma database
    chromaClient  = chromadb.PersistentClient(path = "chroma_persistant_path")
    collectionName = "Rayhaan_CV"
    collection = chromaClient.get_or_create_collection(name = collectionName, embedding_function = embeddingFunction)

    # specify where text files are
    pathToFiles = "./Data/"

    # load files
    documents = load_files(path = pathToFiles)

    # Split text 
    tokenSplitter = SentenceTransformersTokenTextSplitter(chunk_size = 256, chunk_overlap = 32)
    tokenTextSplits = token_text_splits(documents = documents, token_splitter = tokenSplitter)

    # create the list which contains the dictionary which has the "id" and "text"
    splitDocuments = [{"id": i, "text": chunk} for i, chunk in enumerate(tokenTextSplits)]

    # create the embeddings
    create_embeddings(documents = splitDocuments, client = client)
    
    # save everything to the chroma database
    insert_embeddings_to_db(documents = splitDocuments, collection = collection)

if __name__ == "__main__":
    main()