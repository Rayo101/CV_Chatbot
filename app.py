from flask import Flask, render_template, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
import chromadb

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    # Get the form input
    msg = request.form["msg"]
    query = msg    
    
    # Generate LLM output
    response = initial_query(client = client, query = query)

    # Form some composed of the question and LLM response
    combinedQueryResponse = f"{query} {response}"

    # Create embedding function
    embeddingFunction = embedding_functions.OpenAIEmbeddingFunction(api_key = OPENAI_API_KEY, model_name="text-embedding-3-small")

    # Open chroma database
    chromaClient  = chromadb.PersistentClient(path = "chroma_persistant_path")
    collectionName = "Rayhaan_CV"
    chromaCollection = chromaClient.get_collection(collectionName, embedding_function = embeddingFunction)

    # Query Database whith combined query
    queryDatabase = chromaCollection.query(query_texts = combinedQueryResponse, n_results = 3, include = ["documents"])
    queriedDocuments = queryDatabase["documents"][0]

    # Hopefully get better answer than naive RAG. 
    refinedAnswer = refined_query(client = client, chunks = queriedDocuments, query = query)
    return refinedAnswer

def initial_query(client: openAI, query: str, model = "gpt-4.1-nano") -> str:
    prompt = """You are a helpful assistance who is going to be explaining what you know about Rayhaan Perin a physics 
    PhD student. The questions will be on CV and resume topics such as work experience, skills, technical experience, education etc and other stuff related to a job. Please answer as if you're explaining the CV to a potential job recruiter."""

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

def refined_query(client: openAI, query: str, chunks: list,  model = "gpt-4.1-nano"):
    chunks1 = "\n\n".join(chunks)
    prompt = """You are a helpful assistance who is going to be explaining what you know about Rayhaan Perin a physics 
    PhD student. The questions will be on CV and resume topics such as work experience, skills, technical experience, education etc and other stuff related to a job. This is almost as if it's a virtual interview. Do not lie. If you genuinly don't know then say so. Please answer as if you're explaining the CV to a potential job recruiter. Your name is CV bot.\n""" + "Context:\n" + chunks1 + "\n" + "Question:\n" + query

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

if __name__ == '__main__':

    # Load API key
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # create client.
    client = OpenAI(api_key = OPENAI_API_KEY)

    # start the app
    app.run()
