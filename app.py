from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import openai
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "mdrbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


#llm = OpenAI(temperature=0.4, max_tokens=500)
#prompt = ChatPromptTemplate.from_messages(
#    [
#        ("system", system_prompt),
#        ("human", "{input}"),
#    ]
#)

#question_answer_chain = create_stuff_documents_chain(llm, prompt)
#rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def generate_response(query, retrieved_docs):
    """
    Generate a response using OpenAI API based on retrieved documents.
    If no documents are found, return "I don’t know."
    """
    if not retrieved_docs:
        return "I do not know."

    documents = [doc.page_content for doc in retrieved_docs]

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use only the provided context to answer."},
        {"role": "user", "content": f"Context: {documents}\nQuestion: {query}\nAnswer:"}
    ]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content.strip()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    print(f"User Input: {user_input}")

    # Retrieve documents from VectorDB
    retrieved_docs = retriever.get_relevant_documents(user_input)

    # Generate response
    response = generate_response(user_input, retrieved_docs)
    print("Response:", response)

    return response

    # If no relevant documents are found, return "I don’t know"
    '''
    if not retrieved_docs:
        print("No relevant context found. Returning: I don't know.")
        return "I don’t know."

    # If context is found, process through RAG chain
    response = rag_chain.invoke({"input": user_input})
    print("Response:", response["answer"])

    return str(response["answer"])
    '''

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
