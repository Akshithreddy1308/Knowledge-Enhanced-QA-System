from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from preprocess import process_and_store
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")

# Dynamic database initialization
db = process_and_store()
retriever = db.as_retriever(search_kwargs={"k": 5})

# Custom Prompt Template
custom_prompt_template = """
You are a helpful assistant who answers questions **strictly based** on the provided context. 
Do not use any external knowledge or assumptions. If the context does not contain the answer, simply respond with:
"I cannot find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=custom_prompt_template,
)

# Set up Retrieval QA with the Custom Prompt
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# Store chat history
dialogue_history = []

def get_answer(query):
    response = qa.run(query)
    dialogue_history.insert(0, {"query": query, "response": response}) # Most recent query at the top
    return response