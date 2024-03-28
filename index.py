# index.py
import os
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_community.retrievers import LlamaIndexRetriever
from fastapi import FastAPI
from pydantic import BaseModel

if os.environ.get('OPENAI_API_KEY') is None:
  exit('You must provide an OPENAI_API_KEY env var.')

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = LlamaIndexRetriever(index=index.as_query_engine())

llm = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=2048)

memory = ConversationBufferWindowMemory(
  memory_key='chat_history',
  return_messages=True,
  k=3
)

conversation = ConversationalRetrievalChain.from_llm(
  llm=llm, 
  retriever=retriever,
  memory=memory,
  max_tokens_limit=1536  
)

class Prompt(BaseModel):
  question: str

app = FastAPI()

@app.post("/prompt")
async def query_chatbot(prompt: Prompt):
  response = conversation.invoke({'question': prompt.question})
  return response['answer']

if __name__=='__main__':
  import uvicorn
  uvicorn.run(app, host="localhost", port=8000)
