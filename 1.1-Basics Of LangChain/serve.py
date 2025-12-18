from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY") # pyright: ignore[reportArgumentType]
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY") # pyright: ignore[reportArgumentType]

model = ChatGroq(model="llama-3.1-8b-instant")

### 1. Create Prompt Template
from langchain_core.prompts import ChatPromptTemplate

generic_template = "Translate the following into {language}:"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("user", "{text}")
    ]
)

### 2. Create Parser

parser = StrOutputParser()

### 3. Create Chain
chain = prompt | model | parser

### 4. App Definition
app = FastAPI(title="LangChain Server",
              version="1.0",
              description="Simple API Server using LangChain runnable interfaces")

### 5. Adding chain routes
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)