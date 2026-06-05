from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

UPLOAD_DIR = "uploads"
DB_DIR = "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
groq_api_key = os.getenv("GROQ_API_KEY")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile"
)

@app.get("/")
def home():
    return {"msg": "RAG Backend Running"}

@app.post("/upload_pdf")
def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )

    return {"msg": "uploaded successfully"}

class Question(BaseModel):
    question: str

@app.post("/get_answer")
def get_answer(data: Question):

    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion:\n{input}"
    )

    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(data.question)

    return {"answer": result}