from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Load PDF
document_loader = PyPDFLoader('ShivamKrGupta.pdf')
documents = document_loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# Embeddings + Chroma
embeddings = HuggingFaceEmbeddings()
vector_db = Chroma.from_documents(chunks, embeddings)

# Retriever
retriever = vector_db.as_retriever()
query = input("Enter your query: ")

retrieved_docs = retriever.invoke(query)
retrieved_texts = "\n".join([doc.page_content for doc in retrieved_docs])

# LLM
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    provider="auto"
)
chat_model = ChatHuggingFace(llm=llm)

# FIXED PROMPT — includes context
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided context to answer the question."),
    ("user", "Context:\n{Context}\n\nQuestion: {question}")
])

# Chain
output_parser = StrOutputParser()
chain = prompt | chat_model | output_parser

# Invoke with both fields
response = chain.invoke({
    "Context": retrieved_texts,
    "question": query
})

print("\n===== RESPONSE =====")
print(response)
