import streamlit as st
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# ---------------------------
# Streamlit Setup
# ---------------------------

st.set_page_config(
    page_title="Python PDF RAG Chatbot"
)

st.title("📚 Python PDF RAG Chatbot")


# ---------------------------
# Load Environment Variables
# ---------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


if not api_key:
    st.error("Grok API key not found. Check your .env file.")
    st.stop()



# ---------------------------
# 1. Load PDF
# ---------------------------

loader = PyPDFLoader(
    "data/python.pdf"
)

documents = loader.load()


st.success(
    f"Loaded {len(documents)} pages successfully!"
)



# ---------------------------
# 2. Split PDF into Chunks
# ---------------------------

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=100

)


chunks = splitter.split_documents(documents)



# Remove Table of Contents

chunks = [

    chunk for chunk in chunks

    if "Table of Contents" not in chunk.page_content

]


st.success(
    f"Created {len(chunks)} chunks!"
)



# ---------------------------
# 3. Create Embeddings
# ---------------------------

embedding_model = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"

)


st.success(
    "Embedding model loaded!"
)



# ---------------------------
# 4. Create FAISS Database
# ---------------------------

vector_db = FAISS.from_documents(

    documents=chunks,

    embedding=embedding_model

)


st.success(
    "FAISS Vector Database Created!"
)



# ---------------------------
# 5. Connect Grok LLM
# ---------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

st.success(
    "Grok connected!"
)



# ---------------------------
# 6. User Question
# ---------------------------

question = st.text_input(
    "Ask something about Python:"
)



# ---------------------------
# 7. RAG Pipeline
# ---------------------------

if question:


    # Retrieve relevant chunks

    results = vector_db.similarity_search_with_score(

        question,

        k=3

    )


    # Extract text

    context = "\n\n".join(

        [

            doc.page_content

            for doc, score in results

        ]

    )



    # Prompt for Grok

    prompt = f"""

You are a Python tutor.

Answer the question using ONLY the provided context.

If the answer is not present in the context,
say "I could not find this information in the PDF."


Context:

{context}


Question:

{question}


Give a simple and accurate explanation.

"""



    # Generate answer

    response = llm.invoke(prompt)



    st.write("## 🤖 Answer")

    st.write(response.content)



    # Optional source display

    with st.expander("📚 Retrieved Sources"):

        for i, (doc, score) in enumerate(results):

            st.write(f"### Source {i+1}")

            st.write(doc.page_content)

            st.write("Similarity score:", score)