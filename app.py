# OS : Linux-5.15.0-91-generic-x86_64-with-glibc2.31
# Python : 3.10.13
# Flask : 3.0.0
# google-generativeai : 0.3.2
# langchain : 0.1.0
# langchain-google-genai : 0.0.6
# Created: Jan. 15. 2024
# Author: D.W. SHIN


import os

import google.generativeai as genai
import markdown
from flask import Flask, jsonify, render_template, request
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Initialize Gogole AI Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="gemini-pro")


# pdf 저장폴더
PDF_DN_FOLDER = "./PDF_DN_FOLDER"

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    query = request.form["msg"]

    response = llm.invoke(query)

    return markdown.markdown(response.content, extensions=["extra"])


@app.route("/savePdf", methods=["POST"])
def savePdf():
    result = {
        "RETURN_FLAG": "LOADED",
        "RETURN_INFO": "",
    }
    os.makedirs(PDF_DN_FOLDER, exist_ok=True)

    file = request.files["file"]
    fullFilename = str(file.filename)
    fname, fextension = os.path.splitext(fullFilename)
    if fextension != ".pdf":
        result = {
            "RETURN_FLAG": "FAIL",
            "RETURN_INFO": "IT IS NOT A PDF FILE.",
        }
    else:
        fileFullPath = os.path.join(PDF_DN_FOLDER, fullFilename)
        file.save(fileFullPath)
        result = {
            "RETURN_FLAG": "SUCCESS",
            "RETURN_INFO": "THE FILE WAS SAVED SUCCESSFULLY.",
        }

    return result


@app.route("/chatWithPdf", methods=["GET", "POST"])
def chatWithPdf():
    msg = request.form["msg"]
    fullFilename = request.form["filename"]

    fileFullPath = os.path.join(PDF_DN_FOLDER, fullFilename)

    # TODO
    # 파일이 있는지 확인하고,
    # 파일이 없다면 프론트에 fail 리턴

    # 1. pdf 가져오기
    loader = PyPDFLoader(fileFullPath)
    documents = loader.load_and_split()

    # 2. pdf split하기
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    doc_splits = text_splitter.split_documents(documents)

    # 3. Gemini 임베딩
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. vector store에서 인덱스 생성
    db = FAISS.from_documents(doc_splits, embeddings)

    # 4-1. Test search
    query = embeddings.embed_query(msg)
    docs = db.similarity_search_by_vector(query)
    print(docs[0].page_content)

    # 5. retriever 정의
    retriever = db.as_retriever()

    # 6. prompt 설정
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 7. chain 설정 및 실행
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = retrieval_chain.invoke(msg)

    return markdown.markdown(response, extensions=["extra"])


if __name__ == "__main__":
    app.run()
