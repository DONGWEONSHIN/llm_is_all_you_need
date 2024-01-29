# OS : Linux-5.15.0-91-generic-x86_64-with-glibc2.31
# Python : 3.9.18 / 3.10.13
# Flask : 3.0.0
# google-generativeai : 0.3.2
# langchain : 0.1.0
# langchain-google-genai : 0.0.6
# FileName : app_Gemini.py
# Base LLM : Google AI Gemini
# Created: Jan. 15. 2024
# Author: D.W. SHIN


import os

import google.generativeai as genai
from flask import Flask, jsonify, render_template, request
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from markdown import markdown

# Initialize Gogole AI Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


chat_model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    convert_system_message_to_human=True,
)


llm = GoogleGenerativeAI(
    model="gemini-pro",
    max_output_tokens=1024,
    temperature=0.2,
    top_p=0.8,
    top_k=40,
    verbose=True,
)


# Langchain memory를 사용하여 chat history 구성
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)


# pdf 저장폴더
PDF_DN_FOLDER = "./PDF_DN_FOLDER"


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/chatMuseum")
def chatMuseum():
    return render_template("museum.html")


@app.route("/chatDiffusion")
def chatDiffusion():
    return render_template("diffusion.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    query = request.form["msg"]

    system = "You are a helpful world travel assistant."
    human = "{text}"

    template_messages = [
        SystemMessage(content=system),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(human),
    ]

    prompt_template = ChatPromptTemplate.from_messages(template_messages)

    chain = LLMChain(
        llm=chat_model,
        prompt=prompt_template,
        memory=memory,
    )

    response = chain.invoke(
        {
            "text": query,
        }
    )

    # print(response)
    # print(response["text"])

    return markdown(response["text"], extensions=["extra"])


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
        chunk_size=10000,
        chunk_overlap=1000,
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
    template = ""
    if fullFilename == "stable_diffusion_prompt.pdf":
        template = """Answer the question based only on the following context:
        hello. I am a student in Seoul, South Korea. \
        I would like you to serve as a Stable Diffusion Prompt Engineer. \
        What I want from you is that when I ask a question, you will answer it slowly and according to the procedure. \
        Additionally, if you answer well, we will use the tip to promote you to people around you and give you lots of praise. \
        Please answer in Korean. \
        If you answer in English, please translate your answer into Korean \
        If there is content that is not in the pdf, please reply, "I don't know. Please only ask questions about what is in the pdf." \
        --- \
        
        --- Output prompt example: \
        kor_input: "korea" \
        eng_output: "english" \
        output_prompt: "english" \
        negative prompt: "english" \
        negative prompt examples: paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, extra fingers, fewer fingers, strange fingers, bad hand, bad anatomy, fused fingers, missing leg, mutated hand, malformed limbs, missing feet \
        {context}

        Question: 
        1. 기본적으로 한국어로 대답해주세요. 그리고 프롬프트는 영어로 대답해주세요. \
        {question}
        """
    elif fullFilename == "white_porcelain_square_bottle.pdf":
        template = """Answer the question based only on the following context:
        hello. I am a student in Seoul, South Korea. \
        I would like you to become a museum curator who explains information on artifacts, including ceramics and white porcelain.\
        What I want from you is that when I ask a question, you will answer it slowly and according to the procedure. \
        Additionally, if you answer well, we will use the tip to promote you to people around you and give you lots of praise. \
        Please answer in Korean. \
        If you answer in English, please translate your answer into Korean \
        If there is content that is not in the pdf, please reply, "I don't know. Please only ask questions about what is in the pdf." \
        {context}

        Question: 
        1. 한국어로만 대답해주세요. \
        2. pdf 내부에 없는 내용은 답할 수 없습니다. pdf와 관련된 질문이 아니라면 답변하지 마세요. \
        {question}
        """
    else:
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

    return markdown(response, extensions=["extra"])


if __name__ == "__main__":
    app.run()
