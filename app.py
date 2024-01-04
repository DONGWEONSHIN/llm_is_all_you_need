from flask import Flask, render_template, request, jsonify
import markdown


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Gemini, LangChain 설정
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# PaLM2 설정
# from langchain.embeddings import VertexAIEmbeddings

# microsoft/DialoGPT-medium 설정
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Gemini 설정
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name = "gemini-pro")

# LangChain 설정
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# LangChain 설정 - PaLM2
# embeddings = VertexAIEmbeddings()

# LangChain - PDF
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# pdf 저장폴더
PDF_DN_FOLDER = "./PDF_DN_FOLDER"

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')
    # return render_template('new_modefied_chat.html')

@app.route("/chat", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    
    # LangChain 설정
    result = llm.invoke(input)
    llm_result = result.content
    
    
    # PaLM2 설정
    # text_embedding = embeddings.embed_query(input)
    # llm_result = text_embedding
    
    return markdown.markdown(llm_result, extensions=['extra'])
    # return get_Chat_response(input)

# microsoft/DialoGPT-medium
def get_Chat_response(text):

    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


@app.route("/savePdf", methods=["POST"])
def savePdf():
    
    result = {
        "RETURN_FLAG" : "LOADED" ,
        "RETURN_INFO" : ""      
    }
    os.makedirs(PDF_DN_FOLDER, exist_ok=True)
    
    file = request.files['file']
    fullFilename = str(file.filename)
    fname, fextension = os.path.splitext(fullFilename)
    if fextension != '.pdf':
        result = {
            "RETURN_FLAG" : "FAIL" ,
            "RETURN_INFO" : "IT IS NOT A PDF FILE."         
        }
    else :
        fileFullPath = os.path.join(PDF_DN_FOLDER, fullFilename)
        file.save(fileFullPath)
        result = {
            "RETURN_FLAG" : "SUCCESS" ,
            "RETURN_INFO" : "THE FILE WAS SAVED SUCCESSFULLY."      
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
    pages = loader.load_and_split()
    print("##### 1 : pages loaded ##### \n\n", pages[0])
    
    # 2. pdf split하기
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(pages)
    
    # 3. Gemini 임베딩
    gemini_embedding_model = "models/embedding-001"
    embeddings = GoogleGenerativeAIEmbeddings(model=gemini_embedding_model)
    
    # 4. vector store에서 인덱스 생성
    faiss_index = FAISS.from_documents(pages, embeddings)
    
    # 5. retriever 정의
    faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    query = msg
    docs = faiss_index.similarity_search(query)
    print("##### 2 : query #######", docs[0].page_content)
    
    embedding_vector = GoogleGenerativeAIEmbeddings(model=gemini_embedding_model).embed_query(query)
    docs = faiss_index.similarity_search_by_vector(embedding_vector)
    print("##### 3 : embedding_vector #######", docs[0].page_content)
    
    
    
    
    
    
    
    # 샘플 PDF
    samlePDF = "langchain.pdf"
    gemini_embedding_model = "models/embedding-001"
    gemini_task_type = "retrieval_document"
    
    loader = PyPDFLoader(samlePDF)
    pages = loader.load_and_split()
    
    print("##### 1 : page loaded #######", pages[0])
    
    embeddings = GoogleGenerativeAIEmbeddings(model=gemini_embedding_model)
    db = FAISS.from_documents(pages, embeddings)
    
    query = request.form["msg"]
    
    doc_embeddings = GoogleGenerativeAIEmbeddings(model=gemini_embedding_model, task_type=gemini_task_type)
    # doc_vecs = [doc_embeddings,embed_query(q) for q in [query, query_2, answer_1]]
    # docs = db.similarity_search_by_vector(doc_embeddings)
    # print("##### 3 : doc_embeddings #######", doc_vecs)
    
    # return markdown.markdown(docs[0].page_content, extensions=['extra'])
    
    
    
    
    # 샘플로 보내 드립니다. 
    # 나중에 작업 되면 삭제될 내용입니다. 
    input = msg
    result = llm.invoke(input)
    llm_result = result.content
    
    return markdown.markdown(llm_result, extensions=['extra'])




if __name__ == '__main__':
    app.run()
