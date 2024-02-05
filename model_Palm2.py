# OS : Linux-5.15.0-91-generic-x86_64-with-glibc2.31
# Python : 3.10.13 / 3.9.18
# Flask : 3.0.0
# google-cloud-aiplatform : 1.38.1
# langchain-core : 0.1.10
# langchain-google-vertexai : 0.0.1.post1
# FileName : model_PaLM2.py
# Base LLM : Vertex AI Palm2
# Created: Jan. 29. 2024
# Author: D.W. SHIN


import os

import vertexai
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
from langchain_google_vertexai import ChatVertexAI, VertexAI, VertexAIEmbeddings
from markdown import markdown


class Palm2:
    chat_model = ChatVertexAI()

    llm = VertexAI(
        model_name="text-bison@001",
        max_output_tokens=1024,
        temperature=0.2,
        top_p=0.8,
        top_k=40,
        verbose=True,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    def __init__(self, project_id, location):
        self.project_id = project_id
        self.location = location

        vertexai.init(project=self.project_id, location=self.location)

    def chat(self, msg):
        query = msg

        system = "You are a helpful assistant."
        human = "{text}"

        template_messages = [
            SystemMessage(content=system),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(human),
        ]

        prompt_template = ChatPromptTemplate.from_messages(template_messages)

        chain = LLMChain(
            llm=self.chat_model,
            prompt=prompt_template,
            memory=self.memory,
        )

        response = chain.invoke(
            {
                "text": query,
            }
        )

        return markdown(response["text"], extensions=["extra"])

    def chatWithPdf(self, msg, fullFilename, pdf_dn_folder):
        fileFullPath = os.path.join(pdf_dn_folder, fullFilename)

        # Ingest PDF files
        loader = PyPDFLoader(fileFullPath)
        documents = loader.load_and_split()

        # Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        doc_splits = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@001")

        # Vector Store Indexing
        db = FAISS.from_documents(doc_splits, embeddings)

        # Test search
        query = embeddings.embed_query(msg)
        docs = db.similarity_search_by_vector(query)
        print(docs[0].page_content)

        # Retrieval
        retriever = db.as_retriever()

        # Customize the default retrieval prompt template
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
         # 프롬프트 수정(20240205)
        elif fullFilename == "Korean_Ancient_History.pdf":
        
            template = """Answer the question based only on the following context:
            You are a knowledgeable expert on “Korean_Ancient_History” and I am a Korean-speaking student asking you a question about the content of “Korean_Ancient_History”. \
            In response to my questions, in principle, you should only answer what is described in “Korean_Ancient_History” and should not answer what you learned from sources other than “Korean_Ancient_History” or what you inferred yourself. \
            You must answer my questions in the following order. \
            step 1. Interpret my question in English and fully understand the content of the question. \
            step 2. Identify the keywords of my question. \
            step 3. You translate “Korean_Ancient_History” into English and analyze everything in depth. \
            step 4. Find the sentences that answer my question in “Korean_Ancient_History” and summarize the content in English, using the terms described in “Korean_Ancient_History”. \
            step 5. When summarizing, summarize within the number of sentences requested in the question, but if the question does not specify a specific number of sentences, summarize within 2 sentences. \
            step 6. The answers summarized in English are translated into Korean and provided to me. \ 
            {context}

            Question:  {question}
            """   
        # 프롬프트 수정(20240205)
        elif fullFilename == "Labor_law.pdf":       
            template = """Answer the question based only on the following context:
            You are a knowledgeable expert on “Labor_law” and I am a Korean-speaking student asking you a question about the content of “Labor_law”. \
            In response to my questions, in principle, you should only answer what is described in “Labor_law” and should not answer what you learned from sources other than “Labor_law” or what you inferred yourself. \
            You must answer my questions in the following order. \
            step 1. Interpret my question in English and fully understand the content of the question. \
            step 2. Identify the keywords of my question. \
            step 3. You translate “Labor_law” into English and analyze everything in depth. \
            step 4. Find the sentences that answer my question in “Labor_law” and summarize the content in English, using the terms described in “Labor_law”. \
            step 5. When summarizing, summarize within the number of sentences requested in the question, but if the question does not specify a specific number of sentences, summarize within 2 sentences. \
            step 6. The answers summarized in English are translated into Korean and provided to me. \ 
            {context}

            Question:  {question}
            """   
        else:
            template = """Answer the question based only on the following context:
            {context}

            Question: {question}
            """
        prompt = ChatPromptTemplate.from_template(template)

        # Configure RetrievalQA chain
        retrieval_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = retrieval_chain.invoke(msg)

        return markdown(response, extensions=["extra"])
