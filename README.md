## 목적
- Gemini를 이용하여 PDF를 업로드 후, RAG 구현하여 PDF에 있는 내용을 질의



## 구성원
- DONGWEONSHIN
- Jeon0866 JinHwan
- wjseoduq
- rnansfyd


## 프로젝트 기간
- 2023.12.29 ~ 2024.1.5 (약 1주일)


## 실행 방법
1. mv .env.local .env
2. fill in GOOGLE_API_KEY
3. pip install -r requirements.txt
4. flask run



## Google AI Gemini 와 Langchain으로 RAG를 시도 - 1차 시도 (문제점 포함)
1. 2023년 12월에 출시된 Gemini는 Google AI Gemini와 Vertext AI Gemini 버전 2가지가 있습니다.
2. 각각의 버전은 코드가 동일하지 않고, 현재 (2024년 1월 기준) 샘플 코드가 많지 않습니다
3. LangChain도 Gemini를 대응하기 위한 샘플 코드가 적고, 그나마 기본적인 코드만 존재 합니다.
4. 1차 RAG 시도 시 버전은 아래와 같습니다.
```
langchain==0.0.353
google-generativeai==0.3.2
langchain-google-genai==0.0.5
```
5. 1차 RAG 시도시에는 PDF 업로드 후, PDF 내용을 질의 하면 10번중 1번만 응답을 합니다.
6. 1차 파일을 기록을 남기고자 아래의 파일을 접두어 'ver1_'를 추가하여 변경합니다
```
app.py
requirements.txt
```


## 2차 시도 (성공)
1. 2차 시도 버전은 아래와 같습니다.
```
langchain==0.1.0
google-generativeai==0.3.2
langchain-google-genai==0.0.6
```
2. 벡터 스토어를 FAISS로 변경 하고, retriever와 chain을 단순화 하여 수정하였습니다.
3. Google AI Gemini 코드와 Vertext AI Gemini를 인지하고 Google AI Gemini로만 진행 하였습니다.
4. RAG 구성을 완료 하고 PDF 업로드 후 질의 하면, PDF의 내용에 대해 답을 합니다.








