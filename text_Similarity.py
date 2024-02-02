# OS : Window 11 Pro 
# Python : 3.9.18
# gspread : 6.0.0
# scikit-learn : 1.4.0
# pandas : 2.2.0
# FileName : text_Silmilarity
# Created : Feb. 1. 2024
# Author : J.H. JEON
# example : Google spread sheet API(GCP)를 크롤링하여 stable diffusion의 입력 값으로 들어갈 정답 프롬프트와 생성된 프롬프트 문장의 유사도를 비교하는 코드.
# reference : https://meir.tistory.com/162

import gspread

filename = "C:/Users/SBA/AppData/Roaming/gspread/proj-team-3-0387d2cb9dba.json"

gc = gspread.service_account(filename)
sh = gc.open("History_Evaluation")

# 전체 sheets 확인
worksheet_list = sh.worksheets()
# print(worksheet_list[2]) # <Worksheet 'diffusion' id:501443638>

# 값 가져오기
val= worksheet_list[2].cell(1, 1).value

# 정답 프롬프트 전체 가져오기
answerVal = worksheet_list[2].get('D15:D44') # 30행 # <class 'gspread.worksheet.ValueRange'>

# PaLM2 생성된 프롬프트 전체
palmVal = worksheet_list[2].get('E15:E44') # 30행

# GPT4 생성된 프롬프트 전체
gptVal = worksheet_list[2].get('K15:K44')

# Gemini 생성된 프롬프트 전체
geminiVal = worksheet_list[2].get('Q15:Q44')

# type 변환
listAnswerVal = list(answerVal)
listPalmVal = list(palmVal)
listGptVal = list(gptVal)
listGeminiVal = list(geminiVal)


# pandas
import re
import numpy as np
import pandas as pd

# 특수문자 제거 
def remove_characters(listmodelVal):
    for idx in range(len(listmodelVal)):
        listmodelVal[idx] = list(map(lambda x: re.sub(r'[^\w]', ' ', x), listmodelVal[idx]))
    
    df_model = pd.DataFrame(listmodelVal)
    return df_model

# 데이터 합치기
def df_mix(df_answer, df_model):
    df_mixDf = pd.concat([df_answer, df_model], axis=1, ignore_index=True)
    return df_mixDf


# 문장간 유사도 측정
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def sentences_similarities(df_mixDf, stop_words=None):
    for idx in df_mixDf.index:
        sentences = df_mixDf[0][idx], df_mixDf[1][idx]
        sentences = np.array(sentences)

        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        cosine_similarities = cosine_similarity(tfidf_matrix) # cosine similarity
        print(cosine_similarities[0])


df_answer = remove_characters(listAnswerVal)
df_palm = remove_characters(listPalmVal)
df_mixPalm = df_mix(df_answer, df_palm)

df_gpt = remove_characters(listGptVal)
df_mixGpt = df_mix(df_answer, df_gpt)

df_gemini = remove_characters(listGeminiVal)
df_mixGemini = df_mix(df_answer, df_gemini)


if __name__ == "__main__":
    #sentences_similarities(df_mixPalm)
    #print("---")
    #sentences_similarities(df_mixGpt)
    #print("---")
    sentences_similarities(df_mixGemini)
