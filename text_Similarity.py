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
gptVal = worksheet_list[2].get('J15:J44')

# Gemini 생성된 프롬프트 전체
geminiVal = worksheet_list[2].get('O15:O44')

# type 변환
listAnswerVal = list(answerVal)
listPalmVal = list(palmVal)
listGptVal = list(gptVal)
listGeminiVal = list(geminiVal)
#print(strPalmVal)

# pandas
import pandas as pd
import re
import numpy as np

# answer + palm prompt 
for i in range(len(listAnswerVal)):
    listAnswerVal[i] = list(map(lambda x: re.sub(r'[^\w]', ' ', x), listAnswerVal[i]))

df_answer = pd.DataFrame(listAnswerVal)

for j in range(len(listPalmVal)):
    listPalmVal[i] = list(map(lambda x: re.sub(r'[^\w]', ' ', x), listPalmVal[i]))

df_palm = pd.DataFrame(listPalmVal)

mix_df = pd.concat([df_answer, df_palm], axis=1, ignore_index=True)




# 문장간 유사도 측정
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 데이터프레임 30행을 차례대로 반복하고 (answer prompt, palm prompt)로 출력하면서 형변환하기
#for z in mix_df.index:
#    sentences = f'("{mix_df[0][z]}", "{mix_df[1][z]}")' 
    #print(sentences)  

#sentences_ = np.array(["A photo of a cat sitting on top of a building", "my name is jinhwan. photo of a cat"])
#print(sentences_)
#tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#tfidf_matrix = tfidf_vectorizer.fit_transform(sentences_)
#print(tfidf_matrix.toarray())
    #print(tfidf_vectorizer.get_feature_names_out()) # 각 문장의 단어가 어떻게 토큰화 되었는지 보여준다.

#print(tfidf_matrix[0:1].toarray())
#cosine_similarities = cosine_similarity(tfidf_matrix)
#print(cosine_similarities[0])



# 문장간 유사도 측정 (1차 정리)
for x in mix_df.index:
    sentence = mix_df[0][x], mix_df[1][x]
    sentence = np.array(sentence)
    #sentence_palm = np.array([sentence_palm])
    #print(sentence)

    tfidf_vectorizer = TfidfVectorizer() # stop_words='english'
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentence)
    #tfidf_matrix_palm = tfidf_vectorizer.fit_transform(sentence_palm)

    cosine_similarities = cosine_similarity(tfidf_matrix)
    print(cosine_similarities[0])






#for z in mix_df.index:
#    answer_sentence = np.array([f"{mix_df[0][z]}"])
#    palm_sentence = f'("{mix_df[1][z]}")'
#    #print(answer_sentence)
#
#    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#    #print(tfidf_vectorizer.get_feature_names_out()) # 각 문장의 단어가 어떻게 토큰화 되었는지 보여준다.
#    #print(tfidf_vectorizer.get_stop_words()) # 영어 불용어 사전을 보여준다. 
#
#
#    tfidf_matrix_answer = tfidf_vectorizer.fit_transform(answer_sentence)
#    print(tfidf_matrix_answer[0:].toarray())
#
#
#    #tfidf_matrix_palm = tfidf_vectorizer.fit_transform([palm_sentence])


#print(tfidf_matrix_answer)
#print(tfidf_matrix_palm)
    
    #cosine_similarities = cosine_similarity(tfidf_matrix_answer)
    #print(cosine_similarities[0])
#print(cosine_similarities)




#df.to_csv("D:/jjh/llm_is_all_you_need/palm_text.csv", index=False)