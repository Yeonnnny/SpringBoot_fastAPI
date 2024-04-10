from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from starlette.responses import JSONResponse
import pickle
import dill
import numpy as np
import pandas as pd

# 데이터 클랜징
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

# LSTM 모델
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold , train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss,classification_report,roc_curve,auc,confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import string
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy import sparse
seed = 42

import tensorflow as tf
from keras.optimizers import Nadam
from keras.layers import Embedding, LSTM, Dense, Bidirectional,Dropout,GlobalMaxPooling1D
from keras.layers import concatenate,Concatenate,Input,BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.initializers import Orthogonal

# 금지어 유사도
from rapidfuzz import fuzz
import random


# 클래스 객체 
class Lstm(BaseModel):
    prodId : str
    productNM : str
    productDesc : str

# 데이터 클랜징 함수
def preprocess_text(df, col_names=['CATALOG_NM', 'CATALOG_DESC']):
    # html태그 확인
    def has_html_tags(col):
        soup = BeautifulSoup(col, 'html.parser')
        if bool(soup.find()) :
            return soup.get_text()
        else:
            return col
    remove_punct_dict = {ord(punct):' ' for punct in string.punctuation}
    lemmar = WordNetLemmatizer()

    # 문장 입력받음 -> 소문자로 변환 -> 구두점 처리 -> 단어로 토큰 -> 불용어 제거 -> 어근 추출
    def len_normalize(text):
        tokens = nltk.word_tokenize(text.lower().translate(remove_punct_dict))
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return [lemmar.lemmatize(token) for token in tokens]

    # 컬럼 선택
    df = df[col_names]

    # DESC 컬럼에 결측치 존재하는 경우
    if df[col_names[1]].isna().sum()!=0:
        df.fillna('',inplace=True)

    # DESC 컬럼 개행문자 제거
    df[col_names[1]] = df[col_names[1]].str.replace('\r',' ').str.replace('\n',' ').str.replace('\\',' ')

    # DESC 컬럼 태그 확인 및 제거
    df[col_names[1]] = df[col_names[1]].apply(has_html_tags)

    # 태그 지우고 공백인 경우, 상품명으로 대체
    mask = df[col_names[1]] == ''  # boolean Series 생성
    df.loc[mask, col_names[1]] = df.loc[mask, col_names[0]]  # 조건을 만족하는 행만 선택하여 업데이트

    # 불완전 태그 제거
    pattern = re.compile(r'<[a-z]\s+[\d\W\s\w]*\s*=?\s?[0-9]*[\d\W\s\w]')
    df[col_names[1]] = df[col_names[1]].apply(lambda x: re.sub(pattern, ' ', x))

    # 불완전 태그 제거한 후 공백인 경우 NM컬럼 값으로 대체
    df[col_names[1]] = df.apply(lambda x: x[col_names[0]] if x[col_names[1]].isspace() else x[col_names[1]], axis=1)

    # 클랜징
    for i in range(len(col_names)):
        df[col_names[i]] = df[col_names[i]].apply(len_normalize)

    return df



# app 개발
app = FastAPI()

@app.post(path="/items", status_code=201)
def predictLstm(lstm:Lstm):

    data = pd.DataFrame({'CATALOG_NM':[lstm.productNM],'CATALOG_DESC':[lstm.productDesc]})

    # 데이터 클랜징 
    cleansingData = preprocess_text(data)
    print(cleansingData)

    # Step 1: Tokenization
    tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(cleansingData['CATALOG_DESC']+ cleansingData['CATALOG_NM'])
    tokenizer.fit_on_texts(pd.concat([cleansingData['CATALOG_DESC'], cleansingData['CATALOG_NM']]))
    vocab_size = len(tokenizer.word_index) + 1

    sequences_new_desc = tokenizer.texts_to_sequences(cleansingData['CATALOG_DESC'])
    sequences_new_nm = tokenizer.texts_to_sequences(cleansingData['CATALOG_NM'])

    x_new_desc = pad_sequences(sequences_new_desc, maxlen=3727, padding='post')
    x_new_nm = pad_sequences(sequences_new_nm, maxlen=31, padding='post')
    print(x_new_nm)

    # =============== 여기까지는 됨 (모델 로딩이 안돼..)===================

    # 모델 로드 
    # model = load_model("02-0.805.hdf5")
    # model = load_model("02-0.805.hdf5", custom_objects={'Orthogonal': Orthogonal})
    try:
        # 모델 로드 시도
        # model = load_model('02-0.805.hdf5')///
        model = tf.keras.models.load_model('01-0.803.hdf5')
        
        # 새 데이터에 대해 예측
        y_pred = model.predict([x_new_desc, x_new_nm])
        print("==============" + y_pred)

        # 임계값
        threshold = 0.8

        # 새 데이터에 대한 예측 결과
        lstm_predict = (y_pred > threshold).astype(int)
        lstm_predict_proba = model.predict([x_new_desc, x_new_nm])
        print("====lstm_predict : "+lstm_predict)
        print("====lstm_predict_proba : "+lstm_predict_proba)
        
    except Exception as e:
        print(e)


    return JSONResponse({'name':"도연"})
    #=========== 금지어 유사도 =============
    # 정상(1)인 경우 값 반환 / 이상(0)인 경우 금지어유사도까지 확인
    # if lstm_predict==1:
    #     result = {'lstm_predict':lstm_predict, 'lstm_predict_proba':np.round(lstm_predict_proba*100,2)}
    #     return JSONResponse(result)
    # else:
    #     # 상품 데이터, 금지어사전 원본
    #     ori_prohibited_words = pd.read_csv('IPR리스트_231218.csv')

    #     # 금지어사전 하나의 리스트로 변환
    #     prohibited_words_list = ori_prohibited_words['KEYWORD'].str.lower().tolist()

    #     # 결과를 저장할 리스트 생성 (상품 아이디와 lstm결과도 넣어줌)
    #     result = [{'prodId':lstm.prodId, 'lstm_predict':lstm_predict,"lstm_predict_proba":lstm_predict_proba}]

    #     # 상품 설명만 사용 (데이터 클랜징된 데이터)
    #     # 선택한 상품 설명에 대해 처리
    #     # 상품 설명의 문장을 단어로 분리
    #     words_in_desc = cleansingData['CATALOG_DESC'][0]

    #     # 각 상품 설명의 단어와 금지어 사전의 모든 단어 비교
    #     for desc_word in words_in_desc:
    #         for prohibited_word in prohibited_words_list:
    #             similarity_score = fuzz.ratio(prohibited_word, desc_word)
    #             if similarity_score >= 80:
    #                 result.append({
    #                     'Product_Description': desc_word, # desc에서 금지어와 유사한 단어
    #                     'Prohibited_Word': prohibited_word, # 금지어리스트의 단어
    #                     'Similarity_Score': similarity_score # 금지어 유사도
    #                 })
        # return JSONResponse(content=result)

        



    # # 토큰화 전처리 파일
    # with open("tokenizer_6_6.pkl", "rb") as f:
    #     tokenizer = dill.load(f)

    #      model = load_model("01-0.805.hdf5")

    #      # Tokenize and pad the new data
    #      sequences_new_desc = tokenizer.texts_to_sequences(cleansingData['CATALOG_DESC'])
    #      sequences_new_nm = tokenizer.texts_to_sequences(cleansingData['CATALOG_NM'])
    #      print(sequences_new_nm)

    #      x_new_desc = pad_sequences(sequences_new_desc, maxlen=3727, padding='post')
    #      x_new_nm = pad_sequences(sequences_new_nm, maxlen=31, padding='post')

    #      # 새 데이터에 대해 예측
    #      y_pred = model.predict([x_new_desc, x_new_nm])

    #      # 임계값
    #      threshold = 0.8

    #      # 새 데이터에 대한 예측 결과
    #      lstm_predict = (y_pred > threshold).astype(int)
    #      lstm_predict_proba = model.predict([x_new_desc, x_new_nm])
        
    #      result = {'lstm_predict':lstm_predict, 'lstm_predict_proba':np.round(lstm_predict_proba*100,2)}

    #      return JSONResponse(result)
    
    

if __name__=='__main__':
    uvicorn.run(app, host="127.0.0.1", port=8090)
    # 구동 코드 : uvicorn main:app --reload