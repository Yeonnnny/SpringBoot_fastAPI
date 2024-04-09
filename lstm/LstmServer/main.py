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

import keras
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, Model,load_model

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


class Lstm(BaseModel):
    productNM : str
    productDesc : str


def dataCleansing(data):
    with open("preprocessing_function.pkl",'rb') as f:
        cleansing_func = dill.load(f)
        return cleansing_func(data)


# app 개발
app = FastAPI()

@app.post(path="/items", status_code=201)
def predictLstm(lstm:Lstm):

    print("나 왔다")
    data = pd.DataFrame({'CATALOG_NM':[lstm.productNM],'CATALOG_DESC':[lstm.productDesc]})

    # 데이터 클랜징 
    cleansingData = dataCleansing(data)


    # 토큰화 전처리 파일
    with open("tokenizer_6_6.pkl", "rb") as f:
        tokenizer = dill.load(f)

        model = load_model("01-0.805.hdf5")

        # Tokenize and pad the new data
        sequences_new_desc = tokenizer.texts_to_sequences(cleansingData['CATALOG_DESC'])
        sequences_new_nm = tokenizer.texts_to_sequences(cleansingData['CATALOG_NM'])

        x_new_desc = pad_sequences(sequences_new_desc, maxlen=3727, padding='post')
        x_new_nm = pad_sequences(sequences_new_nm, maxlen=31, padding='post')

        # 새 데이터에 대해 예측
        y_pred = model.predict([x_new_desc, x_new_nm])

        # 임계값
        threshold = 0.8
        lstm_predict = (y_pred > threshold).astype(int)

        # 테스트 데이터에 대한 예측
        lstm_predict_proba = model.predict([x_new_desc, x_new_nm])
        

        result = {'lstm_predict':lstm_predict, 'lstm_predict_proba':np.round(lstm_predict_proba*100,2)}

        return JSONResponse(result)
    # return JSONResponse({'name':"도연"})
    

if __name__=='__main__':
    uvicorn.run(app, host="127.0.0.1", port=8090)
    # 구동 코드 : uvicorn main:app --reload