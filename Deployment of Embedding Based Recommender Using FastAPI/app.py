import uvicorn
from fastapi import FastAPI
from Recommendation import Recomm
import sklearn
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout ,BatchNormalization,Reshape,Dot,Concatenate,Add,Lambda,Input,Embedding
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.models import Model
from keras.layers.recurrent import LSTM
from keras.models import load_model
user_model = load_model('user_model.h5')
clf = pickle.load(open('knn_model_emb.pkl', 'rb'))
app = FastAPI()
new_df=pd.read_csv('rating_title.csv')
# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, Everyone'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Movie Recommendation Engine': f'{name}'}
#Creating a KNN Model which utilises embeddings as a parameter for prediction of similar movies  
def recommend_movies(embedding):
    distances, indices = clf.kneighbors(embedding.reshape(1, -1),  n_neighbors=10)
    distances,indices = distances.reshape(10,1),indices.reshape(10,1)
    #print(indices)
    df_indices = pd.DataFrame(indices, columns = ['movie_id'])
    #df_distances = pd.DataFrame(distances, columns = ['movie_id'])
    return df_indices.merge(new_df,on='movie_id',how='inner',suffixes=['_u', '_m'])['movie title'].unique()

def fun(arr):
    d = dict();
    for i in range(len(arr)):
        d[i+1]=arr[i]
    return d
@app.post('/predict')
def predict(data:Recomm):
    data=data.dict()
    test_user = data['test_user']
    user_embedding = user_model.predict([test_user]).reshape(1,-1)[0]
    arr=recommend_movies(user_embedding)[:10]
    print('Top 10 movie recommendations for user id',str(test_user),'are:')
    #put_text('Top 10 movie recommendations for user id',str(test_user),'are:')  
    a=fun(arr)
    return 'Top 10 movie recommendations for user id '+str(test_user)+' are:',a

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
