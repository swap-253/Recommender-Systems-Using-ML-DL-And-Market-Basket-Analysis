from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
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
app = Flask(__name__)
new_df=pd.read_csv('rating_title.csv')
#Creating a KNN Model which utilises embeddings as a parameter for prediction of similar movies  
def recommend_movies(embedding):
    distances, indices = clf.kneighbors(embedding.reshape(1, -1),  n_neighbors=10)
    distances,indices = distances.reshape(10,1),indices.reshape(10,1)
    #print(indices)
    df_indices = pd.DataFrame(indices, columns = ['movie_id'])
    #df_distances = pd.DataFrame(distances, columns = ['movie_id'])
    return df_indices.merge(new_df,on='movie_id',how='inner',suffixes=['_u', '_m'])['movie title'].unique()

def predict():
    test_user = input("Enter the User for whom you wanna see top 10 recommendations：", type=NUMBER)
    user_embedding = user_model.predict([test_user]).reshape(1,-1)[0]
    arr=recommend_movies(user_embedding)[:10]
    style(put_text('Top 10 movie recommendations for user id',str(test_user),'are:'), 'color:red')
    #put_text('Top 10 movie recommendations for user id',str(test_user),'are:')  
    for i in range(len(arr)):
      put_text('{0}: {1}'.format(i+1, arr[i]))

"""Hence the nearest movies to the given movie has been recommended by KNN Algorithm"""


app.add_url_rule('/tool', 'webio_view', webio_view(predict),
            methods=['GET', 'POST', 'OPTIONS'])


#if __name__ == '__main__':
    #predict()

app.run(host='localhost', port=80)

#visit http://localhost/tool to open the PyWebIO application.
