# Recommender-Systems-Using-ML-DL-And-Market-Basket-Analysis-Using-Apriori
This repository consists of collaborative filtering Recommender systems like Similarity Recommenders, KNN Recommenders, using Apple's Turicreate, A matrix Factorization system from scratch and a Deep Learning Recommender System which learns using embeddings. Besides this Market Basket Analysis using Apriori Algorithm has also been done. Deployment of Embedding Based Recommender Systems have also been done on local host using Streamlit, Fast API and PyWebIO.
## References
1) **Krish Naik Tutorial For KNN**:-(https://github.com/krishnaik06/Recommendation_complete_tutorial/tree/master/KNN%20Movie%20Recommendation) 
2) **Matrix Factorization Recommender System inspired from Netflix Prize**:- (https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
3) **Analytics Vidhya Tutorial For Recommendation Engine**:-(https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/)
### About Dataset 
The dataset is a movies ratings dataset from movielens and five colaborative filtering recommenders have been implemented taking care of time of ratings and its affect on the ratings of the movie i.e-temporal information. Three subdivisions of movie lens dataset have been taken.
<br>
1)__u.user__:- user info user_id,age,sex,occupation,zip_code
<br>
2)**u.data**:- containing user_id,movie_id,rating,unix_timestamp
<br>
3)**u.item**:- movie id,movie title,release date,video release date,IMDb URL and (One hot encoding of Genre categories)
<br>
<br>
<br>
### Recommenders Used
**1)Similarity Recommender
<br>
2)KNN Recommender
<br>
3)Plot based Content Recommender
<br>
4)cast_genre_director based Content Recommender
<br>
5)Apple's Turicreate Recommender**(highly specialised and accurate)
<br>
**6)A matrix factorization method which has been implemented using Stochastic Gradient Descent, Adam's Optimisation and RMS Prop separately.
<br>
7)A Deep Learning Recommender** which learns embeddings of users and movies and utilises KNN to predict the movie on the basis of user embeddings
<br>
So deployment of Deep Learning Embedding Model has been done using **Fast APIs, PyWebIO and StreamLit platforms** where an user id has to be entered and his preferences are displayed as predicted by the model.
<br>



## Using StreamLit
### Before Entering userid
![strem1](https://user-images.githubusercontent.com/75975560/117648421-cf471c00-b1ab-11eb-85a5-b11dc7298fad.png)
### After Entering userid
![strem2](https://user-images.githubusercontent.com/75975560/117648485-e423af80-b1ab-11eb-8a16-380c6452f875.png)

## Using PyWebIO
### Before Entering userid
![pyweb1](https://user-images.githubusercontent.com/75975560/117648582-04536e80-b1ac-11eb-84bf-6fa545e4f145.png)
### After Entering userid
![pyweb2](https://user-images.githubusercontent.com/75975560/117648603-0b7a7c80-b1ac-11eb-8dd4-e6ff61203507.png)

## Using FastAPI
### The First Screen
![fast1](https://user-images.githubusercontent.com/75975560/117650215-133b2080-b1ae-11eb-9cf3-9cdf525501f8.png)
### Before Executing Get Name
![fast2](https://user-images.githubusercontent.com/75975560/117650219-1504e400-b1ae-11eb-953a-4e303c4f0905.png)
### After Executing Get Name
![fast3](https://user-images.githubusercontent.com/75975560/117650241-19310180-b1ae-11eb-988d-5a2b2c9163d9.png)
### Before Executing Predict
![fast4](https://user-images.githubusercontent.com/75975560/117650260-1d5d1f00-b1ae-11eb-977e-33516dfeca16.png)
### After Executing Predict
![fast5](https://user-images.githubusercontent.com/75975560/117650285-28b04a80-b1ae-11eb-93e7-1e2ff4228327.png)

To look at movies which are frequently highly rated by users is an example of **Market Basket Analysis**. It has been implemented using **Apriori Algorithm** in this repository.


