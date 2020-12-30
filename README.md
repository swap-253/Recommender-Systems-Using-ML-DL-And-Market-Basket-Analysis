# Recommender-Systems-Using-ML-DL-And-Market-Basket-Analysis-Using-Apriori
This repository consists of collaborative filtering Recommender systems like Similarity Recommenders, KNN Recommenders, using Apple's Turicreate, A matrix Factorization system from scratch and a Deep Learning Recommender System which learns using embeddings. Besides this Market Basket Analysis using Apriori Algorithm has also been done.

The dataset is a movies ratings dataset from movielens and five colaborative filtering recommenders have been implemented taking care of time of ratings and its affect on the ratings of the movie i.e-temporal information. Three subdivisions of movie lens dataset have been taken.

1)u.user:- user info user_id,age,sex,occupation,zip_code
2)u.data:- containing user_id,movie_id,rating,unix_timestamp
3)u.item:- movie id,movie title,release date,video release date,IMDb URL and (One hot encoding of Genre categories)

Similarity Recommender
KNN Recommender
Apple's Turicreate Recommender(highly specialised and accurate)
A matrix factorization method which has been implemented using Stochastic Gradient Descent, Adam's Optimisation and RMS Prop separately
A Deep Learning Recommender ehich learns embeddings of users and movies and utilises KNN to predict the movie on the basis of user embeddings
To look at movies which are frequently highly rated by an user is an example of Market Basket Analysis. It has been implemented using Apriori Algorithm in this repository.

