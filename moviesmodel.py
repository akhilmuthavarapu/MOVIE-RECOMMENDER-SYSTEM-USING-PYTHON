# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:50:40 2024

@author: akhil
"""

import pandas as pd
movies=pd.read_csv("tmdb_5000_movies.csv")
print(movies.head(10))

print(movies.describe())
print(movies.info())

print(movies.isnull().sum())

print(movies.columns)

movies=movies[["id","original_title","overview","genres"]]
print(movies)

movies["tags"]=movies["overview"]+movies["genres"]
print(movies)

new_data=movies.drop(columns=["overview","genres"])
print(new_data)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=10000,stop_words="english")
print(cv)

vector=cv.fit_transform(new_data["tags"].values.astype("U")).toarray()
print(vector.shape)
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vector)
print(similarity)

distance=sorted(list(enumerate(similarity[2])),reverse=True,key=lambda vector:vector[1])
for i in distance[0:5]:
    print(new_data.iloc[i[0]].original_title)
print("-----------------------------------------------")

#---------------------recommandation system----------  
def recommand(movies):
    index=new_data[new_data["original_title"]==movies].index[0]
    distance=sorted(list(enumerate(similarity[index])),reverse=True,key=lambda vector:vector[1])
    for i in distance[0:5]:
        print(new_data.iloc[i[0]].original_title)
recommand("Iron Man")

#-----------model dumping into pickle file---------------------

import pickle
try:
    pickle.dump(new_data,open("movies_list.pkl","wb"))
    pickle.dump(similarity,open("similarity.pkl","wb"))
    print(pickle.load(open("movies_list.pkl","rb")))
    print("model saved successfully")
except Exception as e:
    print("error in saving the model: {e}")
