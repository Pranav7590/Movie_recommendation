import numpy as np
import pandas as pd
import ast

# Loading CSV File
df = pd.read_csv('tmdb_5000_credits.csv')
df.head()

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i["name"])
    return Ldef convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i["name"])
    return L

df["cast"] = df["cast"].apply(convert)

def fetch_director(text):
    L=[]
    for i in ast.literal_eval(text):
        if i["job"]=="Director":
            L.append(i["name"])
    return L

df["crew"]=df["crew"].apply(fetch_director)
df["tag"]=df["cast"]+df["crew"]
df=df.drop(columns=["cast","crew"])
df["tag"]=df["tag"].apply(lambda x:"  ".join(x))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(df['tag']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)
def recommend(movie):
    index = df[df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(df.iloc[i[0]].title)

recommend('Superman')
